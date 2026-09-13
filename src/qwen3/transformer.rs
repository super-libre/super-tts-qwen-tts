//! Decoder-only transformer shared by the Qwen3-TTS talker, its code predictor and the
//! speech-tokenizer decoder. The speech-tokenizer encoder borrows its attention layer, see
//! [`Attention`].
//!
//! The three stacks are all Qwen3-style (pre-norm, SwiGLU MLP, RoPE, grouped-query attention)
//! and only differ in a few knobs captured by [`TransformerConfig`]: whether queries/keys get a
//! per-head RMS normalization, whether the residual branches carry a learnt layer scale and
//! whether attention is restricted to a sliding window.
//!
//! The talker uses multimodal RoPE (`mrope`) in the reference implementation. Text-to-speech only
//! ever feeds it text/codec tokens, so the three position streams are identical and the result is
//! exactly the standard 1D rotary embedding implemented here.
//!
//! The parameters live in [`Transformer`], everything that changes while generating (the rotary
//! tables and the per-layer key/value caches) lives in [`TransformerState`], so a stack can be
//! shared while each generation keeps its own cache.

use burn::module::Param;
use burn::nn::{Linear, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{DType, IndexingUpdateOp};

use crate::qwen3::config::Activation;

#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub hidden_act: Activation,
    pub attention_bias: bool,
    /// Qwen3 per-head RMS normalization of the queries and keys.
    pub qk_norm: bool,
    /// Learnt per-channel scaling of the residual branches.
    pub layer_scale: bool,
    /// Restrict every query to the `sliding_window` most recent keys (itself included).
    pub sliding_window: Option<usize>,
}

/// Precomputed rotary tables, `(max_position_embeddings, head_dim)`.
///
/// The rotation pairs channel `i` with `i + D / 2`, the convention of the reference
/// implementation: `out = x * cos + rotate_half(x) * sin` where `rotate_half` swaps the two
/// halves and negates the first. The tables are stored full width, with the sign of the rotated
/// half folded into `sin`, so that the rotation is one gather along the channels followed by
/// arithmetic the fusion folds into the surrounding kernels. Narrowing the halves and
/// concatenating them back costs six kernels per tensor instead.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos: Tensor<2>,
    sin: Tensor<2>,
    /// The channel each output channel is rotated with: `i + D / 2` for the first half, `i - D / 2`
    /// for the second.
    rotate: Tensor<1, Int>,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, theta: f64, max_seq_len: usize, dtype: DType, device: &Device) -> Self {
        let half = dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1f32 / theta.powf(2. * i as f64 / dim as f64) as f32)
            .collect();
        let mut cos = Vec::with_capacity(max_seq_len * dim);
        let mut sin = Vec::with_capacity(max_seq_len * dim);
        for pos in 0..max_seq_len {
            let angles: Vec<f32> = inv_freq.iter().map(|freq| pos as f32 * freq).collect();
            // Both halves see the same angle; the first half subtracts its rotated partner.
            cos.extend(angles.iter().map(|theta| theta.cos()));
            cos.extend(angles.iter().map(|theta| theta.cos()));
            sin.extend(angles.iter().map(|theta| -theta.sin()));
            sin.extend(angles.iter().map(|theta| theta.sin()));
        }
        let rotate: Vec<i64> = (half..dim).chain(0..half).map(|i| i as i64).collect();
        let shape = [max_seq_len, dim];
        Self {
            cos: Tensor::<2>::from_data(TensorData::new(cos, shape), device).cast(dtype),
            sin: Tensor::<2>::from_data(TensorData::new(sin, shape), device).cast(dtype),
            rotate: Tensor::<1, Int>::from_data(TensorData::new(rotate, [dim]), device),
        }
    }

    /// The rows of the tables for `seq_len` positions starting at `offset`, broadcastable over
    /// (B, H, L, D). Sliced once per forward and shared by every layer.
    fn slice(&self, offset: usize, seq_len: usize) -> RotarySlice {
        let dim = self.cos.dims()[1];
        RotarySlice {
            cos: self
                .cos
                .clone()
                .narrow(0, offset, seq_len)
                .reshape([1, 1, seq_len, dim]),
            sin: self
                .sin
                .clone()
                .narrow(0, offset, seq_len)
                .reshape([1, 1, seq_len, dim]),
            rotate: self.rotate.clone(),
        }
    }

    /// The rows of the tables for the single position held by `pos`, looked up on the device
    /// so that the pass does not depend on the position's value.
    fn gather(&self, pos: &Tensor<1, Int>) -> RotarySlice {
        let dim = self.cos.dims()[1];
        RotarySlice {
            cos: self
                .cos
                .clone()
                .select(0, pos.clone())
                .reshape([1, 1, 1, dim]),
            sin: self
                .sin
                .clone()
                .select(0, pos.clone())
                .reshape([1, 1, 1, dim]),
            rotate: self.rotate.clone(),
        }
    }
}

/// The rotary tables narrowed to the positions of one forward pass.
#[derive(Debug, Clone)]
pub struct RotarySlice {
    cos: Tensor<4>,
    sin: Tensor<4>,
    rotate: Tensor<1, Int>,
}

impl RotarySlice {
    /// Applies RoPE to `xs` (B, H, L, D).
    fn apply(&self, xs: Tensor<4>) -> Tensor<4> {
        let rotated = xs.clone().select(3, self.rotate.clone());
        xs * self.cos.clone() + rotated * self.sin.clone()
    }
}

/// Keys and values of a single attention layer.
#[derive(Debug, Clone)]
pub enum KvCache {
    /// Grown one call at a time by concatenation, for sequences of any length.
    Growing(Option<(Tensor<4>, Tensor<4>)>),
    /// Preallocated to a fixed capacity, (B, Hkv, capacity, D), and written in place at the
    /// position of each call, so that the buffers of a forward pass stay where they are: what a
    /// captured graph replays against.
    Fixed(Option<(Tensor<4>, Tensor<4>)>),
}

impl KvCache {
    /// Stores the keys and values `k` and `v` (B, Hkv, L, D) of the tokens at positions
    /// `pos..pos + L` and returns everything cached so far, (B, Hkv, pos + L, D).
    fn append(&mut self, k: Tensor<4>, v: Tensor<4>, pos: usize) -> (Tensor<4>, Tensor<4>) {
        match self {
            Self::Growing(kv) => {
                let (k, v) = match kv.take() {
                    Some((prev_k, prev_v)) => (
                        Tensor::cat(vec![prev_k, k], 2),
                        Tensor::cat(vec![prev_v, v], 2),
                    ),
                    None => (k, v),
                };
                *kv = Some((k.clone(), v.clone()));
                (k, v)
            }
            Self::Fixed(slot) => {
                let (cache_k, cache_v) = slot.take().expect("a fixed cache is never empty");
                let [b, h, capacity, d] = cache_k.dims();
                let len = k.dims()[2];
                assert!(
                    pos + len <= capacity,
                    "a cache of {capacity} positions cannot hold positions {pos}..{}",
                    pos + len
                );
                // Nothing else refers to the cache buffers at this point, so the assignments
                // write into them rather than into copies, and the buffers never move.
                let cache_k = cache_k.slice_assign([0..b, 0..h, pos..pos + len, 0..d], k);
                let cache_v = cache_v.slice_assign([0..b, 0..h, pos..pos + len, 0..d], v);
                let kv = (
                    cache_k.clone().narrow(2, 0, pos + len),
                    cache_v.clone().narrow(2, 0, pos + len),
                );
                *slot = Some((cache_k, cache_v));
                kv
            }
        }
    }

    /// Stores the keys and values `k` and `v` (B, Hkv, 1, D) of one token at the position held
    /// by `pos` and returns the whole cache, (B, Hkv, capacity, D), for the caller to mask. Only
    /// a fixed cache can be written where a tensor says.
    fn append_at(
        &mut self,
        k: Tensor<4>,
        v: Tensor<4>,
        pos: &Tensor<1, Int>,
    ) -> (Tensor<4>, Tensor<4>) {
        let Self::Fixed(slot) = self else {
            panic!("a growing cache cannot be written at a position held by a tensor")
        };
        let (cache_k, cache_v) = slot.take().expect("a fixed cache is never empty");
        let cache_k = cache_k.select_assign(2, pos.clone(), k, IndexingUpdateOp::Assign);
        let cache_v = cache_v.select_assign(2, pos.clone(), v, IndexingUpdateOp::Assign);
        let kv = (cache_k.clone(), cache_v.clone());
        *slot = Some((cache_k, cache_v));
        kv
    }

    /// Moves a fixed cache to buffers of `capacity` positions, keeping its content.
    fn grow(&mut self, capacity: usize) {
        let Self::Fixed(slot) = self else {
            panic!("only a fixed cache has a capacity")
        };
        let (cache_k, cache_v) = slot.take().expect("a fixed cache is never empty");
        let [b, h, old, d] = cache_k.dims();
        assert!(
            capacity >= old,
            "a cache cannot shrink from {old} to {capacity} positions"
        );
        let grow = |cache: Tensor<4>| {
            let (device, dtype) = (cache.device(), cache.dtype());
            Tensor::zeros([b, h, capacity, d], (&device, dtype))
                .slice_assign([0..b, 0..h, 0..old, 0..d], cache)
        };
        *slot = Some((grow(cache_k), grow(cache_v)));
    }

    fn reset(&mut self) {
        // A fixed cache is overwritten from position 0 by the next sequence.
        if let Self::Growing(kv) = self {
            *kv = None;
        }
    }
}

/// Where a forward pass puts its tokens in the cache and how far it attends.
#[derive(Clone, Copy)]
pub(crate) enum Step<'a> {
    /// Tokens at `offset..offset + L`, decided when the pass is built.
    Static { offset: usize },
    /// One token at the position held by `pos`, attending to the whole fixed cache through the
    /// additive `mask` (1, 1, 1, capacity), which hides the positions after the token.
    Dynamic {
        pos: &'a Tensor<1, Int>,
        mask: &'a Tensor<4>,
    },
}

/// Everything a [`Transformer`] needs on top of its parameters to run a generation.
#[derive(Debug, Clone)]
pub struct TransformerState {
    rotary_emb: RotaryEmbedding,
    caches: Vec<KvCache>,
    /// `0..capacity`, for a fixed cache: what a token's position is compared with to mask the
    /// cache positions after it.
    positions: Option<Tensor<1, Int>>,
    sliding_window: Option<usize>,
    dtype: DType,
    device: Device,
}

impl TransformerState {
    pub fn new(cfg: &TransformerConfig, dtype: DType, device: &Device) -> Self {
        Self {
            rotary_emb: RotaryEmbedding::new(
                cfg.head_dim,
                cfg.rope_theta,
                cfg.max_position_embeddings,
                dtype,
                device,
            ),
            caches: vec![KvCache::Growing(None); cfg.num_hidden_layers],
            positions: None,
            sliding_window: cfg.sliding_window,
            dtype,
            device: device.clone(),
        }
    }

    /// Like [`new`](Self::new), with caches preallocated for `batch` sequences of at most
    /// `capacity` positions and written in place, see [`KvCache::Fixed`].
    pub fn new_fixed(
        cfg: &TransformerConfig,
        batch: usize,
        capacity: usize,
        dtype: DType,
        device: &Device,
    ) -> Self {
        let mut state = Self::new(cfg, dtype, device);
        let shape = [batch, cfg.num_key_value_heads, capacity, cfg.head_dim];
        state.caches = (0..cfg.num_hidden_layers)
            .map(|_| {
                KvCache::Fixed(Some((
                    Tensor::zeros(shape, (device, dtype)),
                    Tensor::zeros(shape, (device, dtype)),
                )))
            })
            .collect();
        state.positions = Some(Tensor::arange(0..capacity as i64, device));
        state
    }

    /// The number of positions of the fixed caches.
    pub fn capacity(&self) -> usize {
        self.positions
            .as_ref()
            .expect("only a fixed cache has a capacity")
            .dims()[0]
    }

    /// Moves the fixed caches to buffers of `capacity` positions, keeping their content.
    pub fn grow(&mut self, capacity: usize) {
        for cache in self.caches.iter_mut() {
            cache.grow(capacity);
        }
        self.positions = Some(Tensor::arange(0..capacity as i64, &self.device));
    }

    /// Additive attention mask (1, 1, 1, capacity) for one query at the position held by `pos`
    /// attending to a fixed cache: `-1e9` after the query.
    fn mask(&self, pos: &Tensor<1, Int>) -> Tensor<4> {
        let positions = self
            .positions
            .as_ref()
            .expect("only a fixed cache is attended through a mask");
        let capacity = positions.dims()[0];
        (positions.clone() - pos.clone())
            .greater_elem(0)
            .float()
            .mul_scalar(-1e9)
            .reshape([1, 1, 1, capacity])
    }

    /// Forgets the cached keys and values, to start a new sequence at offset 0.
    pub fn reset(&mut self) {
        for cache in self.caches.iter_mut() {
            cache.reset()
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// What a forward pass over `seq_len` tokens starting at `offset` takes from the state: the
    /// attention mask, the rotary rows of its positions and the per-layer caches.
    pub(crate) fn step(
        &mut self,
        seq_len: usize,
        offset: usize,
    ) -> (Option<Tensor<4, Bool>>, RotarySlice, &mut [KvCache]) {
        let mask = self.attention_mask(seq_len, offset);
        let rotary = self.rotary_emb.slice(offset, seq_len);
        (mask, rotary, self.caches.as_mut_slice())
    }

    /// Additive attention mask of shape (1, 1, seq_len, offset + seq_len), `-inf` on the
    /// positions a query must not attend to.
    ///
    /// The reference implementation lets a query attend to the keys `j` such that
    /// `i - j < w`, i.e. the window covers `w` keys including the query itself.
    fn attention_mask(&self, seq_len: usize, offset: usize) -> Option<Tensor<4, Bool>> {
        // Without a sliding window the causal flag of the attention op is all that is needed.
        let window = self.sliding_window?.saturating_sub(1);
        let kv_len = offset + seq_len;
        let mask: Vec<bool> = (0..seq_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    let causal = j <= i + offset;
                    // Within the window iff `(i + offset) - j <= w`, rearranged to
                    // `j + w >= i + offset` to stay in `usize`.
                    let in_window = j + window >= i + offset;
                    // `true` marks the positions the query must not attend to.
                    !(causal && in_window)
                })
            })
            .collect();
        Some(Tensor::<4, Bool>::from_data(
            TensorData::new(mask, [1, 1, seq_len, kv_len]),
            &self.device,
        ))
    }
}

#[derive(Module, Debug)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    #[module(skip)]
    act_fn: Activation,
}

impl Mlp {
    fn init(cfg: &TransformerConfig, device: &Device) -> Self {
        let linear = |d_in, d_out| {
            crate::qwen3::linear_config(d_in, d_out)
                .with_bias(false)
                .init(device)
        };
        Self {
            gate_proj: linear(cfg.hidden_size, cfg.intermediate_size),
            up_proj: linear(cfg.hidden_size, cfg.intermediate_size),
            down_proj: linear(cfg.intermediate_size, cfg.hidden_size),
            act_fn: cfg.hidden_act,
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let lhs = self.act_fn.forward(self.gate_proj.forward(xs.clone()));
        let rhs = self.up_proj.forward(xs);
        self.down_proj.forward(lhs * rhs)
    }
}

/// Repeats each key/value head `n_rep` times, the grouped-query attention expansion.
fn repeat_kv(xs: Tensor<4>, n_rep: usize) -> Tensor<4> {
    if n_rep == 1 {
        return xs;
    }
    let [b, h, l, d] = xs.dims();
    xs.unsqueeze_dim::<5>(2)
        .expand([b, h, n_rep, l, d])
        .reshape([b, h * n_rep, l, d])
}

#[derive(Module, Debug)]
pub(crate) struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
}

impl Attention {
    pub(crate) fn init(cfg: &TransformerConfig, device: &Device) -> Self {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let linear = |d_in, d_out| {
            crate::qwen3::linear_config(d_in, d_out)
                .with_bias(cfg.attention_bias)
                .init(device)
        };
        let norm = || {
            cfg.qk_norm.then(|| {
                RmsNormConfig::new(head_dim)
                    .with_epsilon(cfg.rms_norm_eps)
                    .init(device)
            })
        };
        Self {
            q_proj: linear(cfg.hidden_size, num_heads * head_dim),
            k_proj: linear(cfg.hidden_size, num_kv_heads * head_dim),
            v_proj: linear(cfg.hidden_size, num_kv_heads * head_dim),
            o_proj: linear(num_heads * head_dim, cfg.hidden_size),
            q_norm: norm(),
            k_norm: norm(),
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
        }
    }

    pub(crate) fn forward(
        &self,
        xs: Tensor<3>,
        mask: Option<&Tensor<4, Bool>>,
        rotary: &RotarySlice,
        cache: &mut KvCache,
        step: Step<'_>,
    ) -> Tensor<3> {
        let [b, l, _] = xs.dims();

        // (B, L, H * D) -> (B, H, L, D)
        let split =
            |xs: Tensor<3>, heads: usize| xs.reshape([b, l, heads, self.head_dim]).swap_dims(1, 2);
        let q = split(self.q_proj.forward(xs.clone()), self.num_heads);
        let k = split(self.k_proj.forward(xs.clone()), self.num_kv_heads);
        let v = split(self.v_proj.forward(xs), self.num_kv_heads);

        // Qwen3 normalizes every head of the queries and keys, over their last dimension.
        let q = match &self.q_norm {
            Some(norm) => norm.forward(q),
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(k),
            None => k,
        };

        let q = rotary.apply(q);
        let k = rotary.apply(k);
        let (k, v) = match step {
            Step::Static { offset } => cache.append(k, v, offset),
            Step::Dynamic { pos, .. } => cache.append_at(k, v, pos),
        };

        let out = if let Step::Dynamic { mask, .. } = step {
            self.decode(q, k, v, Some(mask))
        } else if l == 1 && mask.is_none() {
            self.decode(q, k, v, None)
        } else {
            let k = repeat_kv(k, self.num_kv_groups);
            let v = repeat_kv(v, self.num_kv_groups);

            // One fused kernel rather than a matmul/softmax/matmul chain. Leaving `scale` unset
            // keeps the default `1/sqrt(head_dim)` and the flash-attention path; the causal
            // flag aligns on the bottom right corner, which is what a query attending to a
            // whole key/value cache needs.
            let options = AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: mask.is_none(),
            };
            attention(q, k, v, mask.cloned(), None, options)
                .swap_dims(1, 2)
                .reshape([b, l, self.num_heads * self.head_dim])
        };
        self.o_proj.forward(out)
    }

    /// Attention of a single query (B, H, 1, D) over the cached keys and values (B, Hkv, L, D).
    ///
    /// The attention op has no kernel for one query and falls back to a dozen small ones
    /// (transposes, two matmuls, a scaled masked softmax) on top of the copies expanding the
    /// keys and values to every head. A last query attends to every cached key, so no mask is
    /// needed, and grouping the query heads by the key/value head they share turns the
    /// expansion into a reshape: `(B, Hkv, G, D) @ (B, Hkv, D, L)` scores every head at once.
    /// The result comes out in the (B, Hkv, G, D) order, which is the head order. `mask` is
    /// added to the scores, (1, 1, 1, L) in f32.
    fn decode(
        &self,
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        mask: Option<&Tensor<4>>,
    ) -> Tensor<3> {
        let [b, _, _, _] = q.dims();
        let dtype = q.dtype();
        let q = q.reshape([b, self.num_kv_heads, self.num_kv_groups, self.head_dim]);
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        // The softmax runs in f32, as the attention kernels do.
        let scores = q.matmul(k.transpose()).mul_scalar(scale).cast(DType::F32);
        let scores = match mask {
            Some(mask) => scores + mask.clone(),
            None => scores,
        };
        let probs = burn::tensor::activation::softmax(scores, 3).cast(dtype);
        probs
            .matmul(v)
            .reshape([b, 1, self.num_heads * self.head_dim])
    }
}

/// Learnt per-channel scaling of a residual branch.
#[derive(Module, Debug)]
pub(crate) struct LayerScale {
    scale: Param<Tensor<1>>,
}

impl LayerScale {
    pub(crate) fn init(size: usize, device: &Device) -> Self {
        Self {
            scale: Param::from_tensor(Tensor::ones([size], device)),
        }
    }

    pub(crate) fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        xs * self.scale.val().unsqueeze()
    }
}

#[derive(Module, Debug)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    self_attn_layer_scale: Option<LayerScale>,
    mlp_layer_scale: Option<LayerScale>,
}

impl DecoderLayer {
    fn init(cfg: &TransformerConfig, device: &Device) -> Self {
        let norm = || {
            RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device)
        };
        let layer_scale = || {
            cfg.layer_scale
                .then(|| LayerScale::init(cfg.hidden_size, device))
        };
        Self {
            self_attn: Attention::init(cfg, device),
            mlp: Mlp::init(cfg, device),
            input_layernorm: norm(),
            post_attention_layernorm: norm(),
            self_attn_layer_scale: layer_scale(),
            mlp_layer_scale: layer_scale(),
        }
    }

    fn forward(
        &self,
        xs: Tensor<3>,
        mask: Option<&Tensor<4, Bool>>,
        rotary: &RotarySlice,
        cache: &mut KvCache,
        step: Step<'_>,
    ) -> Tensor<3> {
        let residual = xs.clone();
        let hidden =
            self.self_attn
                .forward(self.input_layernorm.forward(xs), mask, rotary, cache, step);
        let hidden = match &self.self_attn_layer_scale {
            Some(layer_scale) => layer_scale.forward(hidden),
            None => hidden,
        };
        let xs = residual + hidden;

        let hidden = self
            .mlp
            .forward(self.post_attention_layernorm.forward(xs.clone()));
        let hidden = match &self.mlp_layer_scale {
            Some(layer_scale) => layer_scale.forward(hidden),
            None => hidden,
        };
        xs + hidden
    }
}

/// A stack of decoder layers followed by the final RMS normalization.
///
/// The stack works on input embeddings rather than token ids: the various Qwen3-TTS front-ends
/// sum text, codec and speaker embeddings before feeding it.
#[derive(Module, Debug)]
pub struct Transformer {
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
}

impl Transformer {
    pub fn init(cfg: &TransformerConfig, device: &Device) -> Self {
        Self {
            layers: (0..cfg.num_hidden_layers)
                .map(|_| DecoderLayer::init(cfg, device))
                .collect(),
            norm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device),
        }
    }

    /// Runs the stack on `xs` (B, L, hidden) whose first token sits at position `offset` and
    /// returns the normalized hidden states (B, L, hidden). Keys and values are appended to the
    /// per-layer caches of `state`, so consecutive calls must use increasing offsets.
    pub fn forward(&self, xs: Tensor<3>, offset: usize, state: &mut TransformerState) -> Tensor<3> {
        let [_b, seq_len, _] = xs.dims();
        let mask = state.attention_mask(seq_len, offset);
        let rotary = state.rotary_emb.slice(offset, seq_len);
        let mut xs = xs;
        let step = Step::Static { offset };
        for (layer, cache) in self.layers.iter().zip(state.caches.iter_mut()) {
            xs = layer.forward(xs, mask.as_ref(), &rotary, cache, step);
        }
        self.norm.forward(xs)
    }

    /// Runs the stack on one token `xs` (B, 1, hidden) at the position held by `pos`, which
    /// is not read: the pass is the same for every position, so it can be captured once and
    /// replayed. Needs the fixed caches of [`TransformerState::new_fixed`]; the token is
    /// written at its position and attends to the whole cache, masked after its position.
    pub fn forward_at(
        &self,
        xs: Tensor<3>,
        pos: &Tensor<1, Int>,
        state: &mut TransformerState,
    ) -> Tensor<3> {
        let rotary = state.rotary_emb.gather(pos);
        let mask = state.mask(pos);
        let step = Step::Dynamic { pos, mask: &mask };
        let mut xs = xs;
        for (layer, cache) in self.layers.iter().zip(state.caches.iter_mut()) {
            xs = layer.forward(xs, None, &rotary, cache, step);
        }
        self.norm.forward(xs)
    }
}
