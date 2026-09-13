//! Qwen3-TTS-Tokenizer-12Hz: the neural audio codec used by Qwen3-TTS.
//!
//! The decoder turns 12.5 Hz frames of 16 codebook indices into 24 kHz audio in four stages:
//! a split residual vector quantizer producing latents, a sliding-window transformer, two
//! ConvNeXt upsampling stages and a SnakeBeta/transposed convolution vocoder (each frame yields
//! 1920 samples).
//!
//! The encoder is the Mimi encoder of Kyutai as `transformers` packages it (`MimiModel`): a
//! SEANet convolutional stack, a transformer, a downsampling convolution and a split residual
//! vector quantizer. It is only needed to compute the codes of a reference recording for voice
//! cloning, see [`SpeechTokenizer::encode`].

use burn::module::Param;
use burn::nn::conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear};
use burn::prelude::*;
use burn::tensor::activation::{elu, gelu};
use burn::tensor::ops::PadMode;
use burn_store::{KeyRemapper, ModuleSnapshot, SafetensorsStore};

use crate::qwen3::config::{DecoderConfig, EncoderConfig, SpeechTokenizerConfig};
use crate::qwen3::transformer::{
    Attention, KvCache, LayerScale, RotarySlice, Step, Transformer, TransformerConfig,
    TransformerState,
};

impl DecoderConfig {
    fn transformer_config(&self) -> TransformerConfig {
        TransformerConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            hidden_act: self.hidden_act,
            attention_bias: self.attention_bias,
            qk_norm: false,
            layer_scale: true,
            sliding_window: self.sliding_window,
        }
    }
}

/// Extra right padding so that the last frame of a causal convolution is complete.
fn extra_padding(len: usize, kernel_size: usize, padding_total: usize, stride: usize) -> usize {
    let n_frames = (len as f64 + padding_total as f64 - kernel_size as f64) / stride as f64 + 1.;
    let ideal_len =
        (n_frames.ceil() as i64 - 1) * stride as i64 + (kernel_size - padding_total) as i64;
    (ideal_len - len as i64).max(0) as usize
}

/// Conv1d with causal (left) padding, matching `Qwen3TTSTokenizerV2CausalConvNet` and the
/// `MimiConv1d` of the encoder: zeros, or copies of the edge samples for the encoder's
/// downsampling convolution.
#[derive(Module, Debug)]
struct CausalConv1d {
    conv: Conv1d,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    #[module(skip)]
    edge_padding: bool,
}

impl CausalConv1d {
    fn init(
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        dilation: usize,
        stride: usize,
        groups: usize,
        device: &Device,
    ) -> Self {
        let config = Conv1dConfig::new(in_c, out_c, kernel_size)
            .with_stride(stride)
            .with_dilation(dilation)
            .with_groups(groups);
        Self::from_config(config, false, device)
    }

    /// `edge_padding` fills the padding with the edge samples rather than with zeros.
    fn from_config(config: Conv1dConfig, edge_padding: bool, device: &Device) -> Self {
        let kernel_size = (config.kernel_size - 1) * config.dilation + 1;
        Self {
            kernel_size,
            stride: config.stride,
            padding: kernel_size - config.stride,
            edge_padding,
            conv: config.init(device),
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let len = xs.dims()[2];
        let extra = extra_padding(len, self.kernel_size, self.padding, self.stride);
        let mode = if self.edge_padding {
            PadMode::Edge
        } else {
            PadMode::Constant(0.)
        };
        self.conv.forward(xs.pad([(self.padding, extra)], mode))
    }
}

/// ConvTranspose1d whose trailing `kernel_size - stride` samples are trimmed to keep it causal.
#[derive(Module, Debug)]
struct CausalConvTranspose1d {
    conv: ConvTranspose1d,
    right_pad: usize,
}

impl CausalConvTranspose1d {
    fn init(in_c: usize, out_c: usize, kernel_size: usize, stride: usize, device: &Device) -> Self {
        Self {
            conv: ConvTranspose1dConfig::new([in_c, out_c], kernel_size)
                .with_stride(stride)
                .init(device),
            right_pad: kernel_size - stride,
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let xs = self.conv.forward(xs);
        let len = xs.dims()[2];
        xs.narrow(2, 0, len - self.right_pad)
    }
}

#[derive(Module, Debug)]
struct ConvNeXtBlock {
    dwconv: CausalConv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Param<Tensor<1>>,
}

impl ConvNeXtBlock {
    fn init(dim: usize, device: &Device) -> Self {
        Self {
            dwconv: CausalConv1d::init(dim, dim, 7, 1, 1, dim, device),
            norm: LayerNormConfig::new(dim).with_epsilon(1e-6).init(device),
            pwconv1: crate::qwen3::linear_config(dim, 4 * dim).init(device),
            pwconv2: crate::qwen3::linear_config(4 * dim, dim).init(device),
            gamma: Param::from_tensor(Tensor::ones([dim], device)),
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        // The pointwise convolutions are linear layers over the channel dimension.
        let hidden = self.dwconv.forward(xs.clone()).swap_dims(1, 2);
        let hidden = self.norm.forward(hidden);
        let hidden = gelu(self.pwconv1.forward(hidden));
        let hidden = self.pwconv2.forward(hidden) * self.gamma.val().unsqueeze();
        xs + hidden.swap_dims(1, 2)
    }
}

/// `x + sin(x * exp(alpha))^2 / (exp(beta) + eps)` with per-channel `alpha` and `beta`.
#[derive(Module, Debug)]
struct SnakeBeta {
    alpha: Param<Tensor<1>>,
    beta: Param<Tensor<1>>,
}

impl SnakeBeta {
    fn init(channels: usize, device: &Device) -> Self {
        Self {
            alpha: Param::from_tensor(Tensor::zeros([channels], device)),
            beta: Param::from_tensor(Tensor::zeros([channels], device)),
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let channels = self.alpha.shape().dims::<1>()[0];
        let alpha = self.alpha.val().exp().reshape([1, channels, 1]);
        let inv_beta = (self.beta.val().exp() + 1e-9)
            .recip()
            .reshape([1, channels, 1]);
        let snake = (xs.clone() * alpha).sin().square() * inv_beta;
        xs + snake
    }
}

#[derive(Module, Debug)]
struct ResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv1d,
    act2: SnakeBeta,
    conv2: CausalConv1d,
}

impl ResidualUnit {
    fn init(dim: usize, dilation: usize, device: &Device) -> Self {
        Self {
            act1: SnakeBeta::init(dim, device),
            conv1: CausalConv1d::init(dim, dim, 7, dilation, 1, 1, device),
            act2: SnakeBeta::init(dim, device),
            conv2: CausalConv1d::init(dim, dim, 1, 1, 1, 1, device),
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let hidden = self.act1.forward(xs.clone());
        let hidden = self.conv1.forward(hidden);
        let hidden = self.act2.forward(hidden);
        xs + self.conv2.forward(hidden)
    }
}

#[derive(Module, Debug)]
struct DecoderBlock {
    act: SnakeBeta,
    upsample: CausalConvTranspose1d,
    residual_units: Vec<ResidualUnit>,
}

impl DecoderBlock {
    fn init(in_dim: usize, out_dim: usize, upsample_rate: usize, device: &Device) -> Self {
        Self {
            act: SnakeBeta::init(in_dim, device),
            upsample: CausalConvTranspose1d::init(
                in_dim,
                out_dim,
                2 * upsample_rate,
                upsample_rate,
                device,
            ),
            residual_units: [1, 3, 9]
                .into_iter()
                .map(|dilation| ResidualUnit::init(out_dim, dilation, device))
                .collect(),
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let mut xs = self.upsample.forward(self.act.forward(xs));
        for unit in self.residual_units.iter() {
            xs = unit.forward(xs);
        }
        xs
    }
}

/// One codebook, stored as the sum of the vectors assigned to each entry and their count.
#[derive(Module, Debug)]
struct Codebook {
    embedding_sum: Param<Tensor<2>>,
    cluster_usage: Param<Tensor<1>>,
}

impl Codebook {
    fn init(codebook_size: usize, dim: usize, device: &Device) -> Self {
        Self {
            embedding_sum: Param::from_tensor(Tensor::zeros([codebook_size, dim], device)),
            cluster_usage: Param::from_tensor(Tensor::ones([codebook_size], device)),
        }
    }

    /// Looks up `ids` (N) and returns the codebook entries (N, dim).
    fn decode(&self, ids: Tensor<1, Int>) -> Tensor<2> {
        let embedding_sum = self.embedding_sum.val().select(0, ids.clone());
        let cluster_usage = self
            .cluster_usage
            .val()
            .select(0, ids)
            .clamp_min(1e-5)
            .unsqueeze_dim::<2>(1);
        embedding_sum / cluster_usage
    }

    /// Every entry of the codebook, `codebook_size` rows of `dim`, on the host.
    fn entries(&self) -> Vec<f32> {
        let cluster_usage = self
            .cluster_usage
            .val()
            .clamp_min(1e-5)
            .unsqueeze_dim::<2>(1);
        (self.embedding_sum.val() / cluster_usage)
            .into_data()
            .try_to_vec::<f32>()
            .expect("the codebooks are f32")
    }
}

/// Residual vector quantizer decoder: sums the codebook entries of every layer and projects the
/// result back to the latent dimension.
#[derive(Module, Debug)]
struct ResidualVectorQuantizer {
    codebooks: Vec<Codebook>,
    output_proj: Conv1d,
}

impl ResidualVectorQuantizer {
    fn init(
        num_quantizers: usize,
        codebook_size: usize,
        dim: usize,
        output_dim: usize,
        device: &Device,
    ) -> Self {
        Self {
            codebooks: (0..num_quantizers)
                .map(|_| Codebook::init(codebook_size, dim, device))
                .collect(),
            output_proj: Conv1dConfig::new(dim, output_dim, 1)
                .with_bias(false)
                .init(device),
        }
    }

    /// `codes` has shape (B, num_quantizers, T); returns (B, output_dim, T).
    fn decode(&self, codes: Tensor<3, Int>) -> Tensor<3> {
        let [b, nq, t] = codes.dims();
        assert_eq!(
            nq,
            self.codebooks.len(),
            "expected {} quantizer layers, got {nq}",
            self.codebooks.len()
        );
        let mut acc: Option<Tensor<3>> = None;
        for (i, codebook) in self.codebooks.iter().enumerate() {
            let ids = codes.clone().narrow(1, i, 1).reshape([b * t]);
            let quantized = codebook.decode(ids).reshape([b as i32, t as i32, -1]);
            acc = Some(match acc {
                None => quantized,
                Some(acc) => acc + quantized,
            });
        }
        let hidden = acc.expect("no quantizer layers").swap_dims(1, 2);
        self.output_proj.forward(hidden)
    }
}

#[derive(Module, Debug)]
struct SplitResidualVectorQuantizer {
    rvq_first: ResidualVectorQuantizer,
    rvq_rest: ResidualVectorQuantizer,
    n_q_semantic: usize,
}

impl SplitResidualVectorQuantizer {
    fn init(cfg: &DecoderConfig, device: &Device) -> Self {
        let dim = cfg.codebook_dim / 2;
        Self {
            rvq_first: ResidualVectorQuantizer::init(
                cfg.num_semantic_quantizers,
                cfg.codebook_size,
                dim,
                cfg.codebook_dim,
                device,
            ),
            rvq_rest: ResidualVectorQuantizer::init(
                cfg.num_quantizers - cfg.num_semantic_quantizers,
                cfg.codebook_size,
                dim,
                cfg.codebook_dim,
                device,
            ),
            n_q_semantic: cfg.num_semantic_quantizers,
        }
    }

    /// `codes` has shape (B, num_quantizers, T); returns (B, codebook_dim, T).
    fn decode(&self, codes: Tensor<3, Int>) -> Tensor<3> {
        let nq = codes.dims()[1];
        let first = self
            .rvq_first
            .decode(codes.clone().narrow(1, 0, self.n_q_semantic));
        if nq > self.n_q_semantic {
            let rest =
                self.rvq_rest
                    .decode(codes.narrow(1, self.n_q_semantic, nq - self.n_q_semantic));
            first + rest
        } else {
            first
        }
    }
}

#[derive(Module, Debug)]
struct PreTransformer {
    input_proj: Linear,
    model: Transformer,
    output_proj: Linear,
}

impl PreTransformer {
    fn init(cfg: &DecoderConfig, device: &Device) -> Self {
        Self {
            input_proj: crate::qwen3::linear_config(cfg.latent_dim, cfg.hidden_size).init(device),
            model: Transformer::init(&cfg.transformer_config(), device),
            output_proj: crate::qwen3::linear_config(cfg.hidden_size, cfg.latent_dim).init(device),
        }
    }

    fn forward(&self, xs: Tensor<3>, state: &mut TransformerState) -> Tensor<3> {
        state.reset();
        let xs = self.input_proj.forward(xs);
        self.output_proj.forward(self.model.forward(xs, 0, state))
    }
}

/// The SnakeBeta/transposed convolution vocoder turning latents into waveform samples.
#[derive(Module, Debug)]
struct Vocoder {
    pre_conv: CausalConv1d,
    blocks: Vec<DecoderBlock>,
    final_act: SnakeBeta,
    final_conv: CausalConv1d,
}

impl Vocoder {
    fn init(cfg: &DecoderConfig, device: &Device) -> Self {
        let blocks = cfg
            .upsample_rates
            .iter()
            .enumerate()
            .map(|(i, &rate)| {
                let in_dim = cfg.decoder_dim / 2usize.pow(i as u32);
                let out_dim = cfg.decoder_dim / 2usize.pow(i as u32 + 1);
                DecoderBlock::init(in_dim, out_dim, rate, device)
            })
            .collect();
        let output_dim = cfg.decoder_dim / 2usize.pow(cfg.upsample_rates.len() as u32);
        Self {
            pre_conv: CausalConv1d::init(cfg.latent_dim, cfg.decoder_dim, 7, 1, 1, 1, device),
            blocks,
            final_act: SnakeBeta::init(output_dim, device),
            final_conv: CausalConv1d::init(output_dim, 1, 7, 1, 1, 1, device),
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let mut wav = self.pre_conv.forward(xs);
        for block in self.blocks.iter() {
            wav = block.forward(wav);
        }
        let wav = self.final_conv.forward(self.final_act.forward(wav));
        wav.clamp(-1., 1.)
    }
}

/// The speech-tokenizer decoder, the `decoder.` prefix of `speech_tokenizer/model.safetensors`.
#[derive(Module, Debug)]
pub struct Decoder {
    quantizer: SplitResidualVectorQuantizer,
    pre_conv: CausalConv1d,
    pre_transformer: PreTransformer,
    upsample: Vec<UpsampleStage>,
    vocoder: Vocoder,
    num_quantizers: usize,
    total_upsample: usize,
}

#[derive(Module, Debug)]
struct UpsampleStage {
    conv: CausalConvTranspose1d,
    convnext: ConvNeXtBlock,
}

impl Decoder {
    fn init(cfg: &DecoderConfig, device: &Device) -> Self {
        let upsample = cfg
            .upsampling_ratios
            .iter()
            .map(|&factor| UpsampleStage {
                conv: CausalConvTranspose1d::init(
                    cfg.latent_dim,
                    cfg.latent_dim,
                    factor,
                    factor,
                    device,
                ),
                convnext: ConvNeXtBlock::init(cfg.latent_dim, device),
            })
            .collect();
        Self {
            quantizer: SplitResidualVectorQuantizer::init(cfg, device),
            pre_conv: CausalConv1d::init(cfg.codebook_dim, cfg.latent_dim, 3, 1, 1, 1, device),
            pre_transformer: PreTransformer::init(cfg, device),
            upsample,
            vocoder: Vocoder::init(cfg, device),
            num_quantizers: cfg.num_quantizers,
            total_upsample: cfg.total_upsample(),
        }
    }

    /// Decodes a chunk of codes with shape (B, num_quantizers, T) into audio
    /// (B, 1, T * total_upsample).
    fn forward(&self, codes: Tensor<3, Int>, state: &mut TransformerState) -> Tensor<3> {
        let nq = codes.dims()[1];
        assert_eq!(
            nq, self.num_quantizers,
            "expected {} layers of codes, got {nq}",
            self.num_quantizers
        );
        let hidden = self.quantizer.decode(codes);
        let hidden = self.pre_conv.forward(hidden).swap_dims(1, 2);
        let hidden = self.pre_transformer.forward(hidden, state);
        let mut hidden = hidden.swap_dims(1, 2);
        for stage in self.upsample.iter() {
            hidden = stage.convnext.forward(stage.conv.forward(hidden));
        }
        self.vocoder.forward(hidden)
    }
}

// ---------------------------------------------------------------------------------------------
// The encoder.
// ---------------------------------------------------------------------------------------------

impl EncoderConfig {
    /// What the attention layers of the encoder transformer share with the Qwen3 stacks.
    fn transformer_config(&self) -> TransformerConfig {
        TransformerConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim(),
            rms_norm_eps: self.norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            hidden_act: self.hidden_act,
            attention_bias: self.attention_bias,
            qk_norm: false,
            layer_scale: true,
            sliding_window: self.sliding_window,
        }
    }
}

/// A residual unit of the SEANet encoder: two causal convolutions, each preceded by an ELU,
/// around a skip connection.
#[derive(Module, Debug)]
struct SeaNetResidual {
    conv1: CausalConv1d,
    conv2: CausalConv1d,
}

impl SeaNetResidual {
    fn init(cfg: &EncoderConfig, dim: usize, dilation: usize, device: &Device) -> Self {
        let hidden = dim / cfg.compress;
        Self {
            conv1: CausalConv1d::init(
                dim,
                hidden,
                cfg.residual_kernel_size,
                dilation,
                1,
                1,
                device,
            ),
            conv2: CausalConv1d::init(hidden, dim, 1, 1, 1, 1, device),
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let hidden = self.conv1.forward(elu(xs.clone(), 1.));
        xs + self.conv2.forward(elu(hidden, 1.))
    }
}

/// A stage of the SEANet encoder: residual units, then an ELU and a strided convolution that
/// doubles the channels while dividing the length by its stride.
#[derive(Module, Debug)]
struct SeaNetStage {
    residuals: Vec<SeaNetResidual>,
    downsample: CausalConv1d,
}

impl SeaNetStage {
    fn init(cfg: &EncoderConfig, dim: usize, ratio: usize, device: &Device) -> Self {
        Self {
            residuals: (0..cfg.num_residual_layers)
                .map(|j| {
                    let dilation = cfg.dilation_growth_rate.pow(j as u32);
                    SeaNetResidual::init(cfg, dim, dilation, device)
                })
                .collect(),
            downsample: CausalConv1d::init(dim, 2 * dim, 2 * ratio, 1, ratio, 1, device),
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let mut xs = xs;
        for residual in self.residuals.iter() {
            xs = residual.forward(xs);
        }
        self.downsample.forward(elu(xs, 1.))
    }
}

/// The SEANet convolutional stack: 24 kHz samples (B, 1, samples) to 25 Hz latents
/// (B, hidden_size, frames).
#[derive(Module, Debug)]
struct SeaNetEncoder {
    init_conv: CausalConv1d,
    stages: Vec<SeaNetStage>,
    final_conv: CausalConv1d,
}

impl SeaNetEncoder {
    fn init(cfg: &EncoderConfig, device: &Device) -> Self {
        let mut dim = cfg.num_filters;
        let mut stages = Vec::with_capacity(cfg.upsampling_ratios.len());
        // The strides are listed for the decoder, the encoder walks them backwards.
        for &ratio in cfg.upsampling_ratios.iter().rev() {
            stages.push(SeaNetStage::init(cfg, dim, ratio, device));
            dim *= 2;
        }
        Self {
            init_conv: CausalConv1d::init(
                cfg.audio_channels,
                cfg.num_filters,
                cfg.kernel_size,
                1,
                1,
                1,
                device,
            ),
            stages,
            final_conv: CausalConv1d::init(
                dim,
                cfg.hidden_size,
                cfg.last_kernel_size,
                1,
                1,
                1,
                device,
            ),
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let mut xs = self.init_conv.forward(xs);
        for stage in self.stages.iter() {
            xs = stage.forward(xs);
        }
        self.final_conv.forward(elu(xs, 1.))
    }
}

/// The feed-forward block of the encoder transformer: two linear layers around the activation,
/// without gating or biases.
#[derive(Module, Debug)]
struct EncoderMlp {
    fc1: Linear,
    fc2: Linear,
    #[module(skip)]
    act: crate::qwen3::config::Activation,
}

impl EncoderMlp {
    fn init(cfg: &EncoderConfig, device: &Device) -> Self {
        let linear = |d_in, d_out| {
            crate::qwen3::linear_config(d_in, d_out)
                .with_bias(false)
                .init(device)
        };
        Self {
            fc1: linear(cfg.hidden_size, cfg.intermediate_size),
            fc2: linear(cfg.intermediate_size, cfg.hidden_size),
            act: cfg.hidden_act,
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        self.fc2.forward(self.act.forward(self.fc1.forward(xs)))
    }
}

/// A layer of the encoder transformer: pre-norm attention and feed-forward branches, each
/// scaled by a learnt per-channel factor before the residual sum, with layer normalization
/// where the Qwen3 stacks have RMS normalization.
#[derive(Module, Debug)]
struct EncoderLayer {
    self_attn: Attention,
    mlp: EncoderMlp,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    self_attn_layer_scale: LayerScale,
    mlp_layer_scale: LayerScale,
}

impl EncoderLayer {
    fn init(cfg: &EncoderConfig, attention: &TransformerConfig, device: &Device) -> Self {
        let norm = || {
            LayerNormConfig::new(cfg.hidden_size)
                .with_epsilon(cfg.norm_eps)
                .init(device)
        };
        Self {
            self_attn: Attention::init(attention, device),
            mlp: EncoderMlp::init(cfg, device),
            input_layernorm: norm(),
            post_attention_layernorm: norm(),
            self_attn_layer_scale: LayerScale::init(cfg.hidden_size, device),
            mlp_layer_scale: LayerScale::init(cfg.hidden_size, device),
        }
    }

    fn forward(
        &self,
        xs: Tensor<3>,
        mask: Option<&Tensor<4, Bool>>,
        rotary: &RotarySlice,
        cache: &mut KvCache,
    ) -> Tensor<3> {
        let hidden = self.self_attn.forward(
            self.input_layernorm.forward(xs.clone()),
            mask,
            rotary,
            cache,
            Step::Static { offset: 0 },
        );
        let xs = xs + self.self_attn_layer_scale.forward(hidden);
        let hidden = self
            .mlp
            .forward(self.post_attention_layernorm.forward(xs.clone()));
        xs + self.mlp_layer_scale.forward(hidden)
    }
}

/// The transformer between the convolutional stack and the quantizer. Causal, and every frame
/// attends to the `sliding_window` most recent ones, itself included, as in the reference
/// `MimiModel`.
#[derive(Module, Debug)]
struct EncoderTransformer {
    layers: Vec<EncoderLayer>,
}

impl EncoderTransformer {
    fn init(cfg: &EncoderConfig, device: &Device) -> Self {
        let attention = cfg.transformer_config();
        Self {
            layers: (0..cfg.num_hidden_layers)
                .map(|_| EncoderLayer::init(cfg, &attention, device))
                .collect(),
        }
    }

    /// `xs` (B, hidden_size, frames) in the layout of the convolutions, returned the same way.
    fn forward(&self, xs: Tensor<3>, state: &mut TransformerState) -> Tensor<3> {
        state.reset();
        let mut xs = xs.swap_dims(1, 2);
        let seq_len = xs.dims()[1];
        let (mask, rotary, caches) = state.step(seq_len, 0);
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            xs = layer.forward(xs, mask.as_ref(), &rotary, cache);
        }
        xs.swap_dims(1, 2)
    }
}

/// The encoding half of a residual vector quantizer: an input projection and the codebooks.
/// Every codebook quantizes what the previous ones left, to its nearest entry.
#[derive(Module, Debug)]
struct RvqEncoder {
    input_proj: Conv1d,
    codebooks: Vec<Codebook>,
}

impl RvqEncoder {
    fn init(
        num_quantizers: usize,
        codebook_size: usize,
        input_dim: usize,
        dim: usize,
        device: &Device,
    ) -> Self {
        Self {
            input_proj: Conv1dConfig::new(input_dim, dim, 1)
                .with_bias(false)
                .init(device),
            codebooks: (0..num_quantizers)
                .map(|_| Codebook::init(codebook_size, dim, device))
                .collect(),
        }
    }

    /// The codes of `xs` (1, input_dim, frames): one row of `frames` codes per codebook.
    ///
    /// The projection runs on the device; the nearest-neighbour search runs on the host, one
    /// frame at a time in f32: a few million multiply-adds per second of audio, once per voice,
    /// and exact where a matmul on the device would round its products through tf32.
    fn encode(&self, xs: Tensor<3>) -> Vec<Vec<u32>> {
        let projected = self.input_proj.forward(xs);
        let [_, dim, frames] = projected.dims();
        let mut residuals = projected
            .swap_dims(1, 2)
            .reshape([frames, dim])
            .into_data()
            .try_to_vec::<f32>()
            .expect("the latents are f32");
        let mut codes = Vec::with_capacity(self.codebooks.len());
        for codebook in self.codebooks.iter() {
            let entries = codebook.entries();
            let mut layer = Vec::with_capacity(frames);
            for residual in residuals.chunks_exact_mut(dim) {
                let code = nearest(residual, &entries, dim);
                let entry = &entries[code * dim..(code + 1) * dim];
                for (r, e) in residual.iter_mut().zip(entry) {
                    *r -= e;
                }
                layer.push(code as u32);
            }
            codes.push(layer);
        }
        codes
    }
}

/// The index of the row of `entries` (rows of `dim`) closest to `xs` in Euclidean distance.
fn nearest(xs: &[f32], entries: &[f32], dim: usize) -> usize {
    let mut best = 0;
    let mut best_dist = f32::INFINITY;
    for (index, entry) in entries.chunks_exact(dim).enumerate() {
        let dist: f32 = xs.iter().zip(entry).map(|(x, e)| (x - e) * (x - e)).sum();
        if dist < best_dist {
            best_dist = dist;
            best = index;
        }
    }
    best
}

/// The split residual vector quantizer of the encoder: the semantic codebook and the acoustic
/// ones both quantize the same latents, side by side rather than one after the other.
#[derive(Module, Debug)]
struct SplitRvqEncoder {
    semantic_residual_vector_quantizer: RvqEncoder,
    acoustic_residual_vector_quantizer: RvqEncoder,
}

impl SplitRvqEncoder {
    fn init(cfg: &EncoderConfig, num_quantizers: usize, device: &Device) -> Self {
        let rvq = |num_quantizers| {
            RvqEncoder::init(
                num_quantizers,
                cfg.codebook_size,
                cfg.hidden_size,
                cfg.codebook_dim,
                device,
            )
        };
        Self {
            semantic_residual_vector_quantizer: rvq(cfg.num_semantic_quantizers),
            acoustic_residual_vector_quantizer: rvq(num_quantizers - cfg.num_semantic_quantizers),
        }
    }

    /// The codes of `xs` (1, hidden_size, frames), laid out frame by frame.
    fn encode(&self, xs: Tensor<3>) -> Vec<u32> {
        let mut layers = self.semantic_residual_vector_quantizer.encode(xs.clone());
        layers.extend(self.acoustic_residual_vector_quantizer.encode(xs));
        let frames = layers.first().map_or(0, |layer| layer.len());
        let mut codes = Vec::with_capacity(frames * layers.len());
        for frame in 0..frames {
            codes.extend(layers.iter().map(|layer| layer[frame]));
        }
        codes
    }
}

/// The speech-tokenizer encoder, the `encoder.` prefix of `speech_tokenizer/model.safetensors`.
///
/// A SEANet stack takes the 24 kHz samples to 25 Hz latents, a transformer refines them, a
/// strided convolution halves their rate to the 12.5 Hz of the codes and a split residual vector
/// quantizer turns every frame into its codes.
#[derive(Module, Debug)]
pub struct Encoder {
    encoder: SeaNetEncoder,
    encoder_transformer: EncoderTransformer,
    /// Only there when the transformer runs at a higher rate than the codes.
    downsample: Option<CausalConv1d>,
    quantizer: SplitRvqEncoder,
    num_quantizers: usize,
    /// Audio samples per frame of codes.
    downsample_rate: usize,
}

impl Encoder {
    fn init(cfg: &SpeechTokenizerConfig, device: &Device) -> Result<Self, String> {
        let ecfg = &cfg.encoder_config;
        if !ecfg.use_causal_conv {
            return Err("the encoder only supports causal convolutions".to_string());
        }
        if ecfg.pad_mode != "constant" {
            return Err(format!("unsupported encoder pad mode {:?}", ecfg.pad_mode));
        }
        if ecfg.use_conv_shortcut {
            return Err(
                "the encoder's residual units with a convolution shortcut are not supported"
                    .to_string(),
            );
        }
        if ecfg.codebook_dim != ecfg.vector_quantization_hidden_dimension {
            return Err(format!(
                "the codebook dim {} and the quantization dim {} should match",
                ecfg.codebook_dim, ecfg.vector_quantization_hidden_dimension
            ));
        }
        if ecfg.head_dim() * ecfg.num_attention_heads != ecfg.hidden_size {
            return Err(format!("unsupported encoder head dim {}", ecfg.head_dim()));
        }
        let num_quantizers = cfg.encoder_valid_num_quantizers;
        if num_quantizers < ecfg.num_semantic_quantizers || num_quantizers > ecfg.num_quantizers {
            return Err(format!(
                "the number of quantizers should be in {}..={}, got {num_quantizers}",
                ecfg.num_semantic_quantizers, ecfg.num_quantizers
            ));
        }
        // The convolutional stack runs at 25 Hz, the codes at 12.5 Hz.
        let stride = ecfg.conv_frame_rate() / ecfg.frame_rate;
        if stride < 1. || stride.fract() != 0. {
            return Err(format!(
                "the frame rate {} should divide the rate of the convolutions {}",
                ecfg.frame_rate,
                ecfg.conv_frame_rate()
            ));
        }
        let stride = stride as usize;
        let downsample = (stride > 1).then(|| {
            let config = Conv1dConfig::new(ecfg.hidden_size, ecfg.hidden_size, 2 * stride)
                .with_stride(stride)
                .with_bias(false);
            CausalConv1d::from_config(config, true, device)
        });
        Ok(Self {
            encoder: SeaNetEncoder::init(ecfg, device),
            encoder_transformer: EncoderTransformer::init(ecfg, device),
            downsample,
            quantizer: SplitRvqEncoder::init(ecfg, num_quantizers, device),
            num_quantizers,
            downsample_rate: cfg.encode_downsample_rate,
        })
    }

    /// The codes of `pcm`, mono samples at the input sample rate: one frame of `num_quantizers`
    /// codes per `downsample_rate` samples, the last one included when incomplete.
    fn encode(&self, pcm: &[f32], state: &mut TransformerState, device: &Device) -> Vec<u32> {
        let len = pcm.len();
        let xs = Tensor::<3>::from_data(TensorData::new(pcm.to_vec(), [1, 1, len]), device);
        let xs = self.encoder.forward(xs);
        let xs = self.encoder_transformer.forward(xs, state);
        let xs = match &self.downsample {
            Some(downsample) => downsample.forward(xs),
            None => xs,
        };
        let mut codes = self.quantizer.encode(xs);
        let frames = usize::min(
            len.div_ceil(self.downsample_rate),
            codes.len() / self.num_quantizers,
        );
        codes.truncate(frames * self.num_quantizers);
        codes
    }
}

/// The root of `speech_tokenizer/model.safetensors`.
#[derive(Module, Debug)]
pub struct SpeechTokenizerModel {
    decoder: Decoder,
    encoder: Option<Encoder>,
}

/// The speech tokenizer together with the state its transformers need.
#[derive(Debug)]
pub struct SpeechTokenizer {
    model: SpeechTokenizerModel,
    state: TransformerState,
    encoder_state: Option<TransformerState>,
    config: SpeechTokenizerConfig,
    device: Device,
}

impl SpeechTokenizer {
    /// Loads the decoder from a `speech_tokenizer/model.safetensors` file. The codec always
    /// runs in f32, as in the reference implementation.
    pub fn load(
        cfg: &SpeechTokenizerConfig,
        weights: &std::path::Path,
        device: &Device,
    ) -> Result<Self, String> {
        Self::load_with(cfg, weights, device, false)
    }

    /// Loads the encoder as well as the decoder, for [`encode`](Self::encode).
    pub fn load_with_encoder(
        cfg: &SpeechTokenizerConfig,
        weights: &std::path::Path,
        device: &Device,
    ) -> Result<Self, String> {
        Self::load_with(cfg, weights, device, true)
    }

    fn load_with(
        cfg: &SpeechTokenizerConfig,
        weights: &std::path::Path,
        device: &Device,
        with_encoder: bool,
    ) -> Result<Self, String> {
        let decoder_cfg = &cfg.decoder_config;
        let encoder = with_encoder
            .then(|| Encoder::init(cfg, device))
            .transpose()?;
        let mut model = SpeechTokenizerModel {
            decoder: Decoder::init(decoder_cfg, device),
            encoder,
        };
        let mut store = SafetensorsStore::from_file(weights)
            .with_from_adapter(crate::qwen3::CheckpointAdapter)
            .remap(remapper(cfg)?)
            .allow_partial(true);
        let result = model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load {}: {err}", weights.display()))?;
        crate::qwen3::check_apply_result("speech tokenizer", &result)?;
        let f32 = burn::tensor::DType::F32;
        Ok(Self {
            model,
            state: TransformerState::new(&decoder_cfg.transformer_config(), f32, device),
            encoder_state: with_encoder.then(|| {
                TransformerState::new(&cfg.encoder_config.transformer_config(), f32, device)
            }),
            config: cfg.clone(),
            device: device.clone(),
        })
    }

    /// Whether the encoder was loaded, see [`load_with_encoder`](Self::load_with_encoder).
    pub fn has_encoder(&self) -> bool {
        self.model.encoder.is_some()
    }

    /// The sample rate [`encode`](Self::encode) expects.
    pub fn input_sample_rate(&self) -> usize {
        self.config.input_sample_rate
    }

    /// Encodes `pcm`, mono samples at [`input_sample_rate`](Self::input_sample_rate), into
    /// codes laid out frame by frame, `num_code_groups` per frame, as [`decode`](Self::decode)
    /// takes them: one frame per `encode_downsample_rate` samples, the last one included when
    /// incomplete. Needs [`load_with_encoder`](Self::load_with_encoder).
    pub fn encode(&mut self, pcm: &[f32]) -> Result<Vec<u32>, String> {
        let (Some(encoder), Some(state)) = (&self.model.encoder, &mut self.encoder_state) else {
            return Err(
                "the encoder was not loaded, use SpeechTokenizer::load_with_encoder".to_string(),
            );
        };
        if pcm.is_empty() {
            return Err("the recording to encode is empty".to_string());
        }
        Ok(encoder.encode(pcm, state, &self.device))
    }

    pub fn output_sample_rate(&self) -> usize {
        self.config.output_sample_rate
    }

    /// Number of audio samples produced for each codec frame.
    pub fn samples_per_frame(&self) -> usize {
        self.model.decoder.total_upsample
    }

    pub fn num_code_groups(&self) -> usize {
        self.model.decoder.num_quantizers
    }

    /// Decodes `frames` frames of `num_code_groups` codes, laid out frame by frame, into audio
    /// samples.
    ///
    /// Long sequences are decoded in chunks of `chunk_size` frames with `left_context` frames of
    /// context, mirroring the reference `chunked_decode`.
    pub fn decode(&mut self, codes: &[u32], chunk_size: usize, left_context: usize) -> Vec<f32> {
        assert!(chunk_size > 0, "the chunk size must be at least one frame");
        let num_code_groups = self.num_code_groups();
        let num_frames = self.num_frames(codes);
        let mut pcm = Vec::with_capacity(num_frames * self.samples_per_frame());
        let mut start = 0;
        while start < num_frames {
            let end = usize::min(start + chunk_size, num_frames);
            let context = usize::min(left_context, start);
            let chunk = &codes[(start - context) * num_code_groups..end * num_code_groups];
            let wav = self.decode_window(chunk, context, left_context + chunk_size);
            pcm.extend_from_slice(&wav);
            start = end;
        }
        pcm
    }

    /// Decodes the frames of `codes` after its first `context` ones, which are the context
    /// they are decoded with and whose audio is dropped, through a window of `window` frames:
    /// the codes are padded on the right with copies of their last frame up to the window, so
    /// that every chunk of an utterance reaches the decoder with the same shape, the first
    /// ones with their shorter context and the last one with its few frames included. The
    /// decoder is causal, so the padding changes nothing in the samples returned. A GPU
    /// backend compiles and autotunes its kernels per shape, a couple of minutes each on a
    /// cold cache for this decoder, and one shape means one such cost for any stream.
    pub fn decode_window(&mut self, codes: &[u32], context: usize, window: usize) -> Vec<f32> {
        let num_code_groups = self.num_code_groups();
        let frames = self.num_frames(codes);
        assert!(
            context < frames,
            "a window decodes at least one frame past its context"
        );
        assert!(
            frames <= window,
            "{frames} frames do not fit a window of {window}"
        );
        let samples_per_frame = self.samples_per_frame();
        let padded;
        let codes = if frames < window {
            let tail = codes[(frames - 1) * num_code_groups..].repeat(window - frames);
            padded = [codes, tail.as_slice()].concat();
            padded.as_slice()
        } else {
            codes
        };
        let mut wav = self.decode_chunk(codes);
        wav.truncate(frames * samples_per_frame);
        wav.drain(..context * samples_per_frame);
        wav
    }

    /// Decodes every frame of `codes` in a single pass, at whatever length they come; see
    /// [`decode_window`](Self::decode_window) for the variant that keeps one shape.
    pub fn decode_chunk(&mut self, codes: &[u32]) -> Vec<f32> {
        let frames = self.num_frames(codes);
        let codes = self.codes_tensor(codes, frames);
        let wav = self.model.decoder.forward(codes, &mut self.state);
        let len = wav.dims()[2];
        wav.reshape([len])
            .into_data()
            .try_to_vec::<f32>()
            .expect("the decoder returns f32 samples")
    }

    fn num_frames(&self, codes: &[u32]) -> usize {
        let num_code_groups = self.num_code_groups();
        assert_eq!(
            codes.len() % num_code_groups,
            0,
            "expected a multiple of {num_code_groups} codes"
        );
        codes.len() / num_code_groups
    }

    /// Builds a (1, num_code_groups, frames) tensor out of frame-major codes.
    fn codes_tensor(&self, codes: &[u32], frames: usize) -> Tensor<3, Int> {
        let num_code_groups = self.num_code_groups();
        let codes: Vec<i64> = codes.iter().map(|&code| code as i64).collect();
        Tensor::<3, Int>::from_data(
            TensorData::new(codes, [1, frames, num_code_groups]),
            &self.device,
        )
        .swap_dims(1, 2)
    }
}

/// Maps the names of the checkpoint onto the module tree above.
fn remapper(cfg: &SpeechTokenizerConfig) -> Result<KeyRemapper, String> {
    let ecfg = &cfg.encoder_config;
    let cfg = &cfg.decoder_config;
    let mut patterns = vec![
        // The codebooks are nested one level deeper in the checkpoint.
        (
            r"\.vq\.layers\.(\d+)\._codebook\.".to_string(),
            ".codebooks.$1.".to_string(),
        ),
        // The pre-transformer holds its stack directly rather than under `model`.
        (
            r"^decoder\.pre_transformer\.(layers|norm)\.".to_string(),
            "decoder.pre_transformer.model.$1.".to_string(),
        ),
        // The upsampling stages are `nn.Sequential`s of a transposed convolution and a
        // ConvNeXt block.
        (
            r"^decoder\.upsample\.(\d+)\.0\.".to_string(),
            "decoder.upsample.$1.conv.".to_string(),
        ),
        (
            r"^decoder\.upsample\.(\d+)\.1\.".to_string(),
            "decoder.upsample.$1.convnext.".to_string(),
        ),
        // So is the vocoder: a convolution, one block per upsampling rate, an activation and a
        // final convolution.
        (
            r"^decoder\.decoder\.0\.".to_string(),
            "decoder.vocoder.pre_conv.".to_string(),
        ),
    ];
    let n = cfg.upsample_rates.len();
    for i in 1..=n {
        let prefix = format!(r"^decoder\.decoder\.{i}\.block\.");
        let block = format!("decoder.vocoder.blocks.{}.", i - 1);
        patterns.push((format!("{prefix}0\\."), format!("{block}act.")));
        patterns.push((format!("{prefix}1\\."), format!("{block}upsample.")));
        for j in 0..3 {
            patterns.push((
                format!("{prefix}{}\\.", j + 2),
                format!("{block}residual_units.{j}."),
            ));
        }
    }
    patterns.push((
        format!(r"^decoder\.decoder\.{}\.", n + 1),
        "decoder.vocoder.final_act.".to_string(),
    ));
    patterns.push((
        format!(r"^decoder\.decoder\.{}\.", n + 2),
        "decoder.vocoder.final_conv.".to_string(),
    ));

    // The encoder's codebooks are the decoder's module under other names.
    patterns.push((
        r"^encoder\.quantizer\.(\w+)\.layers\.(\d+)\.codebook\.embed_sum$".to_string(),
        "encoder.quantizer.$1.codebooks.$2.embedding_sum".to_string(),
    ));
    patterns.push((
        r"^encoder\.quantizer\.(\w+)\.layers\.(\d+)\.codebook\.".to_string(),
        "encoder.quantizer.$1.codebooks.$2.".to_string(),
    ));
    // The SEANet stack is one `nn.Sequential`: the first convolution, then per stage the
    // residual units (themselves sequences of ELU, convolution, ELU, convolution), an ELU and
    // the strided convolution, and at the end an ELU and the last convolution. The activations
    // take an index without holding parameters.
    let seanet = |index: usize| format!(r"^encoder\.encoder\.layers\.{index}\.");
    patterns.push((seanet(0), "encoder.encoder.init_conv.".to_string()));
    let per_stage = ecfg.num_residual_layers + 2;
    for stage in 0..ecfg.upsampling_ratios.len() {
        let first = 1 + stage * per_stage;
        for residual in 0..ecfg.num_residual_layers {
            let prefix = format!("encoder.encoder.stages.{stage}.residuals.{residual}.");
            for (conv, block) in [("conv1", 1), ("conv2", 3)] {
                patterns.push((
                    format!("{}block\\.{block}\\.", seanet(first + residual)),
                    format!("{prefix}{conv}."),
                ));
            }
        }
        patterns.push((
            seanet(first + ecfg.num_residual_layers + 1),
            format!("encoder.encoder.stages.{stage}.downsample."),
        ));
    }
    patterns.push((
        seanet(1 + ecfg.upsampling_ratios.len() * per_stage + 1),
        "encoder.encoder.final_conv.".to_string(),
    ));
    KeyRemapper::from_patterns(patterns).map_err(|err| format!("invalid remapping: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_picks_the_closest_entry() {
        // Four entries of two values.
        let entries = [0., 0., 1., 0., 0., 1., 1., 1.];
        assert_eq!(nearest(&[0.9, 0.1], &entries, 2), 1);
        assert_eq!(nearest(&[0.4, 0.6], &entries, 2), 2);
        assert_eq!(nearest(&[2., 2.], &entries, 2), 3);
        // A tie goes to the first entry.
        assert_eq!(nearest(&[0.5, 0.5], &entries, 2), 0);
    }

    #[test]
    fn codebook_entries_are_the_sums_divided_by_the_usage() {
        let device = crate::qwen3::test_device();
        let codebook = Codebook {
            embedding_sum: Param::from_tensor(Tensor::from_data([[2., 4.], [3., 0.]], &device)),
            cluster_usage: Param::from_tensor(Tensor::from_data([2., 0.], &device)),
        };
        let entries = codebook.entries();
        assert_eq!(&entries[..2], &[1., 2.]);
        // A never used entry is divided by the floor of the usage rather than by zero.
        assert!((entries[2] - 3e5).abs() < 1., "{}", entries[2]);
        assert_eq!(entries[3], 0.);
    }
}
