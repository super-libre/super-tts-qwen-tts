//! The Qwen3-TTS talker and its code predictor.
//!
//! Three flavors of checkpoints exist, all handled by [`Qwen3Tts`]:
//!
//! - `CustomVoice`: a set of predefined speakers, see [`Qwen3Tts::supported_speakers`],
//!   optionally steered with a natural language instruction (1.7B only),
//! - `VoiceDesign`: the voice is described by a natural language instruction,
//! - `Base`: the voice is cloned from a recording, through its speaker embedding
//!   ([`Qwen3Tts::speaker_embedding`], [`Voice::Embedding`]) and, for a closer match, its
//!   transcript and codes fed to the talker as an in-context example ([`IclReference`]).
//!
//! Text has to be tokenized with the Qwen2/Qwen3 tokenizer and wrapped in the chat template used
//! by the reference implementation, see [`Prompt`].

// `slice_assign` takes one range per dimension, in an array: one for a one-dimensional tensor.
#![allow(clippy::single_range_in_vec_init)]

use burn::nn::{Embedding, EmbeddingConfig, Linear};
use burn::prelude::*;
use burn::tensor::{DType, Distribution, Graph, IndexingUpdateOp, capture};
use burn_store::{FloatCastAdapter, KeyRemapper, ModuleAdapter, ModuleSnapshot, SafetensorsStore};

use crate::qwen3::config::{Activation, CodePredictorConfig, Config, Dialect, TalkerConfig};
use crate::qwen3::sampling::{MASKED, Sampling};
use crate::qwen3::speaker_encoder::{MelConfig, SpeakerEncoder, mel_spectrogram};
use crate::qwen3::transformer::{Transformer, TransformerConfig, TransformerState};
use std::cell::RefCell;
use std::ops::ControlFlow;
use std::rc::Rc;

impl CodePredictorConfig {
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
            // The code predictor only ever sees `num_code_groups + 1` positions.
            max_position_embeddings: self.num_code_groups + 1,
            hidden_act: self.hidden_act,
            attention_bias: self.attention_bias,
            qk_norm: true,
            layer_scale: false,
            sliding_window: None,
        }
    }
}

impl TalkerConfig {
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
            qk_norm: true,
            layer_scale: false,
            sliding_window: None,
        }
    }
}

/// Two-layer MLP projecting text embeddings to the talker dimension.
#[derive(Module, Debug)]
struct ResizeMlp {
    linear_fc1: Linear,
    linear_fc2: Linear,
    #[module(skip)]
    act: Activation,
}

impl ResizeMlp {
    fn init(
        input_size: usize,
        intermediate_size: usize,
        output_size: usize,
        act: Activation,
        device: &Device,
    ) -> Self {
        Self {
            linear_fc1: crate::qwen3::linear_config(input_size, intermediate_size).init(device),
            linear_fc2: crate::qwen3::linear_config(intermediate_size, output_size).init(device),
            act,
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let hidden = self.act.forward(self.linear_fc1.forward(xs));
        self.linear_fc2.forward(hidden)
    }
}

#[derive(Module, Debug)]
struct CodePredictorBackbone {
    /// One embedding table per predicted codebook, in the talker dimension.
    codec_embedding: Vec<Embedding>,
    transformer: Transformer,
}

/// Predicts the codebooks `1..num_code_groups` of a frame from the talker hidden state and the
/// embedding of the first codebook entry.
#[derive(Module, Debug)]
struct CodePredictor {
    model: CodePredictorBackbone,
    lm_head: Vec<Linear>,
    /// Maps talker-sized embeddings to the code predictor dimension when they differ.
    small_to_mtp_projection: Option<Linear>,
}

impl CodePredictor {
    fn init(cfg: &CodePredictorConfig, talker_hidden_size: usize, device: &Device) -> Self {
        let n = cfg.num_code_groups - 1;
        Self {
            model: CodePredictorBackbone {
                codec_embedding: (0..n)
                    .map(|_| EmbeddingConfig::new(cfg.vocab_size, talker_hidden_size).init(device))
                    .collect(),
                transformer: Transformer::init(&cfg.transformer_config(), device),
            },
            lm_head: (0..n)
                .map(|_| {
                    crate::qwen3::linear_config(cfg.hidden_size, cfg.vocab_size)
                        .with_bias(false)
                        .init(device)
                })
                .collect(),
            small_to_mtp_projection: (cfg.hidden_size != talker_hidden_size).then(|| {
                crate::qwen3::linear_config(talker_hidden_size, cfg.hidden_size).init(device)
            }),
        }
    }

    fn project(&self, xs: Tensor<3>) -> Tensor<3> {
        match &self.small_to_mtp_projection {
            Some(projection) => projection.forward(xs),
            None => xs,
        }
    }

    /// Samples the code of codebook `pass + 1` out of `last`, the hidden state of the pass at
    /// its last position (1, 1, hidden_size), and hands its embedding on: to the next pass and
    /// to the talker input of the next frame. Returns the code, which is also written into the
    /// frame.
    fn sample_code(
        &self,
        pass: usize,
        last: Tensor<3>,
        frame: &mut FrameState,
        sampling: &Sampling,
    ) -> Tensor<1, Int> {
        let logits = self.lm_head[pass].forward(last).cast(DType::F32);
        let vocab = logits.dims()[2];
        let noise = frame.noise.clone().narrow(0, pass * vocab, vocab);
        let code = sampling.draw(logits.reshape([vocab]), noise);
        // Written in place, as everything in the frame: a captured pass writes the buffers
        // where they were when it was recorded.
        frame
            .codes
            .inplace(|codes| codes.slice_assign([pass + 1..pass + 2], code.clone()));
        let embed = self.model.codec_embedding[pass].forward(code.clone().reshape([1, 1]));
        let hidden = embed.dims()[2];
        let all = [0..1, 0..1, 0..hidden];
        frame.acc.inplace(|acc| {
            let sum = acc.clone() + embed.clone();
            acc.slice_assign(all.clone(), sum)
        });
        if pass + 1 < self.lm_head.len() {
            frame.next.inplace(|next| next.slice_assign(all, embed));
        }
        code
    }
}

#[derive(Module, Debug)]
struct TalkerBackbone {
    text_embedding: Embedding,
    codec_embedding: Embedding,
    transformer: Transformer,
}

#[derive(Module, Debug)]
struct Talker {
    model: TalkerBackbone,
    text_projection: ResizeMlp,
    codec_head: Linear,
    code_predictor: CodePredictor,
}

impl Talker {
    /// Samples the first code of a frame out of `hidden`, the talker hidden state at the
    /// position before it (1, 1, hidden_size), and starts the frame with it: the code goes
    /// into the frame's codes, the hidden state and the embedding of the code become the input
    /// of the code predictor, and the embedding starts the sum of the next talker input.
    ///
    /// The frame's `bias` says what must not be sampled and `generated` what the repetition
    /// penalty applies to, which the sampled code is added to. Returns the code.
    fn sample_code0(
        &self,
        hidden: Tensor<3>,
        frame: &mut FrameState,
        sampling: &Sampling,
        repetition_penalty: f32,
    ) -> Tensor<1, Int> {
        let logits = self.codec_head.forward(hidden.clone()).cast(DType::F32);
        let vocab = logits.dims()[2];
        let mut logits = logits.reshape([vocab]);
        if repetition_penalty != 1. {
            // The logits of the tokens already generated are divided by the penalty when
            // positive and multiplied by it otherwise.
            let scale = logits
                .clone()
                .greater_equal_elem(0.)
                .float()
                .mul_scalar(1. / repetition_penalty - repetition_penalty)
                .add_scalar(repetition_penalty);
            let factor = (scale.sub_scalar(1.) * frame.generated.clone()).add_scalar(1.);
            logits = logits * factor;
        }
        let logits = logits + frame.bias.clone();
        let len = frame.noise.dims()[0];
        let noise = frame.noise.clone().narrow(0, len - vocab, vocab);
        let code = sampling.draw(logits, noise);
        frame
            .codes
            .inplace(|codes| codes.slice_assign([0..1], code.clone()));
        if repetition_penalty != 1. {
            let one = frame.one.clone();
            frame.generated.inplace(|generated| {
                generated.select_assign(0, code.clone(), one, IndexingUpdateOp::Assign)
            });
        }
        let embed = self
            .model
            .codec_embedding
            .forward(code.clone().reshape([1, 1]));
        let hidden_size = embed.dims()[2];
        frame.first.inplace(|first| {
            first
                .slice_assign([0..1, 0..1, 0..hidden_size], hidden)
                .slice_assign([0..1, 1..2, 0..hidden_size], embed.clone())
        });
        frame
            .acc
            .inplace(|acc| acc.slice_assign([0..1, 0..1, 0..hidden_size], embed));
        code
    }
}

/// The root of the `model.safetensors` file of a Qwen3-TTS checkpoint.
#[derive(Module, Debug)]
pub struct Model {
    talker: Talker,
    /// Only the Base checkpoints ship it.
    speaker_encoder: Option<SpeakerEncoder>,
}

impl Model {
    fn init(cfg: &Config, device: &Device) -> Result<Self, String> {
        let tc = &cfg.talker_config;
        let talker = Talker {
            model: TalkerBackbone {
                text_embedding: EmbeddingConfig::new(tc.text_vocab_size, tc.text_hidden_size)
                    .init(device),
                codec_embedding: EmbeddingConfig::new(tc.vocab_size, tc.hidden_size).init(device),
                transformer: Transformer::init(&tc.transformer_config(), device),
            },
            text_projection: ResizeMlp::init(
                tc.text_hidden_size,
                tc.text_hidden_size,
                tc.hidden_size,
                tc.hidden_act,
                device,
            ),
            codec_head: crate::qwen3::linear_config(tc.hidden_size, tc.vocab_size)
                .with_bias(false)
                .init(device),
            code_predictor: CodePredictor::init(&tc.code_predictor_config, tc.hidden_size, device),
        };
        let speaker_encoder = cfg
            .speaker_encoder_config
            .as_ref()
            .map(|cfg| SpeakerEncoder::init(cfg, device))
            .transpose()?;
        Ok(Self {
            talker,
            speaker_encoder,
        })
    }
}

/// Speaker conditioning of a prompt.
#[derive(Debug, Clone, Copy)]
pub enum Voice<'a> {
    /// No speaker conditioning: VoiceDesign models, or Base models without cloning.
    None,
    /// One of the predefined speakers of a CustomVoice model, see
    /// [`Qwen3Tts::supported_speakers`].
    Speaker(&'a str),
    /// A speaker embedding of shape `(hidden_size,)` for the Base models, see
    /// [`Qwen3Tts::speaker_embedding`].
    Embedding(&'a Tensor<1>),
}

/// In-context voice cloning reference for the Base models: the talker continues a transcript
/// and the codes of its recording, and keeps the voice.
#[derive(Debug, Clone, Copy)]
pub struct IclReference<'a> {
    /// Tokens of `<|im_start|>assistant\n{reference transcript}<|im_end|>\n`.
    pub ref_ids: &'a [u32],
    /// The codes of the reference recording, laid out frame by frame with `num_code_groups`
    /// per frame, as [`SpeechTokenizer::encode`](crate::qwen3::speech_tokenizer::SpeechTokenizer::encode)
    /// returns them.
    pub ref_codes: &'a [u32],
}

/// Inputs of one text-to-speech request.
///
/// The token sequences follow the chat template of the reference implementation and must be
/// produced by the Qwen tokenizer without any extra special tokens:
///
/// - `input_ids`: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`,
/// - `instruct_ids`: `<|im_start|>user\n{instruction}<|im_end|>\n`.
#[derive(Debug, Clone, Copy)]
pub struct Prompt<'a> {
    pub input_ids: &'a [u32],
    pub instruct_ids: Option<&'a [u32]>,
    /// One of [`Qwen3Tts::supported_languages`], `None` or `"auto"` for automatic detection.
    pub language: Option<&'a str>,
    pub voice: Voice<'a>,
    /// The in-context example of a voice clone, Base models only.
    pub icl: Option<IclReference<'a>>,
    /// With `non_streaming_mode` the whole text is part of the prefix. Otherwise only its
    /// first token is and the rest is fed one token per generated frame, which is how the
    /// reference implementation runs voice cloning.
    pub non_streaming_mode: bool,
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of codec frames to generate, 12.5 of them per second of audio. The
    /// default matches the `generation_config.json` of the checkpoints.
    pub max_new_tokens: usize,
    /// Sampling of the first codebook by the talker.
    pub sampling: Sampling,
    /// Sampling of the other codebooks by the code predictor.
    pub subtalker_sampling: Sampling,
    /// Penalty applied to the first-codebook tokens already generated, 1 disables it.
    pub repetition_penalty: f32,
    /// Seeds the device's random number generator, which the sampling draws its noise from.
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 8192,
            sampling: Sampling::TopKThenTopP {
                k: 50,
                p: 1.0,
                temperature: 0.9,
            },
            subtalker_sampling: Sampling::TopKThenTopP {
                k: 50,
                p: 1.0,
                temperature: 0.9,
            },
            repetition_penalty: 1.05,
            seed: 299792458,
        }
    }
}

/// Number of frames the end-of-speech token is suppressed for, as in the reference
/// implementation (`min_new_tokens`).
const MIN_NEW_TOKENS: usize = 2;

/// Number of special (non acoustic) entries at the end of the talker codec vocabulary.
const NUM_SPECIAL_CODEC_TOKENS: usize = 1024;

/// What the passes of a frame read and write besides the transformer caches: the sampled
/// codes, the noise the sampling draws from and the embeddings handed from one pass to the
/// next. A captured pass reads and writes these buffers in place, at the addresses it was
/// recorded with, and the eager passes go through the same buffers so that either kind can
/// follow the other. The sixteen samplings of a frame all happen on the device this way and
/// the host reads the codes once per frame, see [`Qwen3Tts::read_codes`].
struct FrameState {
    /// The codes of the frame being generated, one per codebook.
    codes: Tensor<1, Int>,
    /// Uniform noise for the draws of a frame, one vocabulary-sized row per codebook: the rows
    /// of the code predictor first, the talker's last.
    noise: Tensor<1>,
    /// Added to the talker logits: [`MASKED`] for the tokens that must not be sampled.
    bias: Tensor<1>,
    /// 1 at the first-codebook tokens generated so far, for the repetition penalty.
    generated: Tensor<1>,
    /// A one, what marks a token in `generated`.
    one: Tensor<1>,
    /// The input of the first pass of the code predictor, the talker hidden state and the
    /// embedding of the first code: (1, 2, talker_hidden_size).
    first: Tensor<3>,
    /// The input of every other pass, the embedding of the code just predicted:
    /// (1, 1, talker_hidden_size).
    next: Tensor<3>,
    /// The sum of the embeddings of the codes of the frame so far, which is the next talker
    /// input but for its text: (1, 1, talker_hidden_size).
    acc: Tensor<3>,
}

impl FrameState {
    fn new(tc: &TalkerConfig, dtype: DType, device: &Device) -> Self {
        let n = tc.num_code_groups;
        let code_vocab = tc.code_predictor_config.vocab_size;
        let hidden = tc.hidden_size;
        let f32 = (device, DType::F32);
        Self {
            codes: Tensor::zeros([n], device),
            noise: Tensor::zeros([(n - 1) * code_vocab + tc.vocab_size], f32),
            bias: Tensor::zeros([tc.vocab_size], f32),
            generated: Tensor::zeros([tc.vocab_size], f32),
            one: Tensor::ones([1], f32),
            first: Tensor::zeros([1, 2, hidden], (device, dtype)),
            next: Tensor::zeros([1, 1, hidden], (device, dtype)),
            acc: Tensor::zeros([1, 1, hidden], (device, dtype)),
        }
    }
}

/// A captured pass of either model. It writes its results into the [`FrameState`] and
/// returns the code it sampled.
type Pass = Graph<Tensor<1, Int>, Box<dyn FnMut() -> Tensor<1, Int>>>;

/// The forward passes of the code predictor captured as graphs, one per predicted codebook.
///
/// A frame runs the code predictor once per codebook it predicts, on a single token each time
/// (two the first time), and each of those passes is the same few hundred kernels on tensors of
/// the same shapes. Captured once, a pass replays as one dispatch instead of the few hundred
/// operations Burn would otherwise turn into kernels one at a time, which is what bounded the
/// frame rate. A replay reads and writes the very buffers it was recorded with: its input is
/// the embedding the previous pass wrote into the [`FrameState`], its key/value caches are
/// preallocated and written in place at the position of the pass, and it ends by sampling its
/// code and writing the embedding for the next pass, so the passes of a frame replay back to
/// back with nothing to wait for in between.
///
/// On a device without graph support a replay runs the same code eagerly.
struct CodePredictorGraphs {
    /// Indexed by the codebook the pass predicts, minus one. The passes own their caches.
    passes: Vec<Pass>,
    /// What the passes were captured to sample with.
    sampling: Sampling,
}

/// Writes `pos` into the one-element `buffer` in place: the buffer is what a captured graph
/// reads its position from, so it must stay where it is.
fn write_position(buffer: &mut Tensor<1, Int>, pos: usize, device: &Device) {
    let at = Tensor::from_data([pos as i64], device);
    buffer.inplace(|buffer| buffer.slice_assign([0..1], at));
}

/// What the captured decode steps of the talker share, see [`TalkerGraphs`].
struct TalkerGraphState {
    transformer: TransformerState,
    /// The input token of the step: (1, 1, hidden_size).
    input: Tensor<3>,
    /// Its position.
    pos: Tensor<1, Int>,
}

/// The decode step of the talker captured as a graph.
///
/// Unlike the code predictor's passes, the talker's step is the same computation at a different
/// position every frame, with a cache that grows across frames: it is captured once with the
/// position held by a device tensor, see [`Transformer::forward_at`], and replayed for every
/// frame after the input and the position are written in place. The caches are preallocated and
/// the step attends to all of them through a mask hiding the positions after the current one,
/// so they start small, for the prompt and a few hundred frames, and double when a generation
/// outgrows them, which moves them and captures the step again. The step ends by sampling the
/// first code of the next frame and starting that frame, see [`Talker::sample_code0`].
struct TalkerGraphs {
    state: Rc<RefCell<TalkerGraphState>>,
    /// The number of positions the caches hold.
    capacity: usize,
    pass: Pass,
    /// What the step was captured to sample with.
    sampling: Sampling,
    repetition_penalty: f32,
}

pub struct Qwen3Tts {
    model: Model,
    talker_state: TransformerState,
    code_predictor_state: TransformerState,
    frame: Rc<RefCell<FrameState>>,
    /// Whether the passes of the code predictor are captured and replayed, see
    /// [`CodePredictorGraphs`].
    graph_code_predictor: bool,
    code_predictor_graphs: Option<CodePredictorGraphs>,
    /// Whether the talker's decode steps are captured and replayed, see [`TalkerGraphs`].
    graph_talker: bool,
    talker_graphs: Option<TalkerGraphs>,
    config: Config,
    device: Device,
    dtype: DType,
}

impl Qwen3Tts {
    /// Loads a checkpoint from its `model.safetensors` file, casting the weights to `dtype`.
    pub fn load(
        cfg: &Config,
        weights: &std::path::Path,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, String> {
        if cfg.talker_config.vocab_size <= NUM_SPECIAL_CODEC_TOKENS
            || cfg.talker_config.codec_eos_token_id as usize >= cfg.talker_config.vocab_size
        {
            return Err(format!(
                "unexpected talker vocab size {} for eos token {}",
                cfg.talker_config.vocab_size, cfg.talker_config.codec_eos_token_id
            ));
        }
        let mut model = Model::init(cfg, device)?;
        let mut store = SafetensorsStore::from_file(weights)
            .with_from_adapter(crate::qwen3::CheckpointAdapter.chain(FloatCastAdapter::to(dtype)))
            .remap(remapper(cfg)?)
            .allow_partial(true);
        let mut result = model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load {}: {err}", weights.display()))?;
        // The configuration announces the speaker encoder; a checkpoint without any of its
        // weights loads without it, and simply cannot clone.
        let speaker_encoder = |path: &str| path.starts_with("speaker_encoder.");
        if model.speaker_encoder.is_some()
            && !result.applied.iter().any(|path| speaker_encoder(path))
        {
            model.speaker_encoder = None;
            result.missing.retain(|(path, _)| !speaker_encoder(path));
        }
        crate::qwen3::check_apply_result("talker", &result)?;
        Ok(Self {
            talker_state: TransformerState::new(
                &cfg.talker_config.transformer_config(),
                dtype,
                device,
            ),
            code_predictor_state: TransformerState::new(
                &cfg.talker_config.code_predictor_config.transformer_config(),
                dtype,
                device,
            ),
            frame: Rc::new(RefCell::new(FrameState::new(
                &cfg.talker_config,
                dtype,
                device,
            ))),
            graph_code_predictor: false,
            code_predictor_graphs: None,
            graph_talker: false,
            talker_graphs: None,
            model,
            config: cfg.clone(),
            device: device.clone(),
            dtype,
        })
    }

    /// Makes the generations from now on capture the talker's decode step as graphs and replay
    /// them, see [`TalkerGraphs`]. The capture happens at the start of a generation, whose
    /// length decides the size of the caches.
    pub fn enable_talker_graphs(&mut self) {
        self.graph_talker = true;
    }

    /// Makes the generations from now on capture the passes of the code predictor as graphs
    /// and replay them, see [`CodePredictorGraphs`]. The capture happens at the start of the
    /// first generation.
    pub fn enable_code_predictor_graphs(&mut self) {
        self.graph_code_predictor = true;
    }

    /// How the talker's decode step ran in the last generation, if it was captured: whether
    /// the device replays it as a hardware graph, rather than by running its operations again.
    pub fn talker_graph_is_hardware(&self) -> Option<bool> {
        self.talker_graphs
            .as_ref()
            .map(|graphs| graphs.pass.is_hardware())
    }

    /// The same for the passes of the code predictor.
    pub fn code_predictor_graph_is_hardware(&self) -> Option<bool> {
        self.code_predictor_graphs
            .as_ref()
            .map(|graphs| graphs.passes.iter().all(|pass| pass.is_hardware()))
    }

    /// Makes sure the talker's caches and captured step can hold a prompt of `prefix_len`
    /// tokens plus some frames, and that the step samples as `config` says, see
    /// [`TalkerGraphs`].
    fn prepare_talker_graphs(&mut self, prefix_len: usize, config: &GenerationConfig) {
        if self.talker_graphs.as_ref().is_some_and(|graphs| {
            graphs.capacity > prefix_len
                && graphs.sampling == config.sampling
                && graphs.repetition_penalty == config.repetition_penalty
        }) {
            return;
        }
        // The old step replays against the old caches, so it goes before those do.
        self.talker_graphs = None;
        let capacity = (prefix_len + 256).next_power_of_two();
        let cfg = self.config.talker_config.transformer_config();
        let hidden = self.config.talker_config.hidden_size;
        let state = Rc::new(RefCell::new(TalkerGraphState {
            transformer: TransformerState::new_fixed(&cfg, 1, capacity, self.dtype, &self.device),
            input: Tensor::zeros([1, 1, hidden], (&self.device, self.dtype)),
            pos: Tensor::zeros([1], &self.device),
        }));
        let pass = self.capture_talker_pass(&state, capacity, config);
        self.talker_graphs = Some(TalkerGraphs {
            state,
            capacity,
            pass,
            sampling: config.sampling.clone(),
            repetition_penalty: config.repetition_penalty,
        });
    }

    /// Captures the talker's decode step over `state`, whose caches hold `capacity` positions,
    /// the sampling of the first code of the next frame included.
    fn capture_talker_pass(
        &self,
        state: &Rc<RefCell<TalkerGraphState>>,
        capacity: usize,
        config: &GenerationConfig,
    ) -> Pass {
        // The warm-up runs write the cache at the position held by `pos`: the last one, which
        // the generation has not reached, and whose content is masked until it does.
        write_position(&mut state.borrow_mut().pos, capacity - 1, &self.device);
        let talker = self.model.talker.clone();
        let state = state.clone();
        let frame = self.frame.clone();
        let sampling = config.sampling.clone();
        let repetition_penalty = config.repetition_penalty;
        let mut run: Box<dyn FnMut() -> Tensor<1, Int>> = Box::new(move || {
            let state = &mut *state.borrow_mut();
            let hidden = talker.model.transformer.forward_at(
                state.input.clone(),
                &state.pos,
                &mut state.transformer,
            );
            talker.sample_code0(
                hidden,
                &mut frame.borrow_mut(),
                &sampling,
                repetition_penalty,
            )
        });
        // Compiled and autotuned outside the capture, as for the code predictor.
        let _ = run().into_data();
        capture(&self.device, run)
    }

    /// One decode step of the talker through its captured graph: `input` (1, 1, hidden_size)
    /// at position `pos`. The step ends by sampling the first code of the next frame and
    /// starting that frame, see [`Talker::sample_code0`]; `generated` holds the first codes
    /// sampled so far.
    fn talker_step(
        &mut self,
        input: Tensor<3>,
        pos: usize,
        config: &GenerationConfig,
        generated: &[u32],
    ) {
        let mut graphs = self
            .talker_graphs
            .take()
            .expect("the talker graphs are prepared at the start of a generation");
        if pos >= graphs.capacity {
            let TalkerGraphs {
                state,
                capacity,
                pass,
                sampling,
                repetition_penalty,
            } = graphs;
            // The step replays against the old caches, so it goes before those move.
            drop(pass);
            let capacity = capacity * 2;
            state.borrow_mut().transformer.grow(capacity);
            let pass = self.capture_talker_pass(&state, capacity, config);
            // The runs of the capture sampled codes of their own.
            self.write_generated(generated);
            graphs = TalkerGraphs {
                state,
                capacity,
                pass,
                sampling,
                repetition_penalty,
            };
        }
        {
            let state = &mut *graphs.state.borrow_mut();
            let hidden = input.dims()[2];
            // Written in place: the state holds the only reference to the two buffers.
            state
                .input
                .inplace(|buffer| buffer.slice_assign([0..1, 0..1, 0..hidden], input));
            write_position(&mut state.pos, pos, &self.device);
        }
        // Safety: every buffer the step reads or writes is kept alive by `graphs`, by the
        // frame or by the model, the writes above and the reads of the frame go through the
        // same client and stream as the replay, and nothing else touches those buffers in the
        // meantime.
        let _ = unsafe { graphs.pass.replay() };
        self.talker_graphs = Some(graphs);
    }

    /// Captures the forward passes of the code predictor as graphs sampling as `sampling`
    /// says, unless they already are, see [`CodePredictorGraphs`].
    fn prepare_code_predictor_graphs(&mut self, sampling: &Sampling) {
        if self
            .code_predictor_graphs
            .as_ref()
            .is_some_and(|graphs| &graphs.sampling == sampling)
        {
            return;
        }
        self.code_predictor_graphs = None;
        let code_predictor = &self.model.talker.code_predictor;
        let num_passes = code_predictor.lm_head.len();
        // Two tokens in the first pass, then one per pass.
        let capacity = num_passes + 1;
        let state = Rc::new(RefCell::new(
            TransformerState::new_fixed(
                &self
                    .config
                    .talker_config
                    .code_predictor_config
                    .transformer_config(),
                1,
                capacity,
                self.dtype,
                &self.device,
            )
            .for_capture(),
        ));
        let passes = (0..num_passes)
            .map(|pass| {
                let state = state.clone();
                let frame = self.frame.clone();
                let code_predictor = code_predictor.clone();
                let sampling = sampling.clone();
                let mut run: Box<dyn FnMut() -> Tensor<1, Int>> = Box::new(move || {
                    let state = &mut *state.borrow_mut();
                    let frame = &mut *frame.borrow_mut();
                    let (xs, pos) = match pass {
                        0 => (frame.first.clone(), 0),
                        _ => (frame.next.clone(), pass + 1),
                    };
                    let hidden = code_predictor.model.transformer.forward(
                        code_predictor.project(xs),
                        pos,
                        state,
                    );
                    let len = hidden.dims()[1];
                    code_predictor.sample_code(pass, hidden.narrow(1, len - 1, 1), frame, &sampling)
                });
                // Kernels get compiled and autotuned the first time a pass runs, and the capture
                // keeps every buffer its warm-up runs touch, which for the dozens of candidates
                // an autotune benchmarks amounts to gigabytes. Running the pass once beforehand
                // leaves the capture nothing to tune. The read is what makes the run happen.
                let _ = run().into_data();
                capture(&self.device, run)
            })
            .collect::<Vec<_>>();
        self.code_predictor_graphs = Some(CodePredictorGraphs {
            passes,
            sampling: sampling.clone(),
        });
    }

    /// Draws fresh noise for the samplings of a frame, in place: the captured passes read the
    /// buffer where it is.
    fn refresh_noise(&self) {
        let frame = &mut *self.frame.borrow_mut();
        let len = frame.noise.dims()[0];
        let noise = Tensor::random(
            [len],
            Distribution::Uniform(0., 1.),
            (&self.device, DType::F32),
        );
        frame
            .noise
            .inplace(|buffer| buffer.slice_assign([0..len], noise));
    }

    /// Writes what the talker must not sample: the special tokens, except the end of speech
    /// one when `allow_eos`.
    fn write_bias(&self, allow_eos: bool) {
        let tc = &self.config.talker_config;
        let mut bias = vec![0f32; tc.vocab_size];
        for (token, value) in bias
            .iter_mut()
            .enumerate()
            .skip(tc.vocab_size - NUM_SPECIAL_CODEC_TOKENS)
        {
            if token != tc.codec_eos_token_id as usize || !allow_eos {
                *value = MASKED;
            }
        }
        let bias = Tensor::from_data(TensorData::new(bias, [tc.vocab_size]), &self.device);
        self.frame
            .borrow_mut()
            .bias
            .inplace(|buffer| buffer.slice_assign([0..tc.vocab_size], bias));
    }

    /// Writes which first-codebook tokens the repetition penalty applies to.
    fn write_generated(&self, generated: &[u32]) {
        let vocab = self.config.talker_config.vocab_size;
        let mut mask = vec![0f32; vocab];
        for &code in generated {
            if let Some(entry) = mask.get_mut(code as usize) {
                *entry = 1.;
            }
        }
        let mask = Tensor::from_data(TensorData::new(mask, [vocab]), &self.device);
        self.frame
            .borrow_mut()
            .generated
            .inplace(|buffer| buffer.slice_assign([0..vocab], mask));
    }

    /// Reads the codes of the frame just generated, the one round trip of a frame.
    fn read_codes(&self) -> Vec<u32> {
        self.frame
            .borrow()
            .codes
            .clone()
            .into_data()
            .iter::<i64>()
            .map(|code| code as u32)
            .collect()
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn num_code_groups(&self) -> usize {
        self.config.talker_config.num_code_groups
    }

    /// Whether the checkpoint clones voices, through [`speaker_embedding`](Self::speaker_embedding).
    pub fn has_speaker_encoder(&self) -> bool {
        self.model.speaker_encoder.is_some()
    }

    /// The speaker embedding of a mono recording sampled at 24 kHz, `(hidden_size,)` in the
    /// talker's dtype, to use as [`Voice::Embedding`]. Base models only.
    pub fn speaker_embedding(&self, samples: &[f32]) -> Result<Tensor<1>, String> {
        let (Some(encoder), Some(cfg)) = (
            &self.model.speaker_encoder,
            &self.config.speaker_encoder_config,
        ) else {
            return Err(
                "this checkpoint has no speaker encoder, voice cloning needs a Base model"
                    .to_string(),
            );
        };
        let mel_cfg = MelConfig {
            sample_rate: cfg.sample_rate,
            ..Default::default()
        };
        let (mels, frames) = mel_spectrogram(samples, &mel_cfg)?;
        let mels = Tensor::<3>::from_data(
            TensorData::new(mels, [1, frames, mel_cfg.num_mels]),
            &self.device,
        )
        .cast(self.dtype);
        Ok(encoder.forward(mels).squeeze_dim(0))
    }

    /// Names of the predefined speakers, empty for models without any.
    pub fn supported_speakers(&self) -> Vec<&str> {
        let mut speakers: Vec<_> = self
            .config
            .talker_config
            .spk_id
            .keys()
            .map(|s| s.as_str())
            .collect();
        speakers.sort_unstable();
        speakers
    }

    /// Names of the languages accepted by [`Prompt::language`], `"auto"` included. The
    /// dialects are not listed, they are selected through the speaker as in the reference
    /// implementation.
    pub fn supported_languages(&self) -> Vec<&str> {
        let mut languages: Vec<_> = self
            .config
            .talker_config
            .codec_language_id
            .keys()
            .filter(|name| !name.contains("dialect"))
            .map(|s| s.as_str())
            .collect();
        languages.sort_unstable();
        languages.insert(0, "auto");
        languages
    }

    fn ids_tensor(&self, ids: &[u32]) -> Tensor<2, Int> {
        ids_tensor(ids, &self.device)
    }

    /// Text embeddings projected to the talker dimension, shape (1, len, hidden_size).
    fn text_embed(&self, ids: &[u32]) -> Tensor<3> {
        let talker = &self.model.talker;
        let embeddings = talker.model.text_embedding.forward(self.ids_tensor(ids));
        talker.text_projection.forward(embeddings)
    }

    /// Codec embeddings, shape (1, len, hidden_size).
    fn codec_embed(&self, ids: &[u32]) -> Tensor<3> {
        self.model
            .talker
            .model
            .codec_embedding
            .forward(self.ids_tensor(ids))
    }

    /// The sum of the embeddings of every codebook of `codes`, laid out frame by frame with
    /// `num_code_groups` per frame: (1, frames, hidden_size). Codebook 0 uses the talker's
    /// table, the others the code predictor's.
    fn frames_embed(&self, codes: &[u32]) -> Result<Tensor<3>, String> {
        let groups = self.num_code_groups();
        if codes.is_empty() || !codes.len().is_multiple_of(groups) {
            return Err(format!(
                "expected a non-empty multiple of {groups} reference codes, got {}",
                codes.len()
            ));
        }
        let column = |group: usize| -> Vec<u32> {
            codes.iter().skip(group).step_by(groups).copied().collect()
        };
        let talker = &self.model.talker;
        let mut acc = talker
            .model
            .codec_embedding
            .forward(self.ids_tensor(&column(0)));
        for group in 1..groups {
            let table = &talker.code_predictor.model.codec_embedding[group - 1];
            acc = acc + table.forward(self.ids_tensor(&column(group)));
        }
        Ok(acc)
    }

    fn speaker_embed(&self, voice: Voice) -> Result<Option<Tensor<3>>, String> {
        match voice {
            Voice::None => Ok(None),
            Voice::Speaker(name) => {
                let name = name.to_lowercase();
                match self.config.talker_config.spk_id.get(&name) {
                    Some(&id) => Ok(Some(self.codec_embed(&[id]))),
                    None => Err(format!(
                        "unknown speaker {name:?}, supported speakers: {:?}",
                        self.supported_speakers()
                    )),
                }
            }
            Voice::Embedding(embedding) => {
                let hidden_size = self.config.talker_config.hidden_size;
                let [size] = embedding.dims();
                if size != hidden_size {
                    return Err(format!(
                        "the speaker embedding has {size} values, the talker expects {hidden_size}"
                    ));
                }
                Ok(Some(embedding.clone().cast(self.dtype).reshape([
                    1,
                    1,
                    hidden_size,
                ])))
            }
        }
    }

    fn language_id(&self, language: Option<&str>, voice: Voice) -> Result<Option<u32>, String> {
        let tc = &self.config.talker_config;
        let language = language.unwrap_or("auto").to_lowercase();
        let mut language_id = if language == "auto" {
            None
        } else {
            match tc.codec_language_id.get(&language) {
                Some(&id) => Some(id),
                None => {
                    return Err(format!(
                        "unknown language {language:?}, supported languages: {:?}",
                        self.supported_languages()
                    ));
                }
            }
        };
        // Dialect speakers switch the language tag to their dialect.
        if let (true, Voice::Speaker(name)) = (language == "chinese" || language == "auto", voice)
            && let Some(Dialect::Name(dialect)) = tc.spk_is_dialect.get(&name.to_lowercase())
        {
            match tc.codec_language_id.get(dialect) {
                Some(&id) => language_id = Some(id),
                None => return Err(format!("unknown dialect {dialect:?} for speaker {name:?}")),
            }
        }
        Ok(language_id)
    }

    /// Builds the prefix embeddings fed to the talker, shape (1, len, hidden_size), together
    /// with the text embeddings to add to the generated frames (`trailing`) and the padding
    /// embedding used once they are exhausted.
    fn build_prompt(&self, prompt: &Prompt) -> Result<(Tensor<3>, Tensor<3>, Tensor<3>), String> {
        let cfg = &self.config;
        let tc = &cfg.talker_config;
        let n = prompt.input_ids.len();
        // <|im_start|>assistant\n {text} <|im_end|>\n<|im_start|>assistant\n
        if n < 8 {
            return Err(format!(
                "input_ids is too short ({n}), it must follow the chat template"
            ));
        }
        let text_ids = &prompt.input_ids[3..n - 5];
        if text_ids.is_empty() {
            return Err("the text to synthesize is empty".to_string());
        }

        let mut parts = Vec::new();
        if let Some(instruct_ids) = prompt.instruct_ids
            && !instruct_ids.is_empty()
        {
            parts.push(self.text_embed(instruct_ids));
        }

        let speaker_embed = self.speaker_embed(prompt.voice)?;
        let language_id = self.language_id(prompt.language, prompt.voice)?;

        let special = self.text_embed(&[
            cfg.tts_bos_token_id,
            cfg.tts_eos_token_id,
            cfg.tts_pad_token_id,
        ]);
        let tts_bos = special.clone().narrow(1, 0, 1);
        let tts_eos = special.clone().narrow(1, 1, 1);
        let tts_pad = special.narrow(1, 2, 1);

        let codec_prefill = match language_id {
            None => vec![
                tc.codec_nothink_id,
                tc.codec_think_bos_id,
                tc.codec_think_eos_id,
            ],
            Some(id) => vec![
                tc.codec_think_id,
                tc.codec_think_bos_id,
                id,
                tc.codec_think_eos_id,
            ],
        };
        let mut codec_input = vec![self.codec_embed(&codec_prefill)];
        if let Some(speaker_embed) = speaker_embed {
            codec_input.push(speaker_embed);
        }
        codec_input.push(self.codec_embed(&[tc.codec_pad_id, tc.codec_bos_id]));
        let codec_input = Tensor::cat(codec_input, 1);
        let k = codec_input.dims()[1];

        // <|im_start|>assistant\n
        let role = self.text_embed(&prompt.input_ids[..3]);
        // tts_pad * (k - 2) + tts_bos, summed with the codec prefix but its final codec_bos.
        let hidden_size = tc.hidden_size;
        let text_prefix = Tensor::cat(
            vec![
                tts_pad.clone().expand([1, k - 2, hidden_size]),
                tts_bos.clone(),
            ],
            1,
        );
        let prefix = text_prefix + codec_input.clone().narrow(1, 0, k - 1);
        let mut embeds = vec![role, prefix];

        let codec_pad = self.codec_embed(&[tc.codec_pad_id]);
        let codec_bos = codec_input.narrow(1, k - 1, 1);
        let trailing = match prompt.icl {
            Some(icl) => {
                // <|im_start|>assistant\n {reference transcript} <|im_end|>\n
                let m = icl.ref_ids.len();
                if m < 5 {
                    return Err(format!(
                        "ref_ids is too short ({m}), it must follow the chat template"
                    ));
                }
                // The talker reads the reference transcript then the text, over the reference
                // codes: it continues the recording with the text, in the same voice.
                let mut ids = icl.ref_ids[3..m - 2].to_vec();
                ids.extend_from_slice(text_ids);
                let text_embed = Tensor::cat(vec![self.text_embed(&ids), tts_eos], 1);
                let codec_embed =
                    Tensor::cat(vec![codec_bos, self.frames_embed(icl.ref_codes)?], 1);
                let text_len = text_embed.dims()[1];
                let codec_len = codec_embed.dims()[1];
                if prompt.non_streaming_mode {
                    let text = text_embed + codec_pad;
                    let codec = codec_embed + tts_pad.clone();
                    embeds.push(Tensor::cat(vec![text, codec], 1));
                    tts_pad.clone()
                } else if text_len > codec_len {
                    embeds.push(text_embed.clone().narrow(1, 0, codec_len) + codec_embed);
                    text_embed.narrow(1, codec_len, text_len - codec_len)
                } else {
                    let padding = tts_pad
                        .clone()
                        .expand([1, codec_len - text_len, hidden_size]);
                    let text_embed = Tensor::cat(vec![text_embed, padding], 1);
                    embeds.push(text_embed + codec_embed);
                    tts_pad.clone()
                }
            }
            None if prompt.non_streaming_mode => {
                let text = Tensor::cat(vec![self.text_embed(text_ids), tts_eos], 1) + codec_pad;
                embeds.push(text);
                embeds.push(tts_pad.clone() + codec_bos);
                tts_pad.clone()
            }
            None => {
                embeds.push(self.text_embed(&text_ids[..1]) + codec_bos);
                if text_ids.len() > 1 {
                    Tensor::cat(vec![self.text_embed(&text_ids[1..]), tts_eos], 1)
                } else {
                    tts_eos
                }
            }
        };
        parts.push(Tensor::cat(embeds, 1));
        Ok((Tensor::cat(parts, 1), trailing, tts_pad))
    }

    /// Predicts the remaining codebooks of the frame started by [`Talker::sample_code0`],
    /// into the frame's codes.
    fn predict_codes(&mut self, sampling: &Sampling) {
        if let Some(graphs) = &mut self.code_predictor_graphs {
            for pass in graphs.passes.iter_mut() {
                // Safety: every buffer a pass reads or writes is kept alive by `graphs`, by the
                // frame or by the model, the writes and reads of the frame go through the same
                // client and stream as the replays, and nothing else touches those buffers in
                // the meantime.
                let _ = unsafe { pass.replay() };
            }
            return;
        }
        self.predict_codes_eager(sampling);
    }

    /// [`predict_codes`](Self::predict_codes) as plain operations, with a growing cache.
    fn predict_codes_eager(&mut self, sampling: &Sampling) {
        let code_predictor = &self.model.talker.code_predictor;
        self.code_predictor_state.reset();
        let frame = &mut *self.frame.borrow_mut();
        let mut xs = frame.first.clone();
        let mut offset = 0;
        for pass in 0..code_predictor.lm_head.len() {
            let hidden = code_predictor.model.transformer.forward(
                code_predictor.project(xs),
                offset,
                &mut self.code_predictor_state,
            );
            let len = hidden.dims()[1];
            offset += len;
            code_predictor.sample_code(pass, hidden.narrow(1, len - 1, 1), frame, sampling);
            xs = frame.next.clone();
        }
    }

    /// Generates the codec frames for `prompt`, each frame holding `num_code_groups` codes.
    ///
    /// `on_frame` is called with every generated frame as soon as it is available, which is what
    /// the streaming mode of the example uses to decode the audio while it is generated. It
    /// returns whether to keep generating: [`ControlFlow::Break`] ends the generation after that
    /// frame and returns the frames produced so far, which is how a caller streaming to a
    /// listener who has gone away stops paying for the rest of the utterance. There is no other
    /// way out of the loop — it runs until the end of speech token or `max_new_tokens`, tens of
    /// seconds of work for a long text.
    pub fn generate_with_callback(
        &mut self,
        prompt: &Prompt,
        config: &GenerationConfig,
        mut on_frame: impl FnMut(&[u32]) -> ControlFlow<()>,
    ) -> Result<Vec<Vec<u32>>, String> {
        let (embeds, trailing, tts_pad) = self.build_prompt(prompt)?;
        let eos = self.config.talker_config.codec_eos_token_id;
        let trailing_len = trailing.dims()[1];
        let prefix_len = embeds.dims()[1];
        // The draws take their noise from the device's generator.
        self.device.seed(config.seed);
        if self.graph_code_predictor {
            self.prepare_code_predictor_graphs(&config.subtalker_sampling);
        }
        if self.graph_talker {
            self.prepare_talker_graphs(prefix_len, config);
        }
        let mut generated = Vec::new();
        self.write_generated(&generated);
        // The end of speech token is suppressed for the first few frames, as the reference
        // implementation does through `min_new_tokens`.
        self.write_bias(MIN_NEW_TOKENS == 0);

        let hidden = if self.graph_talker {
            let graphs = self.talker_graphs.as_ref().expect("just prepared");
            let state = &mut *graphs.state.borrow_mut();
            self.model
                .talker
                .model
                .transformer
                .forward(embeds, 0, &mut state.transformer)
        } else {
            self.talker_state.reset();
            self.model
                .talker
                .model
                .transformer
                .forward(embeds, 0, &mut self.talker_state)
        };
        self.refresh_noise();
        self.model.talker.sample_code0(
            hidden.narrow(1, prefix_len - 1, 1),
            &mut self.frame.borrow_mut(),
            &config.sampling,
            config.repetition_penalty,
        );
        let mut frames = Vec::new();
        for step in 0..config.max_new_tokens {
            self.refresh_noise();
            self.predict_codes(&config.subtalker_sampling);
            let frame = self.read_codes();
            let code0 = frame[0];
            if code0 == eos {
                break;
            }
            generated.push(code0);
            let keep_going = on_frame(&frame);
            frames.push(frame);
            if keep_going.is_break() {
                break;
            }
            if step + 1 == config.max_new_tokens {
                break;
            }
            if step + 1 == MIN_NEW_TOKENS {
                self.write_bias(true);
            }
            // The next talker input is the sum of the frame embeddings and of the next text
            // token if any, of the padding embedding otherwise. The step samples the first
            // code of the next frame.
            let text = if step < trailing_len {
                trailing.clone().narrow(1, step, 1)
            } else {
                tts_pad.clone()
            };
            let input = self.frame.borrow().acc.clone() + text;
            if self.graph_talker {
                self.talker_step(input, prefix_len + step, config, &generated);
            } else {
                let hidden = self.model.talker.model.transformer.forward(
                    input,
                    prefix_len + step,
                    &mut self.talker_state,
                );
                self.model.talker.sample_code0(
                    hidden,
                    &mut self.frame.borrow_mut(),
                    &config.sampling,
                    config.repetition_penalty,
                );
            }
        }
        Ok(frames)
    }

    /// The dtype the talker runs in.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

fn ids_tensor(ids: &[u32], device: &Device) -> Tensor<2, Int> {
    let len = ids.len();
    let ids: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
    Tensor::<2, Int>::from_data(TensorData::new(ids, [1, len]), device)
}

/// Maps the names of the checkpoint onto the module tree above: the two stacks that hold their
/// layers next to an embedding table move, and the speaker encoder's first block, a plain
/// convolution where the others are SE-Res2Net blocks, leaves their list.
fn remapper(cfg: &Config) -> Result<KeyRemapper, String> {
    let mut patterns = vec![
        (
            r"^talker\.model\.(layers|norm)\.".to_string(),
            "talker.model.transformer.$1.".to_string(),
        ),
        (
            r"^talker\.code_predictor\.model\.(layers|norm)\.".to_string(),
            "talker.code_predictor.model.transformer.$1.".to_string(),
        ),
        (
            r"^speaker_encoder\.blocks\.0\.".to_string(),
            "speaker_encoder.first.".to_string(),
        ),
    ];
    if let Some(sc) = &cfg.speaker_encoder_config {
        // Each pattern is applied once, in this order, so the shifts do not chain.
        for i in 1..sc.enc_channels.len().saturating_sub(1) {
            patterns.push((
                format!(r"^speaker_encoder\.blocks\.{i}\."),
                format!("speaker_encoder.blocks.{}.", i - 1),
            ));
        }
    }
    KeyRemapper::from_patterns(patterns).map_err(|err| format!("invalid remapping: {err}"))
}
