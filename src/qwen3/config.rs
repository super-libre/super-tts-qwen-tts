//! Configuration of the Qwen3-TTS checkpoints, as shipped in their `config.json` files.

use std::collections::HashMap;

use burn::prelude::*;
use burn::tensor::activation::{gelu, silu};
use serde::Deserialize;

/// Activation of the MLP blocks. Every published checkpoint uses `silu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Silu,
    Gelu,
}

impl Activation {
    pub fn forward<const D: usize>(&self, xs: Tensor<D>) -> Tensor<D> {
        match self {
            Self::Silu => silu(xs),
            Self::Gelu => gelu(xs),
        }
    }
}

/// The small transformer predicting the codebooks `1..num_code_groups` of a frame.
#[derive(Debug, Clone, Deserialize)]
pub struct CodePredictorConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub num_code_groups: usize,
    pub hidden_act: Activation,
    #[serde(default)]
    pub attention_bias: bool,
}

/// Value of the `spk_is_dialect` entries: either `false` or the name of a dialect that
/// overrides the language tag when the language is Chinese (or auto).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Dialect {
    Flag(bool),
    Name(String),
}

#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub text_hidden_size: usize,
    pub text_vocab_size: usize,
    pub num_code_groups: usize,
    pub hidden_act: Activation,
    #[serde(default)]
    pub attention_bias: bool,
    pub code_predictor_config: CodePredictorConfig,
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_pad_id: u32,
    pub codec_think_id: u32,
    pub codec_nothink_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    #[serde(default)]
    pub codec_language_id: HashMap<String, u32>,
    #[serde(default)]
    pub spk_id: HashMap<String, u32>,
    #[serde(default)]
    pub spk_is_dialect: HashMap<String, Dialect>,
}

/// The `config.json` of a Qwen3-TTS checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub talker_config: TalkerConfig,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    pub im_start_token_id: u32,
    pub im_end_token_id: u32,
    pub assistant_token_id: u32,
    /// `base`, `custom_voice` or `voice_design`.
    #[serde(default)]
    pub tts_model_type: String,
    #[serde(default)]
    pub tts_model_size: String,
    /// The speaker encoder of the Base checkpoints, which clone a voice from a recording.
    #[serde(default)]
    pub speaker_encoder_config: Option<SpeakerEncoderConfig>,
}

/// The ECAPA-TDNN speaker encoder of the Base checkpoints, the `speaker_encoder_config` section
/// of `config.json`. The checkpoints only spell out `enc_dim` and `sample_rate`, the rest are
/// the defaults of the reference implementation.
#[derive(Debug, Clone, Deserialize)]
pub struct SpeakerEncoderConfig {
    #[serde(default = "default_mel_dim")]
    pub mel_dim: usize,
    /// The size of the embedding, the talker's hidden size.
    pub enc_dim: usize,
    #[serde(default = "default_enc_channels")]
    pub enc_channels: Vec<usize>,
    #[serde(default = "default_enc_kernel_sizes")]
    pub enc_kernel_sizes: Vec<usize>,
    #[serde(default = "default_enc_dilations")]
    pub enc_dilations: Vec<usize>,
    #[serde(default = "default_enc_attention_channels")]
    pub enc_attention_channels: usize,
    #[serde(default = "default_enc_res2net_scale")]
    pub enc_res2net_scale: usize,
    #[serde(default = "default_enc_se_channels")]
    pub enc_se_channels: usize,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: usize,
}

fn default_mel_dim() -> usize {
    128
}
fn default_enc_channels() -> Vec<usize> {
    vec![512, 512, 512, 512, 1536]
}
fn default_enc_kernel_sizes() -> Vec<usize> {
    vec![5, 3, 3, 3, 1]
}
fn default_enc_dilations() -> Vec<usize> {
    vec![1, 2, 3, 4, 1]
}
fn default_enc_attention_channels() -> usize {
    128
}
fn default_enc_res2net_scale() -> usize {
    8
}
fn default_enc_se_channels() -> usize {
    128
}
fn default_sample_rate() -> usize {
    24000
}

/// The decoder of the speech tokenizer, the `decoder_config` section of
/// `speech_tokenizer/config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub num_quantizers: usize,
    #[serde(default = "default_num_semantic_quantizers")]
    pub num_semantic_quantizers: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    pub hidden_act: Activation,
    #[serde(default)]
    pub attention_bias: bool,
}

fn default_num_semantic_quantizers() -> usize {
    1
}

impl DecoderConfig {
    /// Number of audio samples produced per codec frame.
    pub fn total_upsample(&self) -> usize {
        self.upsample_rates.iter().product::<usize>()
            * self.upsampling_ratios.iter().product::<usize>()
    }
}

/// The encoder of the speech tokenizer, the `encoder_config` section of
/// `speech_tokenizer/config.json`: the `MimiConfig` of `transformers`.
#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    pub audio_channels: usize,
    pub num_filters: usize,
    /// The strides of the convolutional stages, listed for the decoder: the encoder walks them
    /// backwards.
    pub upsampling_ratios: Vec<usize>,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub dilation_growth_rate: usize,
    pub compress: usize,
    pub num_residual_layers: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub intermediate_size: usize,
    pub hidden_act: Activation,
    pub norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    /// How many frames a frame of the transformer attends to, itself included.
    pub sliding_window: Option<usize>,
    pub num_quantizers: usize,
    #[serde(default = "default_num_semantic_quantizers")]
    pub num_semantic_quantizers: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub vector_quantization_hidden_dimension: usize,
    pub sampling_rate: usize,
    /// The frame rate of the codes, 12.5 Hz.
    #[serde(rename = "_frame_rate", default = "default_frame_rate")]
    pub frame_rate: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_true")]
    pub use_causal_conv: bool,
    #[serde(default)]
    pub use_conv_shortcut: bool,
    #[serde(default = "default_pad_mode")]
    pub pad_mode: String,
}

fn default_frame_rate() -> f64 {
    12.5
}
fn default_true() -> bool {
    true
}
fn default_pad_mode() -> String {
    "constant".to_string()
}

impl EncoderConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// The frame rate of the convolutional stack, before the downsampling convolution.
    pub fn conv_frame_rate(&self) -> f64 {
        self.sampling_rate as f64 / self.upsampling_ratios.iter().product::<usize>() as f64
    }
}

/// The `speech_tokenizer/config.json` file. The decoder turns codes into audio; the encoder,
/// only loaded for voice cloning, turns a reference recording into codes.
#[derive(Debug, Clone, Deserialize)]
pub struct SpeechTokenizerConfig {
    pub decoder_config: DecoderConfig,
    pub encoder_config: EncoderConfig,
    pub input_sample_rate: usize,
    pub output_sample_rate: usize,
    pub decode_upsample_rate: usize,
    /// Audio samples per frame of codes.
    pub encode_downsample_rate: usize,
    /// How many of the encoder's codebooks the talker uses.
    pub encoder_valid_num_quantizers: usize,
}
