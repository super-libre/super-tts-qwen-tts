// Vendored verbatim, so the whole of the example's API is here whether or not
// this backend calls it — a module trimmed to what today's caller uses could
// not be diffed against upstream, which is the point of vendoring it as a
// unit. See the provenance note below.
#![allow(dead_code)]
// This crate lints at `clippy::pedantic`; Burn does not. Editing the vendored
// code to satisfy lints it was never written against would cost exactly the
// diffability the module exists for, so the lints are turned off here instead.
// Everything outside `qwen3` is still held to them.
#![allow(clippy::pedantic)]

//! Qwen3-TTS: text-to-speech from the Qwen team, ported to Burn.
//!
//! # Provenance
//!
//! Vendored from the `qwen3-tts` example of
//! <https://github.com/jorge-menjivar/burn>, branch `qwen3-tts`, commit
//! `9cd83cc49`. The edits are mechanical, but for the one noted below: `crate::` became
//! `crate::qwen3::` because these modules are a module of the backend rather
//! than a crate of their own, and the example's `audio` module (a wav reader
//! and writer with a resampler) is left behind — this backend reads no file and
//! writes none: it streams samples to the daemon, and the daemon hands it a
//! cloning reference already decoded to mono 24 kHz, which is the rate both the
//! speaker encoder and the codec encoder want. The longer path pushes two
//! lines past the hundred columns rustfmt wraps at, so those two are
//! rewrapped; nothing else differs. Re-syncing is a
//! diff against that path in the fork, after undoing the rename:
//!
//! ```sh
//! diff <(sed 's/crate::qwen3::/crate::/g' src/qwen3/model.rs) \
//!      ../burn/examples/qwen3-tts/src/model.rs
//! ```
//!
//! `Cargo.toml` pins Burn at that same commit, so these modules and the crates
//! they compile against come from one revision of one branch.
//!
//! One edit is deliberate and waits to be carried back to the fork:
//! `Decoder::context_frames` in [`speech_tokenizer`] counted one attention
//! window where the decoder's stacked layers compound, and nothing for the
//! convolutions after the transformer, so a stream decoded behind it lost most
//! of its history at every seam. It now counts both, with the receptive-field
//! arithmetic and its tests beside it. Until the fork has it, the diff above
//! shows that hunk.
//!
//! It is vendored rather than depended on because the example crate is a
//! demo: it also pulls `clap`, `hf-hub` and `tokenizers/onig`, and cargo
//! unifies features across a dependency graph, so taking it whole would put a
//! C regex library and an HTTP stack into a backend that is cross-compiled
//! and runs with no network at all. The modules below need only `burn`.
//!
//! See [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) and the models on the hub, e.g.
//! `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`.
//!
//! The model generates 12.5 Hz frames of 16 codec tokens that the speech tokenizer decoder in
//! [`speech_tokenizer`] turns into 24 kHz audio. Each frame is produced in two steps:
//!
//! - the *talker*, a Qwen3 transformer fed with summed text and codec embeddings, predicts the
//!   first codebook entry of the next frame,
//! - the *code predictor*, a small Qwen3 transformer conditioned on the talker hidden state,
//!   autoregressively predicts the 15 remaining codebook entries of that frame.
//!
//! The `Base` checkpoints clone a voice from a recording: the [`speaker_encoder`] turns it into
//! an embedding the talker is conditioned on, and the speech tokenizer's encoder turns it into
//! codes the talker continues from, see [`model::Prompt`].

pub mod config;
pub mod model;
pub mod sampling;
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "metal",
    feature = "vulkan",
    feature = "wgpu",
    feature = "cpu"
))]
pub mod sampling_kernel;
pub mod speaker_encoder;
pub mod speech_tokenizer;
pub mod transformer;

use burn::nn::{LinearConfig, LinearLayout};
use burn_store::burn_pack::Tensor as PackTensor;
use burn_store::{ModuleAdapter, ModuleContext};

/// The configuration of every linear layer of this example: the weight keeps the column-major,
/// `[d_output, d_input]` layout of the checkpoints.
///
/// This is not only about loading the weights without transposing them. Generating a frame runs
/// 103 single-token transformer forwards whose linear layers multiply a single row by the weight
/// matrix, and the matmul kernels available for that product are very sensitive to the layout of
/// the matrix: on an RTX 3090 they take ~55 µs with a row-major weight and ~12 µs, the memory
/// bandwidth limit, with a column-major one. The layout of the weights alone changes the frame
/// rate by 2x.
pub(crate) fn linear_config(d_input: usize, d_output: usize) -> LinearConfig {
    LinearConfig::new(d_input, d_output).with_layout(LinearLayout::Col)
}

/// Loads the PyTorch checkpoints into modules built with [`linear_config`].
///
/// Like [`burn_store::PyTorchToBurnAdapter`] this renames the parameters of the normalization
/// layers, `weight` and `bias` in PyTorch, `gamma` and `beta` in Burn. Unlike it, the linear
/// weights are left alone: a column-major linear layer stores its weight as
/// `[d_output, d_input]`, which is exactly the PyTorch layout, so there is nothing to transpose.
#[derive(Debug, Clone, Default)]
pub(crate) struct CheckpointAdapter;

impl CheckpointAdapter {
    fn is_normalization_layer(module_type: &str) -> bool {
        matches!(
            module_type,
            "Struct:BatchNorm" | "Struct:LayerNorm" | "Struct:GroupNorm" | "Struct:RmsNorm"
        )
    }
}

impl ModuleAdapter for CheckpointAdapter {
    fn adapt(&self, mut tensor: PackTensor, ctx: ModuleContext<'_>) -> PackTensor {
        let Some(module_type) = ctx.module_type() else {
            return tensor;
        };
        if !Self::is_normalization_layer(module_type) {
            return tensor;
        }
        let start = tensor.name.rfind('.').map_or(0, |dot| dot + 1);
        let renamed = match &tensor.name[start..] {
            "weight" => "gamma",
            "bias" => "beta",
            _ => return tensor,
        };
        tensor.name.truncate(start);
        tensor.name.push_str(renamed);
        tensor
    }

    /// The store looks the parameters up under their Burn names, which the checkpoint does not
    /// use for the normalization layers.
    fn get_alternative_param_name(&self, param_name: &str, module_type: &str) -> Option<String> {
        if !Self::is_normalization_layer(module_type) {
            return None;
        }
        match param_name {
            "gamma" => Some("weight".to_string()),
            "beta" => Some("bias".to_string()),
            _ => None,
        }
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Turns the outcome of loading a checkpoint into an error unless every parameter of the module
/// was filled in. Tensors of the file that no parameter claims are expected: the codec encoder
/// is only loaded for voice cloning, and ships more codebooks than the talker uses.
pub(crate) fn check_apply_result(
    name: &str,
    result: &burn_store::ApplyResult,
) -> Result<(), String> {
    if !result.errors.is_empty() {
        let errors: Vec<String> = result.errors.iter().map(|e| e.to_string()).collect();
        return Err(format!("{name}: {}", errors.join(", ")));
    }
    if !result.missing.is_empty() {
        let missing: Vec<&str> = result
            .missing
            .iter()
            .take(5)
            .map(|(path, _)| path.as_str())
            .collect();
        return Err(format!(
            "{name}: {} parameters were not found in the checkpoint, e.g. {missing:?}",
            result.missing.len()
        ));
    }
    Ok(())
}

/// The device the unit tests run on.
#[cfg(test)]
pub(crate) fn test_device() -> burn::prelude::Device {
    #[cfg(feature = "cuda")]
    return burn::prelude::Device::cuda(0);
    #[cfg(not(feature = "cuda"))]
    burn::prelude::Device::default()
}
