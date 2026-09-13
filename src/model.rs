// SPDX-License-Identifier: GPL-3.0-only
//! The Qwen TTS session: talker, code predictor, and the codec.
//!
//! Synthesis runs in two halves. The *talker* — a Qwen transformer — generates
//! frames of codec tokens, autoregressively, and the codec *decoder* turns
//! those tokens into audio. Because the frames arrive one at a time, audio can
//! be decoded and sent while the rest of the utterance is still being
//! generated, which is what keeps time-to-first-sound near a second instead of
//! near the length of the utterance.
//!
//! The model itself is Burn's, vendored in [`crate::qwen3`]. Everything here is
//! the part that faces the daemon: which device to run on, how a `/v1` request
//! becomes a prompt, and how generated frames become streamed samples.
//!
//! # Serving a later generation
//!
//! Nothing here names Qwen3, and the parts that vary between checkpoints are
//! read from the files rather than assumed: the frame rate and the sample rate
//! come from the codec config, the speakers and languages from the talker
//! config. Three things would still have to be checked before adding a new
//! generation to the manifest — [`crate::qwen3`] must implement it, [`Kind`]
//! must recognize its `tts_model_type`, and the chat template in
//! [`crate::prompt`] must be the one it was trained with.
//!
//! Everything here is synchronous and compute-bound, and the model wants `&mut`,
//! so the whole thing lives behind one lock and callers run it off the async
//! runtime.

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use burn::prelude::{Device, Tensor};
use burn::tensor::DType;
use tokenizers::Tokenizer;

use crate::lang;
use crate::prompt;
use crate::qwen3::config::{Config, SpeechTokenizerConfig};
use crate::qwen3::model::{GenerationConfig, IclReference, Prompt, Qwen3Tts as Talker, Voice};
use crate::qwen3::speech_tokenizer::SpeechTokenizer;
use crate::voices::{self, Requested};

/// Codec frames per second of the 12 Hz tokenizer the Qwen3 checkpoints use.
///
/// The frame caps below were sized against this rate, and the tests that check
/// them are all that read it. A loaded model computes its own rate from the
/// codec it actually mapped, so a generation whose tokenizer runs at a
/// different rate needs no edit here — only the caps would want revisiting.
#[cfg(test)]
const NOMINAL_FRAMES_PER_SECOND: f64 = 12.5;

/// Longest utterance the talker is allowed to generate, in codec frames.
///
/// About 164 seconds of speech. Far more than the `max_input_chars` of any
/// model here can produce — a 500-character chunk is well under a minute — so
/// it never truncates real speech, and it bounds a talker that fails to emit
/// its end-of-speech token before the daemon's 32 MiB per-response audio cap
/// aborts the stream with nothing useful said.
const MAX_NEW_TOKENS: usize = 2048;

/// The text the warm-up generates from, and then throws away.
///
/// Only its length is load-bearing; nothing listens to the words. Prefixes of
/// it are taken at [`WARM_UP_LENGTHS`], so it has to be at least as long as the
/// largest of them.
const WARM_UP_TEXT: &str = "\
    Speech starts before the utterance is finished, because the talker emits \
    codec frames and the decoder turns them into audio while the rest of the \
    sentence is still being generated. None of this text is ever heard by \
    anyone at all: it exists so that the kernels behind a request are compiled \
    and tuned while the model is still being loaded, rather than while somebody \
    is waiting to hear the first syllable of something they asked for. The \
    length of this passage is the only thing about it that matters at all.";

/// Prompt lengths, in characters, the warm-up generates at.
///
/// A ladder rather than one length, because `CubeCL` keys its autotune entries
/// on the shape of the problem and the prefix pass takes its shape from the
/// prompt. The entries are rounded, so one prompt covers a span of nearby
/// lengths — but only a span. Measured on a cold cache with a single 300
/// character warm-up, requests of 220 and 300 characters were fast while 140,
/// 380 and 430 each stalled for seconds on their own first request, one of them
/// generating at 2.7 frames per second for a whole utterance.
///
/// These five span the `max_input_chars` every model in the manifest declares.
/// A request longer than the last of them pays its own tuning once, which is
/// the same deal as before and now only reachable by raising that cap.
const WARM_UP_LENGTHS: [usize; 5] = [60, 150, 260, 370, 480];

/// Frames the deep warm-up generates, when it runs at all.
///
/// The rungs above each stop after one chunk, which tunes the shapes that
/// depend on the *prompt*. It leaves one thing cold: the talker's key/value
/// cache starts a few hundred positions past the prompt and doubles when a
/// generation outgrows it, re-capturing the graph and tuning the wider shapes
/// as it goes. Twenty-five frames never reach that, so the first real request
/// did — one measured at 2.6 frames per second for a whole utterance, ten
/// seconds of audio taking fifty-one to produce, with its time to first audio
/// perfectly healthy because the cost was spread through the generation rather
/// than sitting in front of it.
///
/// Thirty seconds of speech, which is past the first doubling for every prompt
/// length above.
const DEEP_WARM_UP_FRAMES: usize = 375;

/// The id the warm-up registers its invented voice under.
///
/// Not a uuid, so it cannot collide with one the daemon sends: the daemon's
/// ids are `voice:<uuid>` and this one is never parsed from a request.
const WARM_UP_VOICE: &str = "warm-up";

/// Seconds of invented audio the warm-up clones from.
///
/// A real reference is whatever the user recorded, trimmed to the manifest's
/// `clone_ref_seconds`, so its exact length is not knowable here. What this
/// buys is the part that does not depend on it: a generation conditioned on an
/// embedding rather than on a speaker, which is every request such a checkpoint
/// will ever serve. The encoders are tuned when a real voice is registered.
const WARM_UP_REFERENCE_SECONDS: u32 = 5;

/// Audio for the warm-up to clone a voice from.
///
/// Two tones under a slow tremolo rather than silence or noise: silence gives
/// the mel filterbank a log of zero, and the point is only to put a signal with
/// energy across the band through the encoders, not to sound like anything.
#[allow(clippy::cast_precision_loss)]
fn warm_up_reference(sample_rate: u32, seconds: u32) -> Vec<f32> {
    let rate = sample_rate as f32;
    let samples = (sample_rate as usize) * (seconds as usize);
    (0..samples)
        .map(|i| {
            let t = i as f32 / rate;
            let tremolo = 0.5f32.mul_add((std::f32::consts::TAU * 3.0 * t).sin(), 0.5);
            0.25 * tremolo
                * ((std::f32::consts::TAU * 220.0 * t).sin()
                    + 0.5 * (std::f32::consts::TAU * 1310.0 * t).sin())
        })
        .collect()
}

/// How slow the ladder has to be before the deep warm-up is worth running.
///
/// The work the deep pass does is tuning, and tuning is kept on disk, so it is
/// paid once per host and build rather than once per process — which is why it
/// must not be paid on every load. The ladder itself is the probe: it takes a
/// second or two against a warm cache and a minute or more against a cold one,
/// so a slow ladder means this process is the one that would have handed the
/// bill to the first request.
const DEEP_WARM_UP_THRESHOLD: std::time::Duration = std::time::Duration::from_secs(10);

/// Frames decoded at a time while streaming: two seconds of audio.
///
/// This is the floor on how often audio can be sent, so it is also most of
/// time-to-first-sound. Smaller chunks send sooner but re-decode the context
/// below more often.
const STREAM_CHUNK_FRAMES: usize = 25;

/// Frames of context each streamed chunk is decoded with and then discarded.
///
/// The decoder is convolutional, so a chunk decoded alone does not match the
/// same frames decoded as part of the whole: the seam is audible. Decoding each
/// chunk with the frames before it and dropping their audio makes the streamed
/// result identical to a single decode. Below about 50 the seams return, above
/// it there is nothing left to gain.
const STREAM_CONTEXT_FRAMES: usize = 50;

/// The one shape every chunk reaches the decoder as: a full context plus a full
/// chunk.
///
/// A GPU backend compiles and tunes its kernels per shape, so a decoder fed the
/// lengths a stream naturally produces — 25 frames, then 50, then 75, then
/// whatever is left over at the end — pays that cost four times an utterance on
/// a cold cache. Padding every chunk up to one window pays it once. The decoder
/// is causal and [`SpeechTokenizer::decode_window`] truncates the padding's
/// samples, so the audio is unchanged.
const DECODE_WINDOW_FRAMES: usize = STREAM_CONTEXT_FRAMES + STREAM_CHUNK_FRAMES;

/// How many seconds of speech `frames` is, at `per_second` frames per second.
///
/// The cast is exact for every count this can be handed: the talker is capped
/// at [`MAX_NEW_TOKENS`] frames, four orders of magnitude below where an `f64`
/// starts losing integers.
#[allow(clippy::cast_precision_loss)]
fn seconds_of(frames: usize, per_second: f64) -> f64 {
    frames as f64 / per_second
}

/// Which family a checkpoint belongs to, from its `tts_model_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Nine predefined speakers, optionally steered by an instruction.
    CustomVoice,
    /// No speakers; the voice is built from a natural-language description.
    VoiceDesign,
    /// No speakers either; the voice is cloned from a recording the daemon
    /// registers over `POST /v1/voices`.
    Base,
}

impl Kind {
    fn parse(model_type: &str) -> Result<Self> {
        match model_type {
            "custom_voice" => Ok(Self::CustomVoice),
            "voice_design" => Ok(Self::VoiceDesign),
            "base" => Ok(Self::Base),
            other => bail!("unsupported checkpoint type {other:?}"),
        }
    }
}

/// What a request's `voice` resolved to, in the terms the model conditions on.
///
/// The three shapes the protocol defines do not map one to one onto the three
/// checkpoint families — a described voice is carried as an instruction, not as
/// a voice — so this is the vocabulary in between.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Conditioning {
    /// Nothing: the checkpoint speaks in whatever voice it defaults to.
    None,
    /// One of the checkpoint's predefined speakers.
    Speaker(String),
    /// A description, which reaches the model as an instruction rather than as
    /// a voice.
    Description(String),
    /// A voice registered over `POST /v1/voices`, by the uuid of its wire id.
    Cloned(String),
}

/// What a validated request carries into synthesis.
///
/// Built by [`QwenTts::prepare`] before the response status line is sent, so a
/// bad voice or language is still a `400` rather than an error frame inside an
/// otherwise successful stream.
#[derive(Debug, Clone)]
pub struct Prepared {
    input_ids: Vec<u32>,
    instruct_ids: Option<Vec<u32>>,
    language: String,
    voice: Conditioning,
    text_chars: u32,
}

impl Prepared {
    /// Length of the request text in characters, for the alignment mark.
    ///
    /// Counted here rather than by the caller so the span a mark reports is
    /// measured against the same text that was tokenized.
    #[must_use]
    pub fn text_chars(&self) -> u32 {
        self.text_chars
    }
}

/// Why a request cannot be synthesized, in the daemon's own vocabulary.
#[derive(Debug)]
pub enum RequestError {
    /// The voice is not one this checkpoint can resolve.
    UnknownVoice(String),
    /// The language is not one this checkpoint speaks.
    UnsupportedLanguage(String),
    /// The text could not be tokenized.
    Tokenize(String),
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownVoice(m) | Self::UnsupportedLanguage(m) | Self::Tokenize(m) => {
                f.write_str(m)
            }
        }
    }
}

/// How a synthesis ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The talker emitted end-of-speech, or hit [`MAX_NEW_TOKENS`].
    Finished,
    /// The caller asked to stop — a cancel, or the daemon hanging up.
    Stopped,
}

/// A cloned voice, derived once from its reference clip and held for the life
/// of the loaded model.
///
/// Both halves come from the same recording. The embedding is what the talker
/// is conditioned on; the codes and the transcript are the in-context example
/// the reference implementation also feeds it when it has the words. The
/// embedding is a tensor, so this — and the model it belongs to — never leaves
/// the model thread.
struct RegisteredVoice {
    embedding: Tensor<1>,
    /// The reference clip as codec frames, with the tokens of its transcript.
    /// `None` when the clip was registered without one.
    icl: Option<(Vec<u32>, Vec<u32>)>,
}

/// A loaded Qwen TTS checkpoint and the codec that turns its frames into audio.
pub struct QwenTts {
    talker: Talker,
    speech_tokenizer: SpeechTokenizer,
    tokenizer: Tokenizer,
    kind: Kind,
    /// `true` for the 0.6B checkpoints, which ignore an instruction.
    ignores_instructions: bool,
    speakers: Vec<String>,
    default_speaker: Option<String>,
    device_name: &'static str,
    sample_rate: u32,
    /// Codec frames generated per second of speech, computed from the codec
    /// that was actually mapped rather than assumed from the generation.
    frames_per_second: f64,
    /// Cloned voices the daemon has registered against this load, keyed by the
    /// uuid of their wire id. Empty for every checkpoint but a Base one, and
    /// deliberately not persisted: the daemon re-registers after every load.
    voices: HashMap<String, RegisteredVoice>,
}

// The model has no `Debug` worth printing and `Tokenizer` has none at all, so
// this is written by hand. The heavy fields are named as opaque rather than
// omitted: a `Debug` that silently drops a field misrepresents the value.
impl std::fmt::Debug for QwenTts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenTts")
            .field("talker", &"<qwen3::model::Qwen3Tts>")
            .field("speech_tokenizer", &"<qwen3::speech_tokenizer>")
            .field("tokenizer", &"<tokenizers::Tokenizer>")
            .field("kind", &self.kind)
            .field("ignores_instructions", &self.ignores_instructions)
            .field("speakers", &self.speakers)
            .field("default_speaker", &self.default_speaker)
            .field("device_name", &self.device_name)
            .field("sample_rate", &self.sample_rate)
            .field("frames_per_second", &self.frames_per_second)
            // The voices themselves are tensors with nothing printable in them;
            // which ids are held is the part worth seeing.
            .field("voices", &self.voices.keys())
            .finish()
    }
}

/// The accelerator this build was compiled for, as the manifest names it.
///
/// One build serves one accelerator: Burn's backends are cargo features, and
/// the asset that carries this binary declares the matching `accel`. The
/// constant is what `GET /v1/status` reports and what the mismatch warning in
/// [`select_device`] compares against.
const BUILT_FOR: &str = if cfg!(feature = "cuda") {
    "cuda"
} else if cfg!(feature = "rocm") {
    "rocm"
} else if cfg!(feature = "vulkan") {
    "vulkan"
} else if cfg!(feature = "metal") {
    "metal"
} else if cfg!(feature = "wgpu") {
    "wgpu"
} else {
    "cpu"
};

/// Whether [`BUILT_FOR`] is a GPU, which decides the talker's dtype.
const ON_GPU: bool = cfg!(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal",
    feature = "wgpu"
));

/// The device this build runs on, and the name to report for it.
///
/// The daemon sends the accelerator the *installed asset* targets, and it is
/// absent when the daemon has no record of one — an install from a local
/// directory, for instance. Either way the answer is the same: a build has
/// exactly one backend compiled into it, so the request is a cross-check
/// rather than a choice, and a mismatch is worth a line in the log because it
/// means the wrong asset was installed for the host.
fn select_device(requested: Option<&str>) -> (Device, &'static str) {
    if let Some(d) = requested.map(str::trim).filter(|s| !s.is_empty())
        && !d.eq_ignore_ascii_case(BUILT_FOR)
    {
        log::warn!(
            "the daemon asked for {d:?} but this build only has {BUILT_FOR}; using {BUILT_FOR}"
        );
    }
    // One arm per backend, in the order a build that somehow enabled several
    // would prefer them. The cascade mirrors Burn's own example: cargo
    // features are additive, so the arms have to exclude each other by hand.
    #[cfg(feature = "cuda")]
    return (Device::cuda(0), "cuda");
    #[cfg(all(not(feature = "cuda"), feature = "rocm"))]
    return (Device::rocm(0), "rocm");
    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "vulkan"))]
    return (
        Device::vulkan(burn::prelude::DeviceKind::DefaultDevice),
        "vulkan",
    );
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        feature = "metal"
    ))]
    return (
        Device::metal(burn::prelude::DeviceKind::DefaultDevice),
        "metal",
    );
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        not(feature = "metal"),
        feature = "wgpu"
    ))]
    return (
        Device::wgpu(burn::prelude::DeviceKind::DefaultDevice),
        "wgpu",
    );
    // Both CPU backends report `cpu`: they are one accelerator as far as the
    // manifest and the daemon are concerned, and which one a build carries is
    // a build decision rather than something the host can act on.
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        not(feature = "metal"),
        not(feature = "wgpu"),
        feature = "cpu"
    ))]
    return (Device::cpu(), "cpu");
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        not(feature = "metal"),
        not(feature = "wgpu"),
        not(feature = "cpu"),
        feature = "flex"
    ))]
    return (Device::flex(), "cpu");
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        not(feature = "metal"),
        not(feature = "wgpu"),
        not(feature = "cpu"),
        not(feature = "flex")
    ))]
    (Device::default(), "cpu")
}

/// Point `CubeCL`'s kernel cache at the directory the daemon granted.
///
/// `CubeCL` compiles every kernel it meets at runtime — a few hundred of them,
/// about twenty seconds — and keeps them on disk so only the first run of a
/// build pays. Left to itself it would write under `$HOME`, which the sandbox
/// mounts read-only, so it would recompile on every load. The daemon hands
/// over a writable directory for exactly this; see `SUPER_TTS_BACKEND_CACHE_DIR`
/// in the subprocess protocol.
///
/// Must run before the first device is created, because the configuration is
/// frozen the first time anything reads it. A backend spawned by an older
/// daemon gets no such directory and keeps `CubeCL`'s own default.
pub fn configure_kernel_cache(cache_dir: Option<&Path>) {
    use burn::cubecl::config::cache::CacheConfig;
    use burn::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.compilation.cache = true;
    if let Some(dir) = cache_dir {
        config.environment.path = CacheConfig::Directory(dir.to_path_buf());
    }
    // `false` means something already read the configuration and this call is
    // too late to matter. Nothing in this backend touches a device before
    // `main` calls this, so it is a guard rather than a case to handle.
    if !CubeClRuntimeConfig::try_set(config) {
        log::warn!("the CubeCL configuration was already read; the kernel cache keeps its default");
    }
}

/// The pre-warmed autotune cache, shipped in the release tarball beside the
/// binary rather than downloaded.
///
/// One file for every GPU, not one per architecture and not one per model.
/// Nothing in the cache is keyed by model — the namespaces are keyed by CubeCL
/// version, device and kernel family — so entries a machine cannot use are
/// simply never looked up. That makes merging every architecture into one file
/// free for the machines that do not match, and it is what lets this ride along
/// with the executable instead of needing a per-host download.
const KERNEL_BUNDLE: &str = "kernels/autotune.bundle";

/// Seed the kernel cache from the bundle in this build, if there is one.
///
/// This is the cold-load cost bought back. `CubeCL` compiles every kernel it
/// meets at runtime and then *autotunes* — several candidates benchmarked per
/// operation per shape — and the tuning is the slow half: a load that
/// recompiles everything but keeps its tuning takes under a minute against four
/// from cold. A bundle is that tuning, done once per GPU and shipped.
///
/// Best-effort by construction, and every failure here is a slow load rather
/// than a broken one. Entries for another GPU import cleanly and are then never
/// looked up — every namespace carries the `CubeCL` version and a device
/// fingerprint — so the worst case is a little wasted disk. That is why this
/// warns and returns instead of failing the load.
///
/// Must run after [`configure_kernel_cache`], which decides *which* environment
/// gets filled, and before anything touches a device.
pub fn import_kernel_bundle(backend_dir: &Path) {
    let path = backend_dir.join(KERNEL_BUNDLE);
    if !path.is_file() {
        log::info!(
            "no {KERNEL_BUNDLE} in this build; every operation is tuned on this machine, \
             which is the slow part of a first load"
        );
        return;
    }

    let started = std::time::Instant::now();
    let bundle = match burn::cubecl::bundle::open(&path) {
        Ok(bundle) => bundle,
        Err(err) => {
            log::warn!("ignoring {}: {err}", path.display());
            return;
        }
    };
    let report = burn::cubecl::bundle::import(bundle.as_ref());
    // `skipped` counts keys the cache already held, so a second load of the
    // same model reports everything skipped — the import is insert-only and
    // idempotent, and re-running it is how a bundle updated in place is picked
    // up without any marker to keep in sync.
    log::info!(
        "imported {} kernel-cache entries from {KERNEL_BUNDLE} in {:.1?} \
         ({} namespaces, {} already present, {} refused)",
        report.imported,
        started.elapsed(),
        report.namespaces.len(),
        report.skipped,
        report.failed
    );
}

/// Write the current kernel cache out as a bundle, for shipping.
///
/// The other half of [`import_kernel_bundle`], and the reason a bundle can
/// exist at all: there is no way to produce one except by running the work on
/// the hardware it is for. Call it after a load, which warms the cache by
/// running the same ladder a real first request would.
///
/// # Errors
/// Returns an error if the cache cannot be read or `out` cannot be written.
pub fn export_kernel_bundle(out: &Path, name: &str, everything: bool) -> Result<()> {
    use burn::cubecl::bundle::{BundleFormat, ExportOptions};

    // Autotune by default, and the difference is not small: on an RTX 3090 the
    // tuning results for this model are about half a megabyte, while the
    // compiled PTX beside them is 147. The half megabyte is also the expensive
    // half to produce — a load that recompiles every kernel but keeps its
    // tuning takes under a minute, against four from cold — and the durable
    // one, since a compiled kernel is keyed by the source that generated it
    // and dies with the next build while a tuning result is keyed by operation
    // and shape and does not.
    //
    // So the small file carries most of the win and survives releases, which
    // is what makes it shippable in the tarball. `everything` is there for
    // measuring what the other 147 MB would buy.
    let namespaces = if everything {
        Vec::new()
    } else {
        vec!["autotune".to_string(), "throughput".to_string()]
    };
    let options = ExportOptions {
        name: name.to_string(),
        format: BundleFormat::Sqlite,
        namespaces,
        ..Default::default()
    };
    let manifest =
        burn::cubecl::bundle::export(&[burn::cubecl::environment::path()], out, &options)
            .map_err(|err| anyhow!("exporting the kernel cache to {}: {err}", out.display()))?;
    // Megabytes to one decimal, which is what a person reading a release
    // listing wants; the integer division keeps clippy's precision lint out of
    // a log line.
    let tenths = std::fs::metadata(out).map_or(0, |m| m.len() * 10 / (1024 * 1024));
    log::info!(
        "wrote {} ({}.{} MB) for CubeCL {}",
        out.display(),
        tenths / 10,
        tenths % 10,
        manifest.cubecl_version
    );
    Ok(())
}

/// The one file layout this backend and its manifest agree on.
fn model_file(model_dir: &Path, relative: &str) -> Result<PathBuf> {
    let path = model_dir.join(relative);
    if !path.is_file() {
        bail!(
            "{} is missing; the daemon downloads it from `[[models.files]]` before loading",
            path.display()
        );
    }
    Ok(path)
}

impl QwenTts {
    /// Load a checkpoint from a backend directory.
    ///
    /// # Errors
    /// Returns an error if a declared file is missing, the checkpoint is of a
    /// family this backend does not serve, or the weights cannot be mapped.
    pub fn load(backend_dir: &Path, model_name: &str, device: Option<&str>) -> Result<Self> {
        let dir = backend_dir.join("models").join(model_name);
        let config_file = model_file(&dir, "config.json")?;
        let weights_file = model_file(&dir, "model.safetensors")?;
        let st_config_file = model_file(&dir, "speech_tokenizer/config.json")?;
        let st_weights_file = model_file(&dir, "speech_tokenizer/model.safetensors")?;
        let tokenizer_file = model_file(&dir, "tokenizer.json")?;

        let config: Config = serde_json::from_slice(&std::fs::read(&config_file)?)
            .with_context(|| format!("parsing {}", config_file.display()))?;
        let st_config: SpeechTokenizerConfig =
            serde_json::from_slice(&std::fs::read(&st_config_file)?)
                .with_context(|| format!("parsing {}", st_config_file.display()))?;
        let kind = Kind::parse(&config.tts_model_type)?;
        // The 0.6B checkpoints were not trained on instructions; the reference
        // implementation drops them, and so does this.
        let ignores_instructions = config.tts_model_size == "0b6";

        let (device, device_name) = select_device(device);
        // bf16 halves the talker's weights and its bandwidth on a GPU. On the
        // CPU it is slower than f32 rather than faster.
        let dtype = if ON_GPU { DType::BF16 } else { DType::F32 };
        log::info!(
            "loading {model_name} on {device_name} ({dtype:?}) from {}",
            dir.display()
        );

        let tokenizer = Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow!("loading {}: {e}", tokenizer_file.display()))?;

        let mut talker = Talker::load(&config, &weights_file, dtype, &device)
            .map_err(|e| anyhow!("building the talker from {}: {e}", weights_file.display()))?;
        // Replay the forward passes as captured graphs instead of walking the
        // operations again per frame: one driver call per pass rather than a
        // few hundred, which is most of the difference between this backend and
        // real time. Falls back to running eagerly on a device that cannot
        // capture, so it is safe to ask for unconditionally.
        talker.enable_talker_graphs();
        talker.enable_code_predictor_graphs();

        // The codec runs in f32 whatever the talker's dtype: it is a small part
        // of the compute and the decoder is where audible artefacts would show.
        //
        // The Base checkpoints also need the codec's *encoder*, to turn a
        // cloning reference into the codes an in-context example is made of.
        // Only they: it is weights and load time no other family would use.
        let speech_tokenizer = if kind == Kind::Base {
            SpeechTokenizer::load_with_encoder(&st_config, &st_weights_file, &device)
        } else {
            SpeechTokenizer::load(&st_config, &st_weights_file, &device)
        }
        .map_err(|e| anyhow!("building the codec from {}: {e}", st_weights_file.display()))?;
        // A Base checkpoint whose weights carry no speaker encoder can still
        // read text, but nothing it says would be in the requested voice, and
        // the daemon would have no way to learn that except by the audio
        // sounding wrong. Refusing at load names the reason once.
        if kind == Kind::Base && !talker.has_speaker_encoder() {
            bail!(
                "this Base checkpoint ships no speaker encoder, so it cannot clone a voice \
                 from reference audio"
            );
        }

        let speakers: Vec<String> = talker
            .supported_speakers()
            .into_iter()
            .map(str::to_string)
            .collect();
        // `ryan` is the reference implementation's own default; falling back to
        // the first speaker keeps a checkpoint with a different roster working.
        let default_speaker = speakers
            .iter()
            .find(|s| s.as_str() == "ryan")
            .or_else(|| speakers.first())
            .cloned();
        let sample_rate = u32::try_from(speech_tokenizer.output_sample_rate())
            .context("the codec declares an out-of-range output sample rate")?;
        // Read from the codec rather than assumed, so a generation whose
        // tokenizer runs at another rate reports its own.
        let samples_per_frame = speech_tokenizer.samples_per_frame();
        if samples_per_frame == 0 {
            bail!("the codec declares no samples per frame");
        }
        #[allow(clippy::cast_precision_loss)]
        let frames_per_second = f64::from(sample_rate) / samples_per_frame as f64;

        log::info!(
            "loaded: {kind:?}, {} speakers, {} languages, {sample_rate} Hz, \
             {frames_per_second} frames/s",
            speakers.len(),
            talker.supported_languages().len()
        );
        let mut model = Self {
            talker,
            speech_tokenizer,
            tokenizer,
            kind,
            ignores_instructions,
            speakers,
            default_speaker,
            device_name,
            sample_rate,
            frames_per_second,
            voices: HashMap::new(),
        };
        model.warm_up();
        Ok(model)
    }

    /// The rate of the audio this model produces, in Hz.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The device the model is actually running on, as `GET /v1/status` reports it.
    #[must_use]
    pub fn device_name(&self) -> &'static str {
        self.device_name
    }

    /// Validate a request and turn it into the tokens synthesis needs.
    ///
    /// Done before the response status line is sent, so a bad voice or language
    /// is a `400` the daemon can report as such rather than an error frame in a
    /// stream that already claimed success.
    ///
    /// `configured_voice` is the description the backend's own `[[options]]`
    /// ask for, already resolved by [`voices::configured`]. It fills in for a
    /// request that named no voice and is ignored by a request that named one:
    /// a setting is what the user wants by default, and the `voice` field is
    /// what they asked for this time.
    ///
    /// # Errors
    /// Returns [`RequestError`] when the voice or language is not one this
    /// checkpoint can resolve, or the text cannot be tokenized.
    pub fn prepare(
        &self,
        text: &str,
        voice: Requested<'_>,
        configured_voice: Option<&str>,
        language: Option<&str>,
        instructions: Option<&str>,
    ) -> Result<Prepared, RequestError> {
        // Not an error: the options are the backend's, not the model's, and a
        // user who configured a voice and then loaded a checkpoint that builds
        // none should keep the checkpoint they picked rather than lose every
        // request to a setting they cannot see from here.
        if configured_voice.is_some() && self.kind != Kind::VoiceDesign {
            log::debug!("this checkpoint designs no voice; ignoring the configured description");
        }
        let language = self.resolve_language(language)?;
        let conditioning = self.resolve_voice(voice, configured_voice)?;
        let description = match &conditioning {
            Conditioning::Description(d) => Some(d.as_str()),
            _ => None,
        };

        let instructions = instructions.map(str::trim).filter(|s| !s.is_empty());
        let instruct = match (description, instructions) {
            // A described voice and delivery guidance are one instruction to
            // the model, so they are joined rather than one being dropped.
            (Some(description), extra) => Some(prompt::join_instructions(description, extra)),
            (None, Some(extra)) if self.ignores_instructions => {
                log::info!("this checkpoint ignores instructions; dropping {extra:?}");
                None
            }
            (None, extra) => extra.map(str::to_string),
        };

        let encode = |s: &str| -> Result<Vec<u32>, RequestError> {
            self.tokenizer
                .encode(s, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| RequestError::Tokenize(format!("tokenizing the request: {e}")))
        };
        let input_ids = encode(&prompt::input_text(text))?;
        let instruct_ids = match &instruct {
            Some(i) => Some(encode(&prompt::instruct_text(i))?),
            None => None,
        };

        Ok(Prepared {
            input_ids,
            instruct_ids,
            language,
            voice: conditioning,
            text_chars: u32::try_from(text.chars().count()).unwrap_or(u32::MAX),
        })
    }

    /// Map a BCP-47 tag to the language name the talker conditions on.
    fn resolve_language(&self, language: Option<&str>) -> Result<String, RequestError> {
        let Some(tag) = language.map(str::trim).filter(|s| !s.is_empty()) else {
            return Ok(lang::AUTO.to_string());
        };
        let name = lang::to_model_name(tag).ok_or_else(|| {
            RequestError::UnsupportedLanguage(format!("this model does not speak {tag}"))
        })?;
        // The table and the checkpoint can disagree if a future checkpoint
        // drops a language, and conditioning on a name it does not know would
        // silently pick the wrong entry.
        if name != lang::AUTO && !self.talker.supported_languages().contains(&name) {
            return Err(RequestError::UnsupportedLanguage(format!(
                "this model does not speak {tag}"
            )));
        }
        Ok(name.to_string())
    }

    /// Resolve a request voice into what this checkpoint can condition on.
    ///
    /// `configured` is the backend's configured description, which only a
    /// VoiceDesign checkpoint has anywhere to put — the other two families
    /// condition on a speaker or a clone, and handing either a description
    /// would refuse every request from a user who set the option once.
    fn resolve_voice(
        &self,
        voice: Requested<'_>,
        configured: Option<&str>,
    ) -> Result<Conditioning, RequestError> {
        match (self.kind, voice) {
            (Kind::CustomVoice, Requested::Speaker(s)) => {
                if self.speakers.iter().any(|k| k == s) {
                    Ok(Conditioning::Speaker(s.to_string()))
                } else {
                    Err(RequestError::UnknownVoice(format!(
                        "unknown voice {s}; this model has {}",
                        self.speakers.join(", ")
                    )))
                }
            }
            (Kind::CustomVoice, Requested::Default) => Ok(self
                .default_speaker
                .clone()
                .map_or(Conditioning::None, Conditioning::Speaker)),
            (Kind::CustomVoice, Requested::Description(_)) => Err(RequestError::UnknownVoice(
                "this model speaks with one of its own voices, not a described one".to_string(),
            )),
            (Kind::VoiceDesign, Requested::Description(d)) => {
                Ok(Conditioning::Description(d.to_string()))
            }
            (Kind::VoiceDesign, Requested::Default) => Ok(Conditioning::Description(
                configured
                    .unwrap_or(voices::DEFAULT_DESCRIPTION)
                    .to_string(),
            )),
            (Kind::VoiceDesign, Requested::Speaker(s)) => Err(RequestError::UnknownVoice(format!(
                "unknown voice {s}; this model builds a voice from a description, \
                 so its voice ids look like desc:<description>"
            ))),
            // Registration is a separate request, and the daemon makes it
            // before the first synthesis naming the voice. An id that is not
            // here is one whose registration failed or never happened, which is
            // worth saying plainly rather than synthesizing in some other voice.
            (Kind::Base, Requested::Cloned(uuid)) => {
                if self.voices.contains_key(uuid) {
                    Ok(Conditioning::Cloned(uuid.to_string()))
                } else {
                    Err(RequestError::UnknownVoice(format!(
                        "the voice {uuid} was never registered with this load"
                    )))
                }
            }
            // A Base checkpoint conditioned on nothing speaks in a voice that
            // changes from one request to the next, so there is no sensible
            // default to fall back to.
            (Kind::Base, Requested::Default) => Err(RequestError::UnknownVoice(
                "this model speaks in a cloned voice, so a request has to name one".to_string(),
            )),
            (Kind::Base, Requested::Speaker(s) | Requested::Description(s)) => {
                Err(RequestError::UnknownVoice(format!(
                    "unknown voice {s}; this model clones a voice from a recording, \
                     so its voice ids look like voice:<uuid>"
                )))
            }
            (Kind::CustomVoice | Kind::VoiceDesign, Requested::Cloned(_)) => {
                Err(RequestError::UnknownVoice(
                    "this model cannot clone a voice from reference audio".to_string(),
                ))
            }
        }
    }

    /// Whether this checkpoint clones a voice from a recording.
    #[must_use]
    pub fn clones_voices(&self) -> bool {
        self.kind == Kind::Base
    }

    /// The sample rate a cloning reference has to arrive at.
    ///
    /// Read from the codec rather than assumed: the encoder and the speaker
    /// encoder both read the clip at the rate the checkpoint was trained on,
    /// and resampling is the daemon's job, not this backend's.
    #[must_use]
    pub fn reference_sample_rate(&self) -> u32 {
        u32::try_from(self.speech_tokenizer.input_sample_rate()).unwrap_or(u32::MAX)
    }

    /// Register a cloned voice from its reference recording, mono at
    /// [`Self::reference_sample_rate`].
    ///
    /// Derives everything the voice needs once and keys it by `uuid`, because
    /// the daemon sends one synthesis per sentence and deriving it per request
    /// would repeat this work for every sentence of a paragraph.
    ///
    /// A transcript is optional and is what separates the two ways this family
    /// clones: without one the talker is conditioned on the speaker embedding
    /// alone, with one the recording is also fed back as an in-context example,
    /// which is what the reference implementation does when it has the words.
    ///
    /// # Errors
    /// Returns an error if the checkpoint does not clone, or if deriving the
    /// embedding, encoding the recording or tokenizing the transcript fails.
    pub fn register_voice(
        &mut self,
        uuid: &str,
        transcript: Option<&str>,
        samples: &[f32],
    ) -> Result<()> {
        if !self.clones_voices() {
            bail!("this model does not clone voices");
        }
        if samples.is_empty() {
            bail!("the reference recording is empty");
        }
        let embedding = self
            .talker
            .speaker_embedding(samples)
            .map_err(|e| anyhow!("deriving the speaker embedding: {e}"))?;
        let icl = match transcript.map(str::trim).filter(|t| !t.is_empty()) {
            Some(transcript) => {
                let ref_codes = self
                    .speech_tokenizer
                    .encode(samples)
                    .map_err(|e| anyhow!("encoding the reference recording: {e}"))?;
                let ref_ids = self
                    .tokenizer
                    .encode(prompt::reference_text(transcript), false)
                    .map(|e| e.get_ids().to_vec())
                    .map_err(|e| anyhow!("tokenizing the reference transcript: {e}"))?;
                Some((ref_ids, ref_codes))
            }
            None => None,
        };
        log::info!(
            "registered the voice {uuid} from {:.1}s of audio{}",
            seconds_of(samples.len(), f64::from(self.reference_sample_rate())),
            if icl.is_some() {
                ", with its transcript"
            } else {
                ""
            }
        );
        self.voices
            .insert(uuid.to_string(), RegisteredVoice { embedding, icl });
        Ok(())
    }

    /// Release a registered cloned voice, reporting whether it was held.
    pub fn forget_voice(&mut self, uuid: &str) -> bool {
        self.voices.remove(uuid).is_some()
    }

    /// Synthesize, calling `on_audio` with each decoded chunk of samples.
    ///
    /// `on_audio` returns `false` to stop early — a cancel, or the daemon
    /// hanging up — which ends the synthesis as [`Outcome::Stopped`] rather
    /// than as a failure. `should_continue` is checked once per generated
    /// frame, so a cancel is noticed within about 80 ms of speech rather than
    /// at the next chunk boundary.
    ///
    /// # Errors
    /// Returns an error if generation fails.
    pub fn synthesize(
        &mut self,
        prepared: &Prepared,
        should_continue: impl FnMut() -> bool,
        on_audio: impl FnMut(&[f32]) -> bool,
    ) -> Result<Outcome> {
        self.generate(prepared, MAX_NEW_TOKENS, should_continue, on_audio)
    }

    /// Compile and tune the kernels, so the first request does not.
    ///
    /// CubeCL compiles every GPU kernel the first time it meets one and tunes
    /// each operation against its candidates. On a cold cache that is minutes,
    /// not seconds — measured at over six on an RTX 3090 — and it would
    /// otherwise all land on whoever sends the first request, after
    /// `GET /v1/status` has already said `ready`. Two seconds of speech are
    /// generated and thrown away here instead, which walks the talker, the code
    /// predictor, the sampler and the codec decoder, so the work happens while
    /// the daemon is still showing its loading indicator.
    ///
    /// The text is a few hundred characters rather than a word, and that
    /// matters more than it looks. CubeCL keys its autotune entries on the
    /// shape of the problem, and the prefix pass — one forward over the whole
    /// prompt, before any frame is generated — has a shape that comes from the
    /// prompt's length. Warming up on `"Hello."` tunes a shape no real request
    /// has, and the first real one then stalls before its first sound: 35
    /// seconds, measured, against the 0.4 it takes once the shape is known.
    ///
    /// Everything it costs is paid once per build and device: the second run
    /// reads the cache the daemon granted and this returns in about a second.
    ///
    /// A failure here is logged and swallowed. The model is loaded and usable;
    /// refusing the load over a warm-up would turn a slow first request into no
    /// service at all.
    fn warm_up(&mut self) {
        // A Base checkpoint has no default voice to warm up with — it speaks
        // only in a voice that was registered — so one is invented here, from a
        // few seconds of synthetic sound, and released when the warm-up is
        // done. That is what lets the ladder below run at all: everything it
        // warms is a generation conditioned on an embedding, and there is no
        // embedding without a voice.
        //
        // Registered *without* a transcript, deliberately. A transcript would
        // also run the codec encoder and put the clip in the prefix as an
        // in-context example, and both of those are keyed on the length of the
        // recording — five synthetic seconds tunes shapes that nobody's actual
        // recording will reuse. That cost belongs in `POST /v1/voices`, which
        // happens once per voice and where nobody is waiting to hear speech.
        let scratch = if self.clones_voices() {
            let reference =
                warm_up_reference(self.reference_sample_rate(), WARM_UP_REFERENCE_SECONDS);
            match self.register_voice(WARM_UP_VOICE, None, &reference) {
                Ok(()) => Some(WARM_UP_VOICE),
                Err(e) => {
                    log::warn!("skipping the warm-up: {e:#}");
                    return;
                }
            }
        } else {
            None
        };
        self.warm_up_shapes(scratch);
        if let Some(voice) = scratch {
            self.forget_voice(voice);
        }
    }

    /// The warm-up itself, conditioned on `cloned` when the checkpoint needs a
    /// voice to speak at all.
    fn warm_up_shapes(&mut self, cloned: Option<&str>) {
        let voice = match cloned {
            Some(uuid) => Requested::Cloned(uuid),
            None => Requested::Default,
        };
        let start = std::time::Instant::now();
        for length in WARM_UP_LENGTHS {
            let text: String = WARM_UP_TEXT.chars().take(length).collect();
            let prepared = match self.prepare(&text, voice, None, Some("en"), None) {
                Ok(p) => p,
                Err(e) => {
                    log::warn!("skipping the warm-up: {e}");
                    return;
                }
            };
            // Stopped by the callback after one chunk rather than by a low
            // `max_new_tokens`: the talker's captured graph is sized from the
            // token limit, so warming up under the real one tunes the kernels a
            // real request will use. One chunk is the shortest generation that
            // still reaches the decoder.
            let mut frames = 0_usize;
            let keep_going = || {
                frames += 1;
                frames <= STREAM_CHUNK_FRAMES
            };
            if let Err(e) = self.generate(&prepared, MAX_NEW_TOKENS, keep_going, |_| true) {
                log::warn!("the warm-up failed after {:.1?}: {e:#}", start.elapsed());
                return;
            }
        }
        let ladder = start.elapsed();
        log::info!(
            "warmed {} prompt lengths up in {ladder:.1?}",
            WARM_UP_LENGTHS.len()
        );
        if ladder < DEEP_WARM_UP_THRESHOLD {
            return;
        }
        // Cold. Generate far enough to outgrow the initial key/value cache, so
        // the doubling and the wider shapes it needs are tuned here rather than
        // inside somebody's first request.
        let deep = std::time::Instant::now();
        // The longest rung, not the shortest: a generation ends at the model's
        // end-of-speech token or at the cap below, whichever comes first, and a
        // sixty-character prompt reaches its end-of-speech in well under a
        // hundred frames — nowhere near a doubling.
        let longest = WARM_UP_LENGTHS[WARM_UP_LENGTHS.len() - 1];
        let text: String = WARM_UP_TEXT.chars().take(longest).collect();
        let Ok(prepared) = self.prepare(&text, voice, None, Some("en"), None) else {
            return;
        };
        let mut frames = 0_usize;
        let keep_going = || {
            frames += 1;
            frames <= DEEP_WARM_UP_FRAMES
        };
        let outcome = self.generate(&prepared, MAX_NEW_TOKENS, keep_going, |_| true);
        match outcome {
            // The frame count says which of the two ended it, and a count well
            // under the cap means this pass warmed nothing it was meant to.
            Ok(_) => log::info!(
                "warmed a long generation up in {:.1?}, {frames} frames",
                deep.elapsed()
            ),
            Err(e) => log::warn!("the deep warm-up failed: {e:#}"),
        }
    }

    /// The generation both [`Self::synthesize`] and [`Self::warm_up`] run, with
    /// the frame cap as the only difference between them.
    fn generate(
        &mut self,
        prepared: &Prepared,
        max_new_tokens: usize,
        mut should_continue: impl FnMut() -> bool,
        mut on_audio: impl FnMut(&[f32]) -> bool,
    ) -> Result<Outcome> {
        let generation = GenerationConfig {
            max_new_tokens,
            ..Default::default()
        };

        let frames_per_second = self.frames_per_second;
        // Split borrows: generation holds `&mut self.talker` while the decoder
        // holds `&mut self.speech_tokenizer` and the prompt borrows the voice
        // out of `self.voices`.
        let Self {
            talker,
            speech_tokenizer,
            voices: registry,
            ..
        } = self;

        let cloned = match &prepared.voice {
            Conditioning::Cloned(uuid) => registry.get(uuid),
            _ => None,
        };
        let voice = match (&prepared.voice, cloned) {
            (Conditioning::Speaker(s), _) => Voice::Speaker(s),
            (Conditioning::Cloned(_), Some(v)) => Voice::Embedding(&v.embedding),
            // `prepare` resolved this id against the same registry, so getting
            // here means the voice was deleted between the two — rare, but the
            // wrong voice is worse than a refusal.
            (Conditioning::Cloned(uuid), None) => {
                bail!("the voice {uuid} was released before this request could use it")
            }
            _ => Voice::None,
        };
        let icl = cloned
            .and_then(|v| v.icl.as_ref())
            .map(|(ref_ids, ref_codes)| IclReference { ref_ids, ref_codes });
        let prompt = Prompt {
            input_ids: &prepared.input_ids,
            instruct_ids: prepared.instruct_ids.as_deref(),
            language: Some(&prepared.language),
            voice,
            icl,
            // The whole text goes in the prefix — except when cloning, where
            // the reference implementation feeds it one token per frame and
            // this port follows it.
            non_streaming_mode: cloned.is_none(),
        };
        let mut decoder = ChunkDecoder::new(speech_tokenizer);

        let mut stopped = false;
        let frames = talker
            .generate_with_callback(&prompt, &generation, |frame| {
                if !should_continue() {
                    stopped = true;
                    return ControlFlow::Break(());
                }
                if let Some(pcm) = decoder.push(frame)
                    && !on_audio(&pcm)
                {
                    stopped = true;
                    return ControlFlow::Break(());
                }
                ControlFlow::Continue(())
            })
            .map_err(|e| anyhow!("generating speech: {e}"))?;
        log::debug!(
            "generated {} frames ({:.1}s)",
            frames.len(),
            seconds_of(frames.len(), frames_per_second)
        );
        if stopped {
            return Ok(Outcome::Stopped);
        }

        // Whatever is left over from the last full chunk.
        if let Some(pcm) = decoder.flush()
            && !on_audio(&pcm)
        {
            return Ok(Outcome::Stopped);
        }
        Ok(Outcome::Finished)
    }
}

/// Decodes codec frames to audio in fixed chunks as they are generated.
///
/// Each chunk is decoded together with the frames that precede it, whose audio
/// is then discarded, so the streamed result is sample-for-sample what a single
/// decode of the whole utterance would have produced. Every such decode goes
/// through one window of [`DECODE_WINDOW_FRAMES`], short chunks padded up to
/// it, so the decoder is compiled and tuned for a single shape.
struct ChunkDecoder<'a> {
    speech_tokenizer: &'a mut SpeechTokenizer,
    /// Every frame generated so far, flattened.
    codes: Vec<u32>,
    /// Codes per frame, learned from the first frame.
    num_code_groups: usize,
    /// Frames whose audio has already been handed to the caller.
    written: usize,
}

impl<'a> ChunkDecoder<'a> {
    fn new(speech_tokenizer: &'a mut SpeechTokenizer) -> Self {
        Self {
            speech_tokenizer,
            codes: Vec::new(),
            num_code_groups: 0,
            written: 0,
        }
    }

    /// Take one generated frame, decoding a chunk once enough have arrived.
    fn push(&mut self, frame: &[u32]) -> Option<Vec<f32>> {
        if self.num_code_groups == 0 {
            self.num_code_groups = frame.len();
        }
        self.codes.extend_from_slice(frame);
        let pending = self.codes.len() / self.num_code_groups - self.written;
        if pending >= STREAM_CHUNK_FRAMES {
            self.decode_pending()
        } else {
            None
        }
    }

    /// Decode whatever is left after the last full chunk.
    fn flush(&mut self) -> Option<Vec<f32>> {
        self.decode_pending()
    }

    fn decode_pending(&mut self) -> Option<Vec<f32>> {
        if self.num_code_groups == 0 {
            return None;
        }
        let frames = self.codes.len() / self.num_code_groups;
        if frames <= self.written {
            return None;
        }
        let (context, span) = decode_span(self.written, frames);
        debug_assert!(span <= DECODE_WINDOW_FRAMES);
        let start = (self.written - context) * self.num_code_groups;
        // `decode_window` drops the context's audio itself: it was decoded only
        // so the seam between this chunk and the last one matches a single
        // decode.
        let pcm = self.speech_tokenizer.decode_window(
            &self.codes[start..frames * self.num_code_groups],
            context,
            DECODE_WINDOW_FRAMES,
        );
        self.written = frames;
        Some(pcm)
    }
}

/// The frames one decode covers, given how many have been written and how many
/// exist: its context, and the total it sees — context included.
///
/// Free of the decoder so the invariant that total never outgrows
/// [`DECODE_WINDOW_FRAMES`] can be checked without one.
fn decode_span(written: usize, frames: usize) -> (usize, usize) {
    let context = usize::min(STREAM_CONTEXT_FRAMES, written);
    (context, context + frames - written)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The talker cannot outrun the daemon's 32 MiB cap on one response: at
    /// 24 kHz mono s16le, the longest utterance it may generate must stay
    /// under it, or a runaway generation would be aborted with nothing said.
    #[test]
    fn the_generation_cap_stays_under_the_daemons_audio_cap() {
        let seconds = seconds_of(MAX_NEW_TOKENS, NOMINAL_FRAMES_PER_SECOND);
        let bytes = seconds * 24_000.0 * 2.0;
        assert!(
            bytes < 32.0 * 1024.0 * 1024.0,
            "{seconds:.0}s is {bytes:.0} bytes, past the 32 MiB cap"
        );
    }

    /// Time-to-first-sound is a chunk of audio, so the chunk has to be short
    /// enough that speech starts promptly once generation is under way.
    #[test]
    fn a_stream_chunk_is_a_couple_of_seconds() {
        let seconds = seconds_of(STREAM_CHUNK_FRAMES, NOMINAL_FRAMES_PER_SECOND);
        assert!((0.5..=3.0).contains(&seconds), "{seconds}s per chunk");
    }

    /// Every decode is padded up to one window, so no decode may outgrow it —
    /// `decode_window` asserts, and it would do so mid-utterance. The chunk and
    /// context sizes are what has to keep fitting, whatever they are changed to.
    #[test]
    fn no_decode_of_a_stream_outgrows_the_window() {
        for total in 1..=4 * DECODE_WINDOW_FRAMES {
            let mut written = 0;
            for generated in 1..=total {
                if generated - written < STREAM_CHUNK_FRAMES {
                    continue;
                }
                let (_, span) = decode_span(written, generated);
                assert!(span <= DECODE_WINDOW_FRAMES, "chunk of {span} frames");
                written = generated;
            }
            if total > written {
                let (_, span) = decode_span(written, total);
                assert!(span <= DECODE_WINDOW_FRAMES, "flush of {span} frames");
            }
        }
    }

    #[test]
    fn checkpoint_families_are_recognized_by_type() {
        assert_eq!(Kind::parse("custom_voice").unwrap(), Kind::CustomVoice);
        assert_eq!(Kind::parse("voice_design").unwrap(), Kind::VoiceDesign);
        assert_eq!(Kind::parse("base").unwrap(), Kind::Base);
    }

    /// A family this backend does not serve is refused rather than guessed at.
    #[test]
    fn an_unknown_checkpoint_family_is_refused() {
        assert!(Kind::parse("something_else").is_err());
    }

    /// The accelerator reported to the daemon is the one this build was
    /// compiled for. A build whose `accel` and backend disagree would be
    /// installed for the wrong hosts.
    /// The warm-up takes prefixes of one string, so that string has to be at
    /// least as long as the longest rung of the ladder — otherwise the top
    /// rungs all warm the same shape and the lengths they were meant to cover
    /// stall on their first real request, silently.
    #[test]
    fn the_warm_up_text_reaches_every_length_it_is_cut_to() {
        let longest = WARM_UP_LENGTHS.iter().copied().max().expect("a ladder");
        assert!(
            WARM_UP_TEXT.chars().count() >= longest,
            "warm-up text is {} chars, short of the {longest} it is cut to",
            WARM_UP_TEXT.chars().count()
        );
    }

    #[test]
    fn the_reported_accelerator_matches_the_compiled_backend() {
        let (_, name) = select_device(None);
        // `flex` and `cpu` are two CPU backends and both report "cpu".
        assert_eq!(name, BUILT_FOR);
        assert_eq!(ON_GPU, BUILT_FOR != "cpu");
    }
}
