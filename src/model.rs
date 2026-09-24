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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result, anyhow, bail};
use burn::prelude::{Device, Tensor};
use burn::tensor::DType;
use tokenizers::Tokenizer;

use crate::lang;
use crate::progress::{Finish, Measure, Phase, Report, Step, Tracker};
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

/// Frames of already-decoded history each chunk is decoded behind, at most.
///
/// The decoder states how much history makes a chunk come out exactly as a
/// decode of the whole utterance would —
/// [`SpeechTokenizer::decode_context_frames`], 580 frames for the 12 Hz codec,
/// since its eight windowed attention layers compound — and carrying all of it
/// would decode 24 frames for every one heard. Most of it buys nothing
/// audible. Measured on the 12 Hz codec against a whole decode of the same
/// codes, the stream matches it to 37 dB behind 73 frames (one window, what
/// this replaced) and to 65 dB behind 150, and 60 dB is where the decoder's
/// own arithmetic noise sits: its f32 matmuls run on tensor cores at TF32,
/// and a whole decode is 60 dB from the fp32 reference for that alone. So 150
/// puts the seams under the noise, at 1.8x the decode work of 73; every frame
/// past it is paid for and not heard.
const STREAM_CONTEXT_FRAMES: usize = 150;

/// A seed for one request's sampling, drawn from the process's entropy.
///
/// The vendored [`GenerationConfig`] defaults to one fixed seed, which is what
/// a demo wants: the same text plays the same way twice. A service must not.
/// Sampling decides how an utterance comes out — a draw can fade, clip or
/// rush — and with one seed the same sentence gets the same draw every time it
/// is spoken, with no way out but changing the words; the reference
/// implementation draws fresh noise on every call. The seed goes in the log
/// with the utterance, so a draw worth studying can still be repeated.
fn fresh_seed() -> u64 {
    use std::hash::{BuildHasher, RandomState};
    RandomState::new().hash_one(0u64)
}

/// The `[[options]]` entry a request's sampling temperature arrives under.
///
/// The name is half a contract: `server.rs` reads the header the daemon spells
/// it as, and the tests either side are what hold the two together.
pub const TEMPERATURE_OPTION: &str = "temperature";

/// The range [`TEMPERATURE_OPTION`] accepts, inclusive.
///
/// The manifest declares this as the `min` and `max` of a `float` option, with
/// a `step`, which is what makes it a slider the daemon bounds — so a mistyped
/// `0.09` never reaches a request as though the backend had agreed to it. It
/// is repeated here because a header is a wire boundary and the settings sheet
/// is not the only thing that could set one; the test below is what holds the
/// two spellings together.
///
/// The ends are where they are for a reason: under about 0.6 the talker starts
/// repeating a frame until it hits the generation cap, and over about 1.2 it
/// wanders off the text.
///
/// The manifest also declares 0.9 as the option's `default`, because a slider
/// always rests somewhere and that position has to mean something: an option
/// with no default would show the user 0.6 while the model spoke at 0.9. It
/// restates what [`GenerationConfig`] ships, which is a second place for one
/// number to live — the test below is what keeps them the same number.
pub const TEMPERATURE_RANGE: (f32, f32) = (0.6, 1.2);

/// The temperature an option value asks for, or `None` to leave the
/// checkpoint's own.
///
/// The daemon holds the slider and refuses to store anything outside
/// [`TEMPERATURE_RANGE`], so in practice this sees a number in that range or
/// nothing at all. It checks anyway: a header is a wire boundary, a temperature
/// of zero is a division by zero in the sampler, and a large one is a request
/// that never terminates.
#[must_use]
pub fn temperature_option(value: Option<&str>) -> Option<f32> {
    let value = value.map(str::trim).filter(|v| !v.is_empty())?;
    let (low, high) = TEMPERATURE_RANGE;
    match value.parse::<f32>() {
        Ok(t) if (low..=high).contains(&t) => Some(t),
        _ => {
            log::warn!(
                "ignoring the {TEMPERATURE_OPTION} option {value:?}: it is not a number in \
                 {low}..={high}"
            );
            None
        }
    }
}

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
    /// The sampling temperature this request asked for, or `None` to keep the
    /// checkpoint's. Resolved with the rest of the request rather than read on
    /// the model thread, where the headers it came from are already gone.
    temperature: Option<f32>,
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
    /// What *is* persisted is the expensive half of each registration — see
    /// [`crate::voice_cache`] — so re-registering is a file read.
    voices: HashMap<String, RegisteredVoice>,
    /// The checkpoint's name, which every derived voice artifact is keyed on:
    /// the embedding comes from this checkpoint's speaker encoder and means
    /// nothing to another.
    model_name: String,
    /// Where to rebuild a cached embedding, and in which dtype. Held rather
    /// than recomputed so the rule that picked them at load cannot drift from
    /// the one that reads them back.
    device: Device,
    dtype: DType,
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
            .field("model_name", &self.model_name)
            .field("device", &self.device)
            .field("dtype", &self.dtype)
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

/// Whether this build compiles its kernels at runtime, through `CubeCL`,
/// which is what makes a first load an initial setup. The `flex` CPU backend
/// does not.
const BUILDS_KERNELS: bool = cfg!(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal",
    feature = "wgpu",
    feature = "cpu"
));

/// The entries a first load's warm-up writes to the kernel cache, tuning
/// results and compiled kernels both: 1754 on CUDA and 726 on Vulkan, measured
/// on an empty cache with the 1.7B CustomVoice checkpoint on an RTX 3090. ROCm
/// and `CubeCL`'s CPU backend are taken to be CUDA, Metal and the generic
/// `wgpu` to be Vulkan, unmeasured. Only the pace of the bar rides on it: past
/// the estimate it slows down short of the end rather than stopping, see
/// `progress::estimate`, and a checkpoint that writes fewer ends its step
/// early.
const WARM_UP_CACHE_ENTRIES: u64 = if !BUILDS_KERNELS {
    0
} else if cfg!(any(feature = "vulkan", feature = "metal", feature = "wgpu")) {
    726
} else {
    1754
};

/// The units of work a warm-up in `phase` is expected to do, which is what
/// its progress is measured against: the entries it writes to the kernel
/// cache, and one per frame it generates — every rung of the ladder, and on
/// an initial setup the deep pass after it, both counted as far as their
/// caps. Frames are what keep a warm load's bar, and a build without a kernel
/// cache, moving.
fn warm_up_work(phase: Phase) -> u64 {
    let ladder = WARM_UP_LENGTHS.len() * (STREAM_CHUNK_FRAMES + 1);
    match phase {
        Phase::InitialSetup => WARM_UP_CACHE_ENTRIES + (ladder + DEEP_WARM_UP_FRAMES + 1) as u64,
        Phase::Loading => ladder as u64,
    }
}

/// The type the talker computes in on `device`.
///
/// A half-width type halves the talker's weights and its bandwidth on a GPU.
/// On the CPU it is slower than f32 rather than faster.
///
/// bf16 is what the checkpoints were trained in and what CUDA and ROCm get.
/// Never on Vulkan, whatever the device reports: the SPIR-V extension for bf16
/// allows no arithmetic on it, yet CubeCL emits some, and NVIDIA's driver
/// crashes compiling it. Vulkan, and the generic `wgpu` backend, which is
/// Vulkan on Linux, get f16 instead. It holds the talker: the largest value
/// of a 1.7B prefill, the product inside the MLP of its third layer, is
/// 9.6e3 of the 6.5e4 f16 reaches, and the speech ends where it should. A
/// device that computes in neither type gets f32, which is Metal's case:
/// CubeCL's Metal backend reports no bf16.
fn talker_dtype(device: &Device) -> DType {
    let half = if matches!(BUILT_FOR, "vulkan" | "wgpu") {
        DType::F16
    } else {
        DType::BF16
    };
    if ON_GPU && device.supports_dtype(half) {
        half
    } else {
        DType::F32
    }
}

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
        let _ = CACHE_DIR.set(dir.to_path_buf());
    }
    // `false` means something already read the configuration and this call is
    // too late to matter. Nothing in this backend touches a device before
    // `main` calls this, so it is a guard rather than a case to handle.
    if !CubeClRuntimeConfig::try_set(config) {
        log::warn!("the CubeCL configuration was already read; the kernel cache keeps its default");
    }
}

/// Where `CubeCL` keeps this backend's kernels, when the daemon grants a place.
static CACHE_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Where a load records that `model` finished a warm-up with this build, so
/// that the next load of it is not an initial setup. It sits beside the
/// kernels it vouches for: clearing the cache clears it too, and a new build
/// looks for one of its own. Without a cache directory there is none, and
/// every load is an initial setup, since every load compiles everything.
fn warm_marker(model: &str) -> Option<PathBuf> {
    CACHE_DIR.get().map(|dir| {
        dir.join("qwen-tts-warm")
            .join(format!("{model}-{}-{BUILT_FOR}", env!("CARGO_PKG_VERSION")))
    })
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
    /// Load a checkpoint from a backend directory, then warm it up, reporting
    /// how far it has got into `report` as it goes.
    ///
    /// # Errors
    /// Returns an error if a declared file is missing, the checkpoint is of a
    /// family this backend does not serve, or the weights cannot be mapped.
    pub fn load(
        backend_dir: &Path,
        model_name: &str,
        device: Option<&str>,
        report: &(dyn Fn(Report) + Sync),
    ) -> Result<Self> {
        // A build without kernels to compile has no initial setup to tell
        // apart: every one of its loads is the same.
        let marker = BUILDS_KERNELS.then(|| warm_marker(model_name)).flatten();
        let phase = if !BUILDS_KERNELS || marker.as_ref().is_some_and(|m| m.exists()) {
            Phase::Loading
        } else {
            Phase::InitialSetup
        };
        let tracker = Tracker::new(phase, report);
        let (model, warmed) = std::thread::scope(|scope| {
            scope.spawn(|| tracker.run());
            // Stops the sampler however this ends, a panic included: the
            // scope waits for it before it lets a panic through.
            let _finish = Finish(&tracker);
            let mut model = Self::load_weights(backend_dir, model_name, device, phase, &tracker)?;
            let step = match phase {
                Phase::InitialSetup => Step::BuildingKernels,
                Phase::Loading => Step::WarmingUp,
            };
            let frames = Arc::new(AtomicU64::new(0));
            let entries = crate::progress::cache_entries();
            tracker.enter(
                step,
                Measure::warm_up(Arc::clone(&frames), warm_up_work(phase)),
            );
            let warmed = model.warm_up(&frames);
            // What `WARM_UP_CACHE_ENTRIES` and `warm_up_work` estimate, for
            // checking them against a card they were not measured on.
            log::info!(
                "the warm-up wrote {} kernel-cache entries and generated {} frames, \
                 against {} units expected",
                crate::progress::cache_entries().saturating_sub(entries),
                frames.load(Ordering::Relaxed),
                warm_up_work(phase)
            );
            anyhow::Ok((model, warmed))
        })?;
        if warmed && let Some(marker) = marker {
            let written = marker
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|()| std::fs::write(&marker, b""));
            if let Err(e) = written {
                log::warn!("could not mark {} warm: {e}", marker.display());
            }
        }
        Ok(model)
    }

    /// The part of [`Self::load`] before the warm-up, under its tracker.
    fn load_weights(
        backend_dir: &Path,
        model_name: &str,
        device: Option<&str>,
        phase: Phase,
        tracker: &Tracker<'_>,
    ) -> Result<Self> {
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
        let dtype = talker_dtype(&device);
        log::info!(
            "loading {model_name} on {device_name} ({dtype:?}) from {} ({phase:?})",
            dir.display()
        );
        let total = [&weights_file, &st_weights_file]
            .into_iter()
            .try_fold(0, |sum, file| {
                std::fs::metadata(file).map(|m| sum + m.len())
            })?;
        let read = Arc::new(AtomicU64::new(0));
        tracker.enter(
            Step::LoadingWeights,
            Measure::Bytes {
                read: Arc::clone(&read),
                total,
            },
        );

        let tokenizer = Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow!("loading {}: {e}", tokenizer_file.display()))?;

        let mut talker = Talker::load(&config, &weights_file, dtype, &device, &read)
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
        // Its convolutions are the exception on Vulkan, see below.
        //
        // The Base checkpoints also need the codec's *encoder*, to turn a
        // cloning reference into the codes an in-context example is made of.
        // Only they: it is weights and load time no other family would use.
        let mut speech_tokenizer = if kind == Kind::Base {
            SpeechTokenizer::load_with_encoder(&st_config, &st_weights_file, &device, &read)
        } else {
            SpeechTokenizer::load(&st_config, &st_weights_file, &device, &read)
        }
        .map_err(|e| anyhow!("building the codec from {}: {e}", st_weights_file.display()))?;
        // Where the talker computes in f16, which is Vulkan, so do the decoder's
        // convolutions, or the codec alone runs slower than real time. See
        // `SpeechTokenizer::convolve_in` for what that costs.
        if dtype == DType::F16 {
            speech_tokenizer.convolve_in(DType::F16);
        }
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
        let model = Self {
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
            model_name: model_name.to_string(),
            device,
            dtype,
        };
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
    /// ask for, already resolved by [`voices::configured`]. It is the wording
    /// behind one voice id — [`voices::CUSTOM_VOICE_ID`] — and reaches nothing
    /// else: a request naming another design gets that design, and one naming
    /// no voice at all gets the default. It used to fill in for any request
    /// that named none, which was right while the option was the only way to
    /// choose and became a lie the moment the designs were voices: the daemon
    /// sends no voice when the user has stored no preference, so a typed
    /// description would have overridden the "· default" the picker was
    /// showing them.
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
        temperature: Option<f32>,
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
            temperature,
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
    /// would refuse every request from a user who set the option once. Within
    /// that checkpoint it reaches exactly one id, [`voices::CUSTOM_VOICE_ID`].
    fn resolve_voice(
        &self,
        voice: Requested<'_>,
        configured: Option<&str>,
    ) -> Result<Conditioning, RequestError> {
        conditioning_for(
            self.kind,
            &self.speakers,
            self.default_speaker.as_deref(),
            &|uuid| self.voices.contains_key(uuid),
            voice,
            configured,
        )
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
        let transcript = transcript.map(str::trim).filter(|t| !t.is_empty());
        let seconds = seconds_of(samples.len(), f64::from(self.reference_sample_rate()));

        // The cached half first. Deriving is expensive in a way that has
        // nothing to do with the size of what comes out: the codec encoder
        // meets the whole clip as one tensor, so every clip length is a shape
        // CubeCL has to autotune, and tuning is the slow part of meeting a new
        // shape. What it produces is a few tens of kilobytes.
        let cache = crate::voice_cache::dir().map(|dir| {
            (
                dir,
                crate::voice_cache::key(&self.model_name, transcript, samples),
            )
        });
        if let Some((dir, key)) = cache.as_ref()
            && let Some(derived) = crate::voice_cache::load(dir, key)
        {
            let dims = [derived.embedding.len()];
            // f32 on disk whatever the talker runs in. A bf16 value is exactly
            // representable in f32, so widening to write and narrowing to read
            // gives back the bits that were there.
            let embedding = Tensor::<1>::from_data(
                burn::tensor::TensorData::new(derived.embedding, dims),
                &self.device,
            )
            .cast(self.dtype);
            log::info!("registered the voice {uuid} from {seconds:.1}s of audio, already encoded");
            self.voices.insert(
                uuid.to_string(),
                RegisteredVoice {
                    embedding,
                    icl: derived.icl,
                },
            );
            return Ok(());
        }

        let started = std::time::Instant::now();
        let embedding = self
            .talker
            .speaker_embedding(samples)
            .map_err(|e| anyhow!("deriving the speaker embedding: {e}"))?;
        let embedded = started.elapsed();
        let icl = match transcript {
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
        // Both halves timed separately: the first registration of a clip length
        // can run for the better part of a minute, and which half it was spent
        // in is the difference between tuning the speaker encoder and tuning
        // the codec.
        log::info!(
            "registered the voice {uuid} from {seconds:.1}s of audio{} in {:.1?} \
             (embedding {embedded:.1?})",
            if icl.is_some() {
                ", with its transcript"
            } else {
                ""
            },
            started.elapsed(),
        );

        if let Some((dir, key)) = cache.as_ref() {
            // Cloned to write: the tensor goes on to serve the voice, and the
            // widening cast is what makes the stored copy dtype-independent.
            let values = embedding
                .clone()
                .cast(DType::F32)
                .into_data()
                .try_to_vec::<f32>();
            match values {
                Ok(embedding) => crate::voice_cache::store(
                    dir,
                    key,
                    &crate::voice_cache::Derived {
                        embedding,
                        icl: icl.clone(),
                    },
                ),
                Err(e) => log::warn!("not caching the voice {uuid}: reading the embedding: {e:?}"),
            }
        }

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
    ///
    /// Counts the frames it generates into `frames`, which with the kernels it
    /// compiles and tunes is how its progress is measured, and returns whether
    /// it ran to the end.
    fn warm_up(&mut self, frames: &AtomicU64) -> bool {
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
                    return false;
                }
            }
        } else {
            None
        };
        let warmed = self.warm_up_shapes(scratch, frames);
        if let Some(voice) = scratch {
            self.forget_voice(voice);
        }
        warmed
    }

    /// The warm-up itself, conditioned on `cloned` when the checkpoint needs a
    /// voice to speak at all. Counts its frames into `frames` and returns
    /// whether it ran to the end.
    fn warm_up_shapes(&mut self, cloned: Option<&str>, frames: &AtomicU64) -> bool {
        let voice = match cloned {
            Some(uuid) => Requested::Cloned(uuid),
            None => Requested::Default,
        };
        let start = std::time::Instant::now();
        for length in WARM_UP_LENGTHS {
            let text: String = WARM_UP_TEXT.chars().take(length).collect();
            let prepared = match self.prepare(&text, voice, None, Some("en"), None, None) {
                Ok(p) => p,
                Err(e) => {
                    log::warn!("skipping the warm-up: {e}");
                    return false;
                }
            };
            // Stopped by the callback after one chunk rather than by a low
            // `max_new_tokens`: the talker's captured graph is sized from the
            // token limit, so warming up under the real one tunes the kernels a
            // real request will use. One chunk is the shortest generation that
            // still reaches the decoder.
            let mut generated = 0_usize;
            let keep_going = || {
                generated += 1;
                frames.fetch_add(1, Ordering::Relaxed);
                generated <= STREAM_CHUNK_FRAMES
            };
            if let Err(e) = self.generate(&prepared, MAX_NEW_TOKENS, keep_going, |_| true) {
                log::warn!("the warm-up failed after {:.1?}: {e:#}", start.elapsed());
                return false;
            }
        }
        let ladder = start.elapsed();
        log::info!(
            "warmed {} prompt lengths up in {ladder:.1?}",
            WARM_UP_LENGTHS.len()
        );
        if ladder < DEEP_WARM_UP_THRESHOLD {
            return true;
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
        let Ok(prepared) = self.prepare(&text, voice, None, Some("en"), None, None) else {
            return false;
        };
        let mut generated = 0_usize;
        let keep_going = || {
            generated += 1;
            frames.fetch_add(1, Ordering::Relaxed);
            generated <= DEEP_WARM_UP_FRAMES
        };
        let outcome = self.generate(&prepared, MAX_NEW_TOKENS, keep_going, |_| true);
        match outcome {
            // The frame count says which of the two ended it, and a count well
            // under the cap means this pass warmed nothing it was meant to.
            Ok(_) => {
                log::info!(
                    "warmed a long generation up in {:.1?}, {generated} frames",
                    deep.elapsed()
                );
                true
            }
            Err(e) => {
                log::warn!("the deep warm-up failed: {e:#}");
                false
            }
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
        let mut generation = GenerationConfig {
            max_new_tokens,
            seed: fresh_seed(),
            ..Default::default()
        };
        // Only the talker's own sampling. The code predictor draws the codec's
        // residual detail rather than the shape of the utterance, and the
        // reference implementation gives it its own `subtalker_temperature`;
        // moving both from one field would be a second, unasked-for change to
        // how the audio sounds.
        if let Some(temperature) = prepared.temperature {
            generation.sampling = generation.sampling.with_temperature(temperature);
        }

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
        log::info!(
            "generated {} frames ({:.1}s) from seed {:#x}",
            frames.len(),
            seconds_of(frames.len(), frames_per_second),
            generation.seed
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
/// is then discarded, so the seam between it and the last chunk is what a
/// single decode of both would have produced. The decoder says how many frames
/// make that exact — [`SpeechTokenizer::decode_context_frames`] — and it is
/// more than a stream can afford to re-decode with every chunk, so the stream
/// carries [`STREAM_CONTEXT_FRAMES`] of it, a figure set by measuring the seams
/// against a whole decode rather than by ear: short of it, the loss sits at
/// every seam past the first and nowhere else. Every decode goes through one
/// window of [`Self::window_frames`], short chunks padded up to it, so the
/// decoder is compiled and tuned for a single shape.
struct ChunkDecoder<'a> {
    speech_tokenizer: &'a mut SpeechTokenizer,
    /// Every frame generated so far, flattened.
    codes: Vec<u32>,
    /// Codes per frame, learned from the first frame.
    num_code_groups: usize,
    /// Frames whose audio has already been handed to the caller.
    written: usize,
    /// Frames of context each decode carries, see [`stream_context_frames`].
    context_frames: usize,
    /// The one shape every decode reaches the decoder as: a full context plus a
    /// full chunk.
    ///
    /// A GPU backend compiles and tunes its kernels per shape, so a decoder fed
    /// the lengths a stream naturally produces — a chunk, then two, then three,
    /// then whatever is left over at the end — pays that cost several times an
    /// utterance on a cold cache. Padding every chunk up to one window pays it
    /// once. The decoder is causal and
    /// [`SpeechTokenizer::decode_window`] truncates the padding's samples, so
    /// the audio is unchanged.
    window_frames: usize,
}

/// The context a stream decodes each chunk behind: what the decoder asks for,
/// up to [`STREAM_CONTEXT_FRAMES`]. A decoder with no finite figure — attention
/// over the whole sequence — gets the cap too: what it loses is the same
/// distant history, and a decode from the first frame of every chunk would
/// grow with the utterance.
fn stream_context_frames(decoder: Option<usize>) -> usize {
    decoder.map_or(STREAM_CONTEXT_FRAMES, |exact| {
        exact.min(STREAM_CONTEXT_FRAMES)
    })
}

impl<'a> ChunkDecoder<'a> {
    fn new(speech_tokenizer: &'a mut SpeechTokenizer) -> Self {
        let context_frames = stream_context_frames(speech_tokenizer.decode_context_frames());
        Self {
            speech_tokenizer,
            codes: Vec::new(),
            num_code_groups: 0,
            written: 0,
            context_frames,
            window_frames: context_frames + STREAM_CHUNK_FRAMES,
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
        let (context, span) = decode_span(self.written, frames, self.context_frames);
        debug_assert!(span <= self.window_frames);
        let start = (self.written - context) * self.num_code_groups;
        let codes = &self.codes[start..frames * self.num_code_groups];
        // `decode_window` drops the context's audio itself: it was decoded only
        // so the seam between this chunk and the last one matches a single
        // decode.
        let pcm = self
            .speech_tokenizer
            .decode_window(codes, context, self.window_frames);
        self.written = frames;
        Some(pcm)
    }
}

/// The frames one decode covers, given how many have been written, how many
/// exist and how many of context the decoder asks for: its context, and the
/// total it sees — context included.
///
/// Free of the decoder so the invariant that the total never outgrows a window
/// can be checked without one.
fn decode_span(written: usize, frames: usize, context_frames: usize) -> (usize, usize) {
    let context = usize::min(context_frames, written);
    (context, context + frames - written)
}

/// The voice-resolution matrix: every `(checkpoint family, requested voice)`
/// pair and what the model is conditioned on for it.
///
/// Free of [`QwenTts`] so the whole table is testable without a loaded
/// checkpoint — it is the part of a request most likely to be wrong and the
/// least likely to be caught by listening, since a mis-resolved voice still
/// produces confident speech. `registered` answers whether a cloned id was
/// pushed to this load; taking a predicate rather than the map keeps a test
/// from having to build a `Tensor` to name a voice.
fn conditioning_for(
    kind: Kind,
    speakers: &[String],
    default_speaker: Option<&str>,
    registered: &dyn Fn(&str) -> bool,
    voice: Requested<'_>,
    configured: Option<&str>,
) -> Result<Conditioning, RequestError> {
    match (kind, voice) {
        (Kind::CustomVoice, Requested::Speaker(s)) => {
            if speakers.iter().any(|k| k == s) {
                Ok(Conditioning::Speaker(s.to_string()))
            } else {
                Err(RequestError::UnknownVoice(format!(
                    "unknown voice {s}; this model has {}",
                    speakers.join(", ")
                )))
            }
        }
        (Kind::CustomVoice, Requested::Default) => Ok(default_speaker
            .map(str::to_string)
            .map_or(Conditioning::None, Conditioning::Speaker)),
        (Kind::CustomVoice, Requested::Description(_)) => Err(RequestError::UnknownVoice(
            "this model speaks with one of its own voices, not a described one".to_string(),
        )),
        (Kind::VoiceDesign, Requested::Description(d)) => {
            Ok(Conditioning::Description(d.to_string()))
        }
        // Not the configured description: a request naming no voice is one
        // whose user stored no preference, and the picker is showing them
        // the manifest's `default_voice` — which is this. Reading the
        // option here would speak in a voice the card says is not selected.
        (Kind::VoiceDesign, Requested::Default) => Ok(Conditioning::Description(
            voices::DEFAULT_DESCRIPTION.to_string(),
        )),
        // The one id whose description is not fixed: it stands for whatever
        // the backend's description option says, and for the default when
        // that is empty — so picking Custom and writing nothing is the
        // voice a user who picked nothing gets, not a prompt with no
        // instruction in it.
        (Kind::VoiceDesign, Requested::Speaker(s)) if s == voices::CUSTOM_VOICE_ID => {
            Ok(Conditioning::Description(
                configured
                    .unwrap_or(voices::DEFAULT_DESCRIPTION)
                    .to_string(),
            ))
        }
        // The twelve pre-made designs, which are ordinary declared voices:
        // the daemon has already refused anything this model does not list,
        // so an id that misses here is one the manifest and the table
        // disagree about — a build-time bug, and the test in `voices` is
        // what keeps it one.
        (Kind::VoiceDesign, Requested::Speaker(s)) => voices::design(s)
            .map(|description| Conditioning::Description(description.to_string()))
            .ok_or_else(|| {
                RequestError::UnknownVoice(format!(
                    "unknown voice {s}; this model builds a voice from a description, \
                         so its voice ids are the designs it declares or desc:<description>"
                ))
            }),
        // Registration is a separate request, and the daemon makes it
        // before the first synthesis naming the voice. An id that is not
        // here is one whose registration failed or never happened, which is
        // worth saying plainly rather than synthesizing in some other voice.
        (Kind::Base, Requested::Cloned(uuid)) => {
            if registered(uuid) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3::sampling::Sampling;

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
    /// `decode_window` asserts, and it would do so mid-utterance. The context a
    /// decoder asks for is whatever it asks for, so the invariant is checked
    /// across a range of them rather than for the one a checkpoint happens to
    /// want.
    #[test]
    fn no_decode_of_a_stream_outgrows_the_window() {
        for context_frames in [0, 1, STREAM_CHUNK_FRAMES, 73, STREAM_CONTEXT_FRAMES, 580] {
            let window = context_frames + STREAM_CHUNK_FRAMES;
            for total in 1..=4 * window {
                let mut written = 0;
                for generated in 1..=total {
                    if generated - written < STREAM_CHUNK_FRAMES {
                        continue;
                    }
                    let (_, span) = decode_span(written, generated, context_frames);
                    assert!(span <= window, "chunk of {span} frames past {window}");
                    written = generated;
                }
                if total > written {
                    let (_, span) = decode_span(written, total, context_frames);
                    assert!(span <= window, "flush of {span} frames past {window}");
                }
            }
        }
    }

    /// A decode that asks for no context at all still has to come out
    /// well-formed, since that is the first chunk of every utterance; once
    /// there is history, all of it is carried until the context is full, and
    /// exactly that much afterwards.
    #[test]
    fn a_decode_carries_the_context_it_is_given() {
        assert_eq!(
            decode_span(0, STREAM_CHUNK_FRAMES, 150),
            (0, STREAM_CHUNK_FRAMES)
        );
        assert_eq!(decode_span(25, 50, 150), (25, 50));
        assert_eq!(decode_span(150, 175, 150), (150, 175));
        assert_eq!(decode_span(175, 200, 150), (150, 175));
        assert_eq!(decode_span(400, 425, 150), (150, 175));
    }

    /// Two figures this replaced, each wrong the same way: a hand-picked 50,
    /// then the decoder's own 73, which counted one attention window where its
    /// eight compound. Both left every chunk past the first without most of
    /// its history, a loss that starts at the first seam and never recovers.
    /// The decoder now states the exact figure and the stream carries what it
    /// asks for, up to the measured budget.
    #[test]
    fn the_stream_carries_the_decoders_context_up_to_the_budget() {
        assert_eq!(stream_context_frames(Some(73)), 73);
        assert_eq!(stream_context_frames(Some(150)), 150);
        assert_eq!(stream_context_frames(Some(580)), 150);
        assert_eq!(stream_context_frames(None), 150);
    }

    /// One text asked for twice is two draws, as it is from the reference; the
    /// vendored default would make it one.
    #[test]
    fn every_request_draws_its_own_seed() {
        let seeds: std::collections::HashSet<u64> = (0..16).map(|_| fresh_seed()).collect();
        assert_eq!(seeds.len(), 16);
        assert!(!seeds.contains(&GenerationConfig::default().seed));
    }

    /// The check that matters, on real weights: every chunk the stream hands
    /// out, one frame pushed at a time through [`ChunkDecoder`], against one
    /// decode of the whole utterance. Needs a checkpoint, a GPU and a file of
    /// codes, so it runs by hand:
    ///
    /// ```sh
    /// QWEN_TTS_CODEC=$HOME/.local/share/super-tts/backends/app.super-tts.qwen-tts/models/qwen3-tts-1.7b-base/speech_tokenizer \
    /// QWEN_TTS_CODES=/path/to/codes.bin \
    /// cargo test --release --no-default-features --features cuda -- --ignored --nocapture a_streamed_decode
    /// ```
    ///
    /// `codes.bin` is what the talker generates, frame-major little-endian
    /// `u32`, `num_code_groups` to a frame. Measured 2026-09-21 on 239 frames
    /// of the 1.7B base checkpoint: 65 dB behind [`STREAM_CONTEXT_FRAMES`],
    /// 37 dB behind the 73 it replaced.
    #[test]
    #[ignore = "needs a checkpoint, a GPU and a file of codes; see the doc comment"]
    fn a_streamed_decode_matches_a_whole_decode() {
        let (Ok(codec_dir), Ok(codes_file)) = (
            std::env::var("QWEN_TTS_CODEC"),
            std::env::var("QWEN_TTS_CODES"),
        ) else {
            eprintln!("QWEN_TTS_CODEC and QWEN_TTS_CODES are not set; nothing to check");
            return;
        };
        let codec_dir = std::path::PathBuf::from(codec_dir);
        let cfg: SpeechTokenizerConfig =
            serde_json::from_slice(&std::fs::read(codec_dir.join("config.json")).unwrap()).unwrap();
        let device = crate::qwen3::test_device();
        let mut speech_tokenizer = SpeechTokenizer::load(
            &cfg,
            &codec_dir.join("model.safetensors"),
            &device,
            &Arc::default(),
        )
        .unwrap();
        let codes: Vec<u32> = std::fs::read(codes_file)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&bytes| u32::from_le_bytes(bytes))
            .collect();
        let groups = speech_tokenizer.num_code_groups();
        let frames = codes.len() / groups;

        let whole = speech_tokenizer.decode_chunk(&codes);
        let mut decoder = ChunkDecoder::new(&mut speech_tokenizer);
        eprintln!(
            "{frames} frames, {} of context a chunk, window of {}",
            decoder.context_frames, decoder.window_frames
        );
        let mut streamed = Vec::with_capacity(whole.len());
        let mut timings = Vec::new();
        for frame in codes.chunks_exact(groups) {
            let started = std::time::Instant::now();
            if let Some(pcm) = decoder.push(frame) {
                timings.push(started.elapsed());
                streamed.extend(pcm);
            }
        }
        if let Some(pcm) = decoder.flush() {
            streamed.extend(pcm);
        }
        assert_eq!(streamed.len(), whole.len());

        let signal: f64 = whole.iter().map(|&x| f64::from(x).powi(2)).sum();
        let noise: f64 = whole
            .iter()
            .zip(&streamed)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum();
        let snr = 10. * (signal / noise.max(1e-300)).log10();
        // The first decode carries the shape's kernel tuning; the rest are the cost.
        let steady = &timings[1..];
        let mean =
            steady.iter().sum::<std::time::Duration>() / u32::try_from(steady.len()).unwrap();
        eprintln!(
            "streamed vs whole: {snr:.1} dB; {} chunk decodes, {:.0} ms each after the first ({:.0} ms)",
            timings.len(),
            mean.as_secs_f64() * 1e3,
            timings[0].as_secs_f64() * 1e3
        );
        assert!(
            snr >= 60.,
            "the stream is {snr:.1} dB from a whole decode; 60 is the decoder's own noise"
        );

        // What the window costs against the 98-frame one this replaced, warm.
        let mut time = |window: usize| {
            let context = window - STREAM_CHUNK_FRAMES;
            let prefix = &codes[..window * groups];
            speech_tokenizer.decode_window(prefix, context, window);
            let started = std::time::Instant::now();
            for _ in 0..5 {
                speech_tokenizer.decode_window(prefix, context, window);
            }
            started.elapsed() / 5
        };
        for window in [
            73 + STREAM_CHUNK_FRAMES,
            STREAM_CONTEXT_FRAMES + STREAM_CHUNK_FRAMES,
        ] {
            let took = time(window);
            eprintln!(
                "window of {window} frames: {:.0} ms a decode, warm",
                took.as_secs_f64() * 1e3
            );
        }
    }

    /// The range in this file and the `min`/`max` in the manifest are one
    /// range written twice: the daemon refuses to store anything outside the
    /// manifest's, so a value widened here alone would never arrive, and one
    /// widened there alone would be refused by [`temperature_option`] on the
    /// way in.
    #[test]
    fn the_temperature_range_is_what_the_manifest_declares() {
        let manifest = include_str!("../backend.toml");
        let field = |key| {
            crate::manifest_probe::option_field(manifest, TEMPERATURE_OPTION, key)
                .unwrap_or_else(|| panic!("the manifest declares `{key}`"))
                .parse::<f32>()
                .unwrap_or_else(|_| panic!("`{key}` is a number"))
        };
        assert_eq!(
            (field("min"), field("max")),
            TEMPERATURE_RANGE,
            "the manifest's range and this crate's disagree"
        );
        // Both ends and a grid is what makes it a slider rather than a field,
        // and a slider is not also a dropdown.
        assert!(
            crate::manifest_probe::option_field(manifest, TEMPERATURE_OPTION, "step").is_some(),
            "the option declares no step, so it renders as a text field"
        );
        assert!(
            crate::manifest_probe::declared_choices(manifest, TEMPERATURE_OPTION).is_none(),
            "an option is a dropdown or a slider, not both"
        );
    }

    /// The ends have to be in that order, and the step has to divide something:
    /// the range is what [`temperature_option`] accepts, so a reversed one
    /// would refuse every value.
    #[test]
    fn the_temperature_range_climbs() {
        let (low, high) = TEMPERATURE_RANGE;
        assert!(low < high, "{TEMPERATURE_RANGE:?} is not a range");
        let manifest = include_str!("../backend.toml");
        let step: f32 = crate::manifest_probe::option_field(manifest, TEMPERATURE_OPTION, "step")
            .expect("the manifest declares a step")
            .parse()
            .expect("the step is a number");
        assert!(step > 0.0 && step <= high - low, "unusable step {step}");
    }

    /// The setting a request arrives without is the one the checkpoint ships,
    /// so the range has to reach it — otherwise the slider cannot express
    /// "leave it alone" and every position on it is a change.
    #[test]
    fn the_shipped_temperature_is_inside_the_range() {
        let shipped = match GenerationConfig::default().sampling {
            Sampling::TopKThenTopP { temperature, .. } | Sampling::TopP { temperature, .. } => {
                temperature
            }
            Sampling::ArgMax => panic!("the checkpoints sample; they do not take the argmax"),
        };
        assert_eq!(temperature_option(Some("0.9")), Some(shipped));
        let (low, high) = TEMPERATURE_RANGE;
        assert!((low..=high).contains(&shipped), "{shipped} is out of range");
        // And the manifest says so too: the slider rests on its `default`, so
        // that number and the one the checkpoints ship have to be the same or
        // the settings sheet shows a temperature the model is not using.
        let manifest = include_str!("../backend.toml");
        let declared: f32 =
            crate::manifest_probe::option_field(manifest, TEMPERATURE_OPTION, "default")
                .expect("a slider declares the value it rests on")
                .parse()
                .expect("the default is a number");
        // Both are meant to be the one number 0.9, so the tolerance is only
        // here to keep the lint quiet about comparing floats at all: what this
        // guards is the two drifting apart, not their last bit.
        assert!(
            (declared - shipped).abs() < f32::EPSILON,
            "the manifest's default ({declared}) and the checkpoint's temperature ({shipped}) disagree"
        );
    }

    /// An unset option leaves the checkpoint's own temperature in place. It is
    /// the difference between a setting and a default, and the reason the
    /// manifest declares no `default` of its own.
    #[test]
    fn an_unset_temperature_overrides_nothing() {
        assert_eq!(temperature_option(None), None);
        assert_eq!(temperature_option(Some("")), None);
        assert_eq!(temperature_option(Some("  ")), None);
    }

    /// Both ends inclusive, and every position the slider can stop on between
    /// them. As the daemon spells it: an option value crosses the wire as the
    /// text of an `x-tts-option-*` header, whatever its declared type.
    #[test]
    fn every_temperature_in_the_range_is_accepted() {
        let (low, high) = TEMPERATURE_RANGE;
        for spelled in [format!("{low:?}"), format!("{high:?}"), "0.85".to_string()] {
            assert!(
                temperature_option(Some(&spelled)).is_some(),
                "the manifest accepts {spelled} and this refuses it"
            );
        }
        assert_eq!(temperature_option(Some("0.6")), Some(low));
        assert_eq!(temperature_option(Some("1.2")), Some(high));
    }

    /// Out of range, not merely unusual: zero divides the logits by zero and a
    /// large one never settles on an end-of-speech token.
    #[test]
    fn a_temperature_outside_the_ladder_is_refused() {
        assert_eq!(temperature_option(Some("0")), None);
        assert_eq!(temperature_option(Some("-1")), None);
        assert_eq!(temperature_option(Some("2.5")), None);
        assert_eq!(temperature_option(Some("NaN")), None);
        assert_eq!(temperature_option(Some("hot")), None);
    }

    /// A value between two rungs is still in range, and is taken: the ladder
    /// bounds what is sensible, it does not enumerate what is representable.
    #[test]
    fn a_temperature_between_two_rungs_is_taken() {
        assert_eq!(temperature_option(Some("0.85")), Some(0.85));
    }

    /// Only the talker's. The code predictor draws the codec's residual detail
    /// and the reference implementation gives it its own knob, so one option
    /// must not quietly move two.
    #[test]
    fn a_temperature_moves_the_talker_and_not_the_code_predictor() {
        let default = GenerationConfig::default();
        let moved = default.sampling.clone().with_temperature(0.7);
        assert_eq!(
            moved,
            Sampling::TopKThenTopP {
                k: 50,
                p: 1.0,
                temperature: 0.7
            }
        );
        assert_eq!(
            default.subtalker_sampling,
            GenerationConfig::default().subtalker_sampling,
            "the code predictor keeps what the checkpoint gave it"
        );
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

#[cfg(test)]
mod voice_resolution_tests {
    //! The voice-resolution matrix, which decides what a checkpoint is
    //! conditioned on. A wrong answer here still produces fluent speech, in
    //! the wrong voice, so every arm is pinned rather than listened to.
    use super::{Conditioning, Kind, RequestError, conditioning_for};
    use crate::voices::{self, Requested};

    fn speakers() -> Vec<String> {
        vec!["ryan".to_string(), "vivian".to_string()]
    }

    fn resolve(
        kind: Kind,
        voice: Requested<'_>,
        configured: Option<&str>,
    ) -> Result<Conditioning, RequestError> {
        conditioning_for(
            kind,
            &speakers(),
            Some("ryan"),
            &|uuid| uuid == "known",
            voice,
            configured,
        )
    }

    fn described(result: Result<Conditioning, RequestError>) -> String {
        match result {
            Ok(Conditioning::Description(d)) => d,
            other => panic!("expected a description, got {other:?}"),
        }
    }

    /// The point of the change: Custom is a declared voice id whose wording
    /// comes from the option, so picking it is what reaches the field.
    #[test]
    fn custom_speaks_the_configured_description() {
        let got = resolve(
            Kind::VoiceDesign,
            Requested::Speaker(voices::CUSTOM_VOICE_ID),
            Some("A hoarse pirate."),
        );
        assert_eq!(described(got), "A hoarse pirate.");
    }

    /// An empty field is not an instruction to describe nothing: Custom has to
    /// fall back to the same voice `default_voice` names, or clearing the field
    /// would leave the voice to the sampler.
    #[test]
    fn custom_with_an_empty_field_speaks_the_default() {
        let got = resolve(
            Kind::VoiceDesign,
            Requested::Speaker(voices::CUSTOM_VOICE_ID),
            None,
        );
        assert_eq!(described(got), voices::DEFAULT_DESCRIPTION);
    }

    /// A named design is its own wording, whatever is typed in the field —
    /// otherwise the option would silently override the picker again.
    #[test]
    fn a_named_design_ignores_the_configured_description() {
        let got = resolve(
            Kind::VoiceDesign,
            Requested::Speaker("deep-narrator-male"),
            Some("A hoarse pirate."),
        );
        assert_eq!(
            described(got),
            voices::design("deep-narrator-male").unwrap()
        );
    }

    /// No voice named is no preference stored, and the picker is showing the
    /// manifest's default — so the field must not speak for it.
    #[test]
    fn naming_no_voice_speaks_the_default_not_the_field() {
        let got = resolve(
            Kind::VoiceDesign,
            Requested::Default,
            Some("A hoarse pirate."),
        );
        assert_eq!(described(got), voices::DEFAULT_DESCRIPTION);
    }

    /// Free text per request still wins over everything, which is what keeps
    /// `described` in this model's `voice_kinds` meaningful.
    #[test]
    fn a_request_description_still_wins() {
        let got = resolve(
            Kind::VoiceDesign,
            Requested::Description("A weary sailor."),
            Some("A hoarse pirate."),
        );
        assert_eq!(described(got), "A weary sailor.");
    }

    #[test]
    fn a_design_model_refuses_an_id_it_does_not_declare() {
        let got = resolve(Kind::VoiceDesign, Requested::Speaker("ryan"), None);
        assert!(matches!(got, Err(RequestError::UnknownVoice(_))), "{got:?}");
    }

    /// The other two families are untouched by any of this: a CustomVoice
    /// checkpoint still conditions on its own speakers and a Base one on a
    /// clone, and neither reads the field.
    #[test]
    fn the_other_families_are_unchanged() {
        assert!(matches!(
            resolve(Kind::CustomVoice, Requested::Speaker("vivian"), Some("A hoarse pirate.")),
            Ok(Conditioning::Speaker(s)) if s == "vivian"
        ));
        assert!(matches!(
            resolve(Kind::CustomVoice, Requested::Default, None),
            Ok(Conditioning::Speaker(s)) if s == "ryan"
        ));
        assert!(matches!(
            resolve(Kind::CustomVoice, Requested::Speaker("custom"), None),
            Err(RequestError::UnknownVoice(_))
        ));
        assert!(matches!(
            resolve(Kind::Base, Requested::Cloned("known"), None),
            Ok(Conditioning::Cloned(id)) if id == "known"
        ));
        assert!(matches!(
            resolve(Kind::Base, Requested::Cloned("absent"), None),
            Err(RequestError::UnknownVoice(_))
        ));
        assert!(matches!(
            resolve(Kind::Base, Requested::Default, None),
            Err(RequestError::UnknownVoice(_))
        ));
    }
}
