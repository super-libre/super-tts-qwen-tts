# Super TTS — Qwen TTS backend

[![coverage](https://img.shields.io/endpoint?url=https://super-libre.github.io/super-tts-qwen-tts/coverage.json)](https://super-libre.github.io/super-tts-qwen-tts/)

Qwen's text-to-speech models as a subprocess backend for
[Super TTS](https://github.com/super-libre/super-tts). Ten languages, 24 kHz
output, nine preset voices — or no preset voice at all, and a voice written out
in words instead.

The backend is named for the family and the models for their generation. Today
it serves [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS); a later generation
is a new entry in `backend.toml` rather than a new repository, and the two would
sit side by side under one installed backend.

## What it is

The Super TTS daemon does not compile model inference in-tree. It discovers
backends on disk and drives each one over a `/v1` HTTP contract on a Unix
socket. This is one such backend: a single binary that runs the models through
[Burn](https://github.com/tracel-ai/burn) and answers `POST /v1/synthesize`
with framed PCM.

Burn is what makes this the widest-reaching backend here. Its kernels are
compiled at runtime by CubeCL, so one binary per accelerator covers every GPU
generation the driver can compile for — and the accelerators include ROCm and
Vulkan, which the candle-based backends cannot reach at all.

Synthesis has two halves, and the split is why speech starts before the
utterance is finished:

```
daemon ──POST /v1/synthesize──▶ talker ──▶ 12.5 frames/s of 16 codec tokens
                                                                  │
                                                                  ▼
       ◀── [audio][audio]…[mark][done] ──── codec decoder ──▶ 24 kHz audio
```

The talker generates one frame at a time, so audio is decoded and sent in
two-second chunks while the rest is still being generated. Each chunk is decoded
together with the 50 frames before it, whose audio is then discarded, which
makes the streamed result sample-for-sample what a single decode would have
produced. Without that context the chunk seams are audible.

## Models

Five Qwen3-TTS checkpoints, one loaded at a time. Only the selected model's
files are downloaded, so the others cost nothing.

| Model | Voices | Size on disk | `instructions` |
|---|---|---|---|
| `qwen3-tts-0.6b-custom-voice` | nine presets | 2.5 GB | ignored |
| `qwen3-tts-1.7b-custom-voice` | nine presets | 4.5 GB | honored |
| `qwen3-tts-1.7b-voice-design` | described | 4.5 GB | honored |
| `qwen3-tts-0.6b-base` | cloned | 2.5 GB | ignored |
| `qwen3-tts-1.7b-base` | cloned | 4.5 GB | honored |

All five speak German, English, Spanish, French, Italian, Japanese, Korean,
Portuguese, Russian and Chinese.

The nine preset voices are `ryan` and `aiden` (English), `vivian`, `serena` and
`uncle_fu` (Mandarin), `dylan` (Beijing) and `eric` (Sichuan), `ono_anna`
(Japanese), and `sohee` (Korean). Each sounds best in its own language. The two
dialects are selected by the speaker rather than by the language, which is what
the reference implementation does.

### Designed voices

The VoiceDesign model builds its voice from a description rather than speaking
with a recorded one, so a voice id can be a description:

```
desc:A calm, deep male voice, speaking slowly with a warm tone.
```

That is free text, and a picker cannot offer free text. So the twelve pre-made
designs are declared as ordinary voices — *Neutral narrator (female)*,
*Gravelly veteran (male)*, and ten more, six of each gender, each name saying
which, because the name is all the picker shows. The manifest carries their ids
and labels; `src/voices.rs` holds the sentence each id stands for, and a test
there keeps the two lists the same twelve.

Declared voices are what the settings app's picker offers, so they are chosen
the way a CustomVoice speaker or a cloned voice is: per model, validated by the
daemon against the model that has to resolve it, and stored without reloading
the checkpoint.

A thirteenth, `custom`, is the one whose wording is not fixed. It stands for
whatever the backend's `voice_design_description` option is set to:

| Option | Shape | What it does |
|---|---|---|
| `voice_design_description` | text field | The voice `custom` builds. Empty means `custom` speaks the same neutral description `default_voice` names, so clearing the field restores the default rather than prompting the model with a blank. |

The daemon injects it as an `x-tts-option-*` header on every request.

Precedence runs from the most specific to the least: a request carrying its own
`desc:` wins over everything; a request naming one of the twelve gets that
design, whatever the field says; `custom` gets the field, or the neutral
description when it is empty; and a request naming no voice at all gets that
same neutral description — a prompt with no instruction leaves the voice to the
sampler, and it would then differ from one request to the next.

The other four models ignore the option. A CustomVoice checkpoint conditions on
one of its nine speakers and a Base one on a clone, so neither has anywhere to
put a description, and refusing their requests over a setting left behind from a
model the user has since switched away from would be the wrong answer.

### Cloned voices

The two Base checkpoints have no voices of their own: they speak in a voice
cloned from a recording. The daemon registers that recording once per load over
`POST /v1/voices` — mono `s16le` at 24 kHz, trimmed to the thirty seconds the
manifest budgets — and later syntheses name the voice by its id alone.

There are two ways to clone, and which one runs depends on whether the stored
voice has a transcript:

- **without one**, the talker is conditioned on the speaker embedding the
  reference clip produces;
- **with one**, the clip is *also* encoded back into codec frames and fed in
  front of the request as an in-context example, which is what the reference
  implementation does when it has the words.

`clone_needs_transcript` is therefore left unset: both work, and setting it
would make the daemon refuse a clip stored without words that this backend can
serve perfectly well.

A Base checkpoint asked for no voice at all is refused rather than answered.
Conditioned on nothing it speaks in a voice that changes from one request to the
next, and there is no default that could stand in for the one the caller meant.

### Sampling

The talker picks each frame of speech by sampling, so the same text spoken twice
is not the same recording. How freely it picks is the backend's other option:

| Option | Shape | What it does |
|---|---|---|
| `temperature` | dropdown, `0.6` – `1.2` | How freely the talker picks each frame. Unset leaves the `0.9` the checkpoints' own `generation_config.json` ships. |

It is a dropdown rather than a box because the daemon refuses to store a value
the manifest does not offer, and a temperature is exactly the setting where a
typo is both easy and quiet: `0.09` parses, and speaks. The ladder ends where it
does for a reason — under about 0.6 the talker starts repeating a frame until it
reaches the generation cap, and over about 1.2 it wanders off the text.

It reaches the talker and nothing else. The code predictor, which draws the
codec's residual detail rather than the shape of the utterance, keeps what the
checkpoint gave it; the reference implementation gives that its own
`subtalker_temperature`, and one setting must not quietly move two.

Lowering it is not a way to make the model faster or shorter. Measured over
24 seeds × 40 sentences on the 1.7B CustomVoice checkpoint, the seed alone moves
the silence between words by more than the audible part of this range does, so
reach for it to change how the delivery *feels*, not to fix a particular
utterance you did not like.

**The seed is fixed** at `299792458`, in `GenerationConfig`. It is what makes a
request reproducible: the same text, voice and temperature give the same audio
every time. It is not tuned, and there is nothing to tune — a sweep of 24 seeds
found the shipped one exactly median, and the best seed on half the sentences
was worth about 5% of the silence on the other half.

### What is not here

**`speed`.** These models have no rate control. The contract says a backend
that cannot vary rate ignores the field rather than resampling to fake it, so
this one ignores it.

### Adding a later generation

Nothing in the code names Qwen3, and what varies between checkpoints is read
from the files: the frame rate and sample rate from the codec config, the
speakers and languages from the talker config. Three things still have to line
up before a new generation can be added to the manifest. `src/qwen3` has to
implement it, `Kind` in `src/model.rs` has to recognize its
`tts_model_type`, and the chat template in `src/prompt.rs` has to be the one it
was trained with — the reference tokenization test is what would catch a
template that has moved.

## Requirements

A GPU is not required but is strongly recommended. The talker is a transformer
generating 12.5 frames per second of speech. Measured on an RTX 3090 with a
warm kernel cache, the 0.6B CustomVoice model loads in 3 seconds, sends its
first audio 0.3 seconds into a request and synthesizes at about 6x real time —
19 seconds of speech in 3.2. On a CPU it is slower than real time; the 0.6B
model is the one to try without a GPU.

Releases ship eight builds. On Linux: a CPU build for x86_64 and aarch64, CUDA
12 and CUDA 13, ROCm, and Vulkan. On Apple Silicon Macs: a CPU build and Metal.
The daemon picks the one matching the machine, and ranks a native backend above
Vulkan above the CPU. There is no compute-capability axis — see the manifest
for why.

Only CUDA computes the talker in bf16, the type the checkpoints were trained in;
every other GPU gets f16, because CubeCL cannot compile bf16 there. On Vulkan
its SPIR-V backend emits bf16 arithmetic, which `SPV_KHR_bfloat16` does not
allow — bf16 there is for conversions, dot products and cooperative matrices
only — so those shaders are invalid on any Vulkan driver, and NVIDIA's segfaults
compiling them rather than rejecting them. On ROCm it compiles through LLVM,
whose lowering has no bf16 type at all, and Metal's backend reports none. f16
holds the model: the largest value in a 1.7B prefill is 9.6e3, against the 6.5e4
f16 reaches. A device without f16 gets f32. On Vulkan the codec decoder's
convolutions run in f16 too, with everything around them in f32: a Vulkan device
computes f32 convolutions without its matrix units, and on an RTX 3090 that left
the codec alone slower than real time, 2.9 seconds of decoding for every 2 of
audio. In f16 they take 0.1 seconds. f16 has the 10-bit mantissa of the TF32
that CUDA runs them at, and the decoded audio stays 52 to 65 dB from an f32
decode, around 60, where the decoder's own noise is 65.

Measured on the RTX 3090 with NVIDIA's 610.57.04 driver, the 1.7B CustomVoice
model on Vulkan synthesizes at 3.0 to 3.5x real time, against 3.7 to 4.2x on
CUDA, and loads in 71 seconds from an empty cache. The kernel bundle has no
Vulkan entries yet, so that is every first load.

Weights are downloaded by the daemon before the first load. This process has no
network at all — it runs with `PrivateNetwork=yes` and a read-only backend
directory — so it can neither fetch nor write a model file.

### The kernel cache

CubeCL compiles every GPU kernel it meets at runtime. A synthesis needs a few
hundred of them, about twenty seconds' worth, spread over the first frames and
the first decode. They are then kept on disk, so only the first run of a build
pays.

There is one place to keep them: `SUPER_TTS_BACKEND_CACHE_DIR`, which the
daemon creates and adds to the sandbox's writable paths. Everything else the
backend can see is read-only, and the writable `/tmp` is `PrivateTmp` and dies
with the unit. A daemon too old to grant the directory still works — the
backend logs a warning at startup and recompiles the kernels on every load.

Compilation is not the whole cost. CubeCL also tunes each operation against
its candidate kernels the first time it sees it, and it keys those entries on
the *shape* of the problem — which, for the prefix pass that runs once over the
whole prompt, means the prompt's length. So the load does not stop at mapping
the weights: it generates two seconds of speech at each of five prompt lengths
spanning `max_input_chars`, and throws all of it away. That walks the talker,
the code predictor, the sampler and the codec decoder at the shapes real
requests use.

`ready` then means ready rather than ready-after-one-more-long-wait, and the
wait lands while the daemon is still showing the load's progress.

The ladder is not decoration. With a single warmed length, requests near it
were fast and everything else stalled on its first use — a 140-character
request generated at 2.7 frames per second for a whole utterance, ten seconds
of audio taking forty-seven to produce, on a build whose 300-character
warm-up had covered 220 and 300 perfectly well.

When the cache turns out to be cold — the ladder is its own probe, taking
three seconds warm and minutes cold — one more pass generates thirty seconds
of speech. That is past the point where the talker's key/value cache first
doubles, which re-captures the graph and tunes wider shapes; twenty-five frames
never reach it, and the first real request was paying for it instead.

A Base checkpoint has no voice to warm up with, since it only speaks in one that
was registered. So the warm-up invents one: five seconds of two tones under a
tremolo, cloned through the real speaker encoder and the real codec encoder, and
released again when the ladder is done. That covers the encoders and a
generation conditioned on an embedding rather than on a speaker. What it cannot
cover is the length of somebody's actual recording — the encoders key on that
too — so registering a voice can tune once for a clip length not seen before.
That cost lands in `POST /v1/voices`, which happens once per voice, rather than
in a synthesis.

What that wait costs, on an RTX 3090:

| Load | Time | What it pays for |
|---|---|---|
| Nothing cached | ~4 minutes | Compiling a few hundred kernels and tuning every shape the warm-up walks. Once per GPU. |
| After a backend upgrade | under a minute | Compiling only: the cache keys kernels by build, but the tuning results survive it. |
| Same build again | 5.5 seconds | Mapping the weights, and a warm-up that finds everything already there. |

The first row is the one to design around, and it is the reason the cache
directory exists at all — without it, *every* load is that row, and without the
warm-up the same four minutes land on whoever sends the first request.

While it loads, `GET /v1/status` says how far it has got, in the `phase`,
`step` and `progress` fields the contract defines. `phase` is `initial_setup`
on the first load of a model with a build and `loading` after: a marker beside
the kernels, written once a warm-up runs to the end, tells them apart, so
clearing the cache makes the next load an initial setup again. `step` is
`loading_weights`, measured by the bytes of the checkpoints read, then
`building_kernels` on an initial setup or `warming_up` after. The warm-up is
measured by the entries CubeCL writes to the cache — 1754 of them on CUDA and
726 on Vulkan from empty — plus one per frame it generates, which is what keeps
the bar moving where nothing is compiled. The daemon fails a load whose step and
progress stand still for two minutes; on the RTX 3090 the longest stretch
without either moving was 4.7 seconds on CUDA and 1.6 on Vulkan.

### Shipping a warm cache

The four-minute row above is work that is identical on every machine with the
same GPU, so it does not have to be done on every machine. CubeCL can export a
warm cache as a *bundle*, and this backend ships one in `kernels/`, inside the
release tarball beside the binary.

What it holds is the tuning, not the compiled kernels. On an RTX 3090 that is
147 MB of PTX against **904 KB** of autotune results — and the small half is
both the expensive one to produce and the durable one:

| | size | cost to redo | survives a rebuild |
|---|---:|---|---|
| Compiled kernels (PTX) | 147 MB | under a minute | no — keyed by the source that generated them |
| Autotune results | 904 KB | the rest of the four minutes | yes — keyed by operation and shape |

So the bundle is autotune-only. Shipping the PTX as well would multiply the
tarball by 160 to save the minute, and it would have to be rebuilt and
re-uploaded for every release.

**One file for every GPU, and for every model.** Nothing in the cache is keyed
by model — the namespaces are keyed by CubeCL version, device and kernel family
— so the five checkpoints share whatever shapes they share, and they share most
of them. Warming all five into one cache costs almost nothing over warming the
first:

| warmed, in order | its ladder | cache after |
|---|---:|---:|
| `0.6b-base` | 78.4s | 37 MB |
| `0.6b-custom-voice` | 10.1s | 40 MB |
| `1.7b-base` | 33.5s | 51 MB |
| `1.7b-custom-voice` | 6.6s | 55 MB |
| `1.7b-voice-design` | 5.2s | 55 MB |

Each model after the first costs a fraction of it, and the last adds nothing at
all: only the jump from 0.6B to 1.7B brings genuinely new shapes. All five
together are 261 entries and 904 KB, against 187 and 608 KB for `0.6b-base`
alone — so covering the whole backend costs about 300 KB more than covering one
model of it. Entries for another runtime are simply never looked up, which is
what makes merging every runtime into one file free for the machines that do
not match. Within a runtime they are shared, though: an autotune namespace names
the runtime and the device index (`device-0-0-cuda`), not the GPU model, so any
NVIDIA card reuses the picks this file was tuned with on an RTX 3090 rather
than tuning its own. It is imported once at startup, before any device exists:

```
imported 261 kernel-cache entries from kernels/autotune.bundle in 3.5ms
  (9 namespaces, 0 already present, 0 refused)
```

Nothing about it can make a load fail. A missing file, a corrupt one, or one
warmed on a GPU nobody here has all end in the same place — a warning and the
cold load that happened before bundles existed.

**Adding an architecture.** Run the exporter on the hardware it is for. With
`--warm` it loads the model first, filling the cache by running the same ladder
above; point `SUPER_TTS_BACKEND_CACHE_DIR` at an empty directory so what comes
out is that cold load and nothing else:

```sh
SUPER_TTS_BACKEND_DIR=~/.local/share/super-tts/backends/app.super-tts.qwen-tts \
SUPER_TTS_BACKEND_CACHE_DIR=$(mktemp -d) \
CUDA_CACHE_PATH=~/.cache/qwen-tts-export/nv \
  ./super-tts-backend-qwen-tts export-kernels \
      --warm qwen3-tts-0.6b-base \
      kernels/autotune.bundle \
      "RTX 3090 Linux"
```

The two cache paths want opposite lifetimes, which is easy to get backwards.
`SUPER_TTS_BACKEND_CACHE_DIR` must be **empty every time**: it is what the
bundle is cut from, and a directory carrying another model's entries ships them
too. `CUDA_CACHE_PATH` should be **the same path every time**: it holds the
driver's PTX-to-SASS translations, which no bundle carries and which cost about
110 MB of work per cold load, so reusing it makes repeated exports much faster.

Set it to something, though. Running outside the daemon means running outside
the sandbox, and the NVIDIA driver defaults to `$HOME/.nv/ComputeCache` — so an
export with this unset writes into the cache every other CUDA program on the
machine shares, and quietly makes later measurements on this backend look
better than a new user's would. Under the daemon the question does not arise:
the sandbox sets `CUDA_CACHE_PATH` inside the one writable directory it grants.

Without `--warm` it exports the cache as it stands, which is for a machine that
has been running the backend already and wants to package what it learned.
`--everything` includes the compiled kernels, which is for measuring what that
147 MB would buy rather than for shipping.

To *add* to the file rather than replace it, import the existing bundle into the
empty cache first — the exporter writes whatever the cache holds, and importing
is insert-only, so warming on a second GPU and exporting again yields a file
covering both.

The exporting binary must be the same build as the consuming one: the CubeCL
version is part of every namespace, which is why the exporter lives in this
binary rather than a tool beside it.

### Building

```sh
git clone https://github.com/super-libre/super-tts-qwen-tts
just build-release          # the pure-Rust CPU backend
just build-cuda             # needs the CUDA headers — no GPU, no compute capability
just build-rocm             # needs the ROCm headers
just build-vulkan           # needs nothing; the loader is found at runtime
just build-metal            # macOS; needs nothing beyond Xcode's SDK
```

There is no submodule and no C toolchain to install. The first build is slow
because Burn is a git dependency and has to be fetched and compiled.

Each build carries exactly one accelerator, which is why the recipes pass
`--no-default-features`. Cargo features are additive, so `--features cuda` on
its own would keep the default `flex` backend too and link both.

`just test` runs the suite. One test tokenizes the chat template and compares
against ids produced by the reference `onig` tokenizer; it skips unless
`tokenizer.json` is present, so `just test-tokenizer` fetches it once and runs
everything. `just ci` is the full local gate.

**Burn comes from a fork.** `Cargo.toml` pins `jorge-menjivar/burn` at
`e7897a65`, which is upstream Burn plus six fixes to `burn-cubecl-fusion` the
model needs — a fused reduce that resolved its output reference against its
inputs and crashed, and five more. The revision is not interchangeable with an
upstream one until those land.

**The model itself is vendored**, in `src/qwen3`, from that fork's `qwen3-tts`
example. It is a copy rather than a dependency because the example is a demo
crate: it also pulls `clap`, `hf-hub` and `tokenizers/onig`, and cargo unifies
features across the graph, so depending on it would put a C regex library and
an HTTP stack inside a backend that is cross-compiled and runs with no network.
The module's header records the revision it came from; re-syncing is a diff.

## Installing it

Tag a release and the workflow publishes the tarballs and `backend.toml`; the
daemon installs it from the registry like any other backend. For a local build:

```sh
just stage
```

which puts the binary in the repo root under the name `backend.toml` declares,
making the directory installable with the daemon's import-from-directory path.

## Layout

| Path | What it holds |
|---|---|
| `src/main.rs` | Socket setup; the two environment variables that are the whole interface. |
| `src/server.rs` | The `/v1` routes, and the error codes the contract names. |
| `src/model.rs` | Loading, generation, and the chunked decode that makes streaming seamless. |
| `src/model_thread.rs` | The thread the model lives on, because Burn's generation state is not `Send`. |
| `src/qwen3/` | The model itself, vendored from Burn's `qwen3-tts` example. |
| `src/prompt.rs` | The chat template, and its cross-check against the reference tokenizer. |
| `src/voices.rs` | The three voice id shapes, which ones a checkpoint can resolve, and the twelve designed voices. |
| `src/lang.rs` | BCP-47 codes to the language names the talker conditions on. |
| `src/frames.rs` | The response framing, encoded independently of the daemon's decoder. |
| `backend.toml` | The manifest: models, voices, options, files and their hashes, release assets. |
| `clippy.toml` | The Qwen checkpoint-family names, so `doc_markdown` stops reading them as items. |

## License

GPL-3.0-only. See `LICENSE`, and `NOTICE` for the third-party work this derives
from and the Apache-2.0 weights the daemon fetches at runtime.
