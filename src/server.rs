// SPDX-License-Identifier: GPL-3.0-only
//! The `/v1` contract, served over the Unix socket the daemon hands us.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, PoisonError, RwLock};

use axum::Json;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{HeaderMap, HeaderName, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::frames;
use crate::model::{Outcome, Prepared, QwenTts, RequestError};
use crate::model_thread::ModelThread;
use crate::voices::{self, Requested};

/// Where the model is in its lifecycle, as `GET /v1/status` reports it.
#[derive(Debug, Clone)]
enum LoadState {
    /// Serving, but no model has been asked for yet.
    Starting,
    /// A `POST /v1/load` is in flight.
    Loading,
    /// Ready to synthesize.
    Ready { model: String, device: String },
    /// The load failed; the reason reaches the user through the daemon.
    Failed { reason: String },
}

/// Everything the handlers share.
pub struct AppState {
    backend_dir: PathBuf,
    /// The model, which lives on a thread of its own because it is not `Send`.
    /// Every use of it is a job sent there, and jobs run one at a time — the
    /// serialization a mutex used to provide.
    model: ModelThread,
    state: RwLock<LoadState>,
    /// Set by `POST /v1/cancel`, cleared at the start of each synthesis.
    cancelled: AtomicBool,
}

impl AppState {
    /// State for a backend rooted at `backend_dir`.
    #[must_use]
    pub fn new(backend_dir: PathBuf) -> Self {
        Self {
            backend_dir,
            model: ModelThread::spawn(),
            state: RwLock::new(LoadState::Starting),
            cancelled: AtomicBool::new(false),
        }
    }

    fn set_state(&self, next: LoadState) {
        // A poisoned lock means a handler panicked mid-update. The state is a
        // plain enum with no invariant to repair, so recovering is correct and
        // keeps one panic from taking the backend down with it.
        let mut guard = self.state.write().unwrap_or_else(PoisonError::into_inner);
        *guard = next;
    }

    fn state(&self) -> LoadState {
        self.state
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

/// Build the router.
pub fn router(state: Arc<AppState>) -> axum::Router {
    axum::Router::new()
        .route("/v1/ping", get(ping))
        .route("/v1/status", get(status))
        .route("/v1/load", post(load))
        .route("/v1/synthesize", post(synthesize))
        .route("/v1/cancel", post(cancel))
        .route(
            "/v1/voices",
            post(register_voice).layer(DefaultBodyLimit::max(VOICE_BODY_LIMIT)),
        )
        .route("/v1/voices/{voice}", delete(forget_voice))
        .layer(DefaultBodyLimit::max(1 << 20))
        .with_state(state)
}

async fn ping() -> Json<Value> {
    Json(json!({ "status": "success", "message": "pong" }))
}

async fn status(State(s): State<Arc<AppState>>) -> Json<Value> {
    Json(match s.state() {
        LoadState::Starting => json!({ "status": "success", "state": "starting" }),
        LoadState::Loading => json!({ "status": "success", "state": "loading" }),
        LoadState::Ready { model, device } => json!({
            "status": "success",
            "state": "ready",
            "device": device,
            "model": { "name": model },
        }),
        LoadState::Failed { reason } => json!({
            "status": "error",
            "state": "error",
            "reason": reason,
        }),
    })
}

/// `POST /v1/load` body. `provider` is accepted and ignored — it is a
/// compatibility echo from an older identity scheme, and the contract says a
/// new backend should not validate it.
#[derive(Debug, Deserialize)]
struct LoadRequest {
    /// Optional at this layer only so a body missing it is answered with the
    /// contract's own `invalid_model` rather than axum's bare `422`.
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    device: Option<String>,
}

/// `POST /v1/load` — answer `202` immediately and load in the background.
///
/// The daemon polls `GET /v1/status` for readiness, so returning before the
/// model is resident is the contract rather than a shortcut: mapping several
/// gigabytes of weights takes long enough to time out an HTTP request.
async fn load(State(s): State<Arc<AppState>>, body: Option<Json<LoadRequest>>) -> Response {
    let Some(Json(req)) = body else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_model",
            "malformed load body",
        );
    };
    let Some(name) = req.name.filter(|n| !n.trim().is_empty()) else {
        return json_error(StatusCode::BAD_REQUEST, "invalid_model", "no model name");
    };

    s.set_state(LoadState::Loading);
    let state = Arc::clone(&s);
    let device = req.device.clone();

    let queued = s.model.submit(move |slot| {
        match QwenTts::load(&state.backend_dir, &name, device.as_deref()) {
            Ok(model) => {
                let device = model.device_name().to_string();
                *slot = Some(model);
                state.set_state(LoadState::Ready {
                    model: name,
                    device,
                });
            }
            Err(e) => {
                log::error!("load failed: {e:#}");
                state.set_state(LoadState::Failed {
                    reason: format!("{e:#}"),
                });
            }
        }
    });
    if !queued {
        s.set_state(LoadState::Failed {
            reason: "the model thread is gone".to_string(),
        });
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference_failed",
            "the model thread is gone",
        );
    }

    (
        StatusCode::ACCEPTED,
        Json(json!({ "status": "success", "message": "Loading started" })),
    )
        .into_response()
}

async fn cancel(State(s): State<Arc<AppState>>) -> Json<Value> {
    s.cancelled.store(true, Ordering::SeqCst);
    Json(json!({ "status": "success", "message": "Cancelled" }))
}

/// Body limit for `POST /v1/voices`.
///
/// A reference clip arrives as base64 PCM, a third larger than the raw audio,
/// and the manifest's `clone_ref_seconds` budget of 15 seconds at 24 kHz mono
/// `s16le` is 960 KB once encoded. Two mebibytes leaves room for the transcript
/// and for a longer budget later — a body limit that is merely tight fails as a
/// bare `413` with nothing in it to read.
const VOICE_BODY_LIMIT: usize = 2 << 20;

/// `POST /v1/voices` body.
#[derive(Debug, Deserialize)]
struct RegisterVoiceRequest {
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    transcript: Option<String>,
    #[serde(default)]
    sample_rate: Option<u32>,
    #[serde(default)]
    channels: Option<u32>,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    audio: Option<String>,
}

/// `POST /v1/voices` — derive a cloned voice from its reference recording.
///
/// Called once per voice per load, so everything expensive happens here rather
/// than on the synthesis path.
async fn register_voice(
    State(s): State<Arc<AppState>>,
    body: Option<Json<RegisterVoiceRequest>>,
) -> Response {
    let Some(Json(req)) = body else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "unknown_voice",
            "malformed voices body",
        );
    };
    let Requested::Cloned(uuid) = voices::parse(req.voice.as_deref()) else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "unknown_voice",
            "voice is missing or is not a voice:<uuid> id",
        );
    };
    let uuid = uuid.to_string();
    // The daemon downmixes, resamples and trims before it gets here, so these
    // are assertions about a contract rather than a conversion layer: a backend
    // that quietly reinterpreted mismatched audio would clone the wrong voice.
    if req.channels != Some(1) {
        return json_error(
            StatusCode::BAD_REQUEST,
            "unknown_voice",
            "a reference recording has to be mono",
        );
    }
    if req.format.as_deref() != Some("s16le") {
        return json_error(
            StatusCode::BAD_REQUEST,
            "unknown_voice",
            "a reference recording has to be s16le",
        );
    }
    let Some(audio) = req.audio else {
        return json_error(StatusCode::BAD_REQUEST, "unknown_voice", "audio is missing");
    };
    let samples = match decode_reference(&audio) {
        Ok(samples) => samples,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "unknown_voice", &e),
    };
    if !matches!(s.state(), LoadState::Ready { .. }) {
        return json_error(StatusCode::CONFLICT, "not_ready", "no model is loaded");
    }

    let transcript = req.transcript;
    let rate = req.sample_rate;
    let registered = s
        .model
        .run(move |slot| {
            let Some(model) = slot.as_mut() else {
                return Err((
                    StatusCode::CONFLICT,
                    "not_ready",
                    "no model is loaded".to_string(),
                ));
            };
            if !model.clones_voices() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "unknown_voice",
                    "this model does not clone voices".to_string(),
                ));
            }
            let expected = model.reference_sample_rate();
            if rate != Some(expected) {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "unknown_voice",
                    format!("a reference recording has to be sampled at {expected} Hz"),
                ));
            }
            model
                .register_voice(&uuid, transcript.as_deref(), &samples)
                .map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "inference_failed",
                        format!("{e:#}"),
                    )
                })
        })
        .await;

    match registered {
        Some(Ok(())) => Json(json!({ "status": "success" })).into_response(),
        Some(Err((status, code, message))) => json_error(status, code, &message),
        None => {
            log::error!("the model thread did not answer");
            json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_failed",
                "registering the voice failed",
            )
        }
    }
}

/// `DELETE /v1/voices/{voice}` — release a registered cloned voice.
///
/// A voice that was never registered is already released, so this answers 200
/// either way: the daemon's goal is that the backend not be holding it.
async fn forget_voice(State(s): State<Arc<AppState>>, Path(voice): Path<String>) -> Response {
    let Requested::Cloned(uuid) = voices::parse(Some(&voice)) else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "unknown_voice",
            "not a voice:<uuid> id",
        );
    };
    let uuid = uuid.to_string();
    let held = s
        .model
        .run(move |slot| slot.as_mut().is_some_and(|m| m.forget_voice(&uuid)))
        .await;
    if held == Some(true) {
        log::info!("released the voice {voice}");
    }
    Json(json!({ "status": "success" })).into_response()
}

/// Decode a base64 `s16le` reference clip into the samples the model reads.
fn decode_reference(audio: &str) -> Result<Vec<f32>, String> {
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(audio)
        .map_err(|e| format!("the reference recording is not valid base64: {e}"))?;
    if bytes.is_empty() {
        return Err("the reference recording is empty".to_string());
    }
    if bytes.len() % 2 != 0 {
        return Err("the reference recording is not a whole number of s16le samples".to_string());
    }
    let (samples, _) = bytes.as_chunks::<2>();
    Ok(samples
        .iter()
        .map(|b| f32::from(i16::from_le_bytes(*b)) / 32768.0)
        .collect())
}

/// The headers the daemon injects the backend's two `[[options]]` under.
///
/// `x-tts-option-<name>` is the contract's own spelling, with the names
/// [`voices::PRESET_OPTION`] and [`voices::DESCRIPTION_OPTION`] declare; the
/// test below is what keeps these strings and those names the same pair.
/// Written out rather than joined at runtime because they are read once per
/// request and a `HeaderMap` lookup wants a `&str` it does not have to own.
const PRESET_HEADER: &str = "x-tts-option-voice_design_preset";
/// The free-text half of the pair. See [`PRESET_HEADER`].
const DESCRIPTION_HEADER: &str = "x-tts-option-voice_design_description";

/// The voice the backend is configured to design, read off one request.
///
/// Read per request rather than remembered from the load. The daemon reloads
/// the model when an option changes, so remembering would usually work — but
/// the header is what the contract says is authoritative, and a setting that
/// only takes effect after several gigabytes are mapped again is a setting the
/// user will think is broken.
///
/// A header that is not UTF-8 is read as unset. The daemon stores option values
/// as text, so there is nothing a backend could recover from bytes that are not
/// — and refusing the synthesis over it would take the voice away rather than
/// fall back to it.
fn configured_voice(headers: &HeaderMap) -> Option<String> {
    let value = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
    voices::configured(value(PRESET_HEADER), value(DESCRIPTION_HEADER)).map(str::to_string)
}

/// `POST /v1/synthesize` body.
#[derive(Debug, Deserialize)]
struct SynthesizeRequest {
    /// Optional at this layer only so a body missing it is answered with the
    /// contract's own `invalid_text` rather than axum's bare `422`.
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    instructions: Option<String>,
}

/// `POST /v1/synthesize` — stream framed audio as the talker generates it.
async fn synthesize(
    State(s): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Option<Json<SynthesizeRequest>>,
) -> Response {
    let Some(Json(req)) = body else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_text",
            "malformed synthesize body",
        );
    };
    let Some(text) = req.text.filter(|t| !t.trim().is_empty()) else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_text",
            "text is missing or empty",
        );
    };
    if !matches!(s.state(), LoadState::Ready { .. }) {
        return json_error(StatusCode::CONFLICT, "not_ready", "no model is loaded");
    }
    // Resolved here, where the headers are, rather than on the model thread:
    // the job sent there outlives this request's borrow of them.
    let configured = configured_voice(&headers);

    // Validate and tokenize before the status line is sent. Once the response
    // is a 200 the only way to report a bad voice is an error frame, which the
    // daemon surfaces as a synthesis failure rather than a bad request.
    let prepared = s
        .model
        .run(move |slot| {
            let model = slot.as_ref().ok_or(None::<RequestError>)?;
            let sample_rate = model.sample_rate();
            model
                .prepare(
                    &text,
                    voices::parse(req.voice.as_deref()),
                    configured.as_deref(),
                    req.language.as_deref(),
                    req.instructions.as_deref(),
                )
                .map(|p| (p, sample_rate))
                .map_err(Some)
        })
        .await;

    let (prepared, sample_rate) = match prepared {
        Some(Ok(ready)) => ready,
        Some(Err(None)) => {
            return json_error(StatusCode::CONFLICT, "not_ready", "no model is loaded");
        }
        Some(Err(Some(e))) => return request_error(&e),
        None => {
            log::error!("the model thread did not answer");
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_failed",
                "preparing the request failed",
            );
        }
    };

    s.cancelled.store(false, Ordering::SeqCst);

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(4);
    let state = Arc::clone(&s);
    if !s
        .model
        .submit(move |slot| stream_synthesis(slot.as_mut(), &state, &prepared, sample_rate, &tx))
    {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference_failed",
            "the model thread is gone",
        );
    }

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/vnd.super-tts.frames")
        .header(HeaderName::from_static("x-tts-sample-rate"), sample_rate)
        .header(HeaderName::from_static("x-tts-channels"), 1)
        .header(HeaderName::from_static("x-tts-format"), "s16le")
        .body(Body::from_stream(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
        .unwrap_or_else(|e| {
            json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_failed",
                &e.to_string(),
            )
        })
}

/// Run the synthesis, sending frames as the codec decodes them.
///
/// Runs on the model thread. Every exit path sends a terminal frame: the daemon
/// treats a stream that stops without one as a failure, which is exactly right
/// for a crash but would be wrong for the ordinary end of an utterance.
fn stream_synthesis(
    model: Option<&mut QwenTts>,
    state: &Arc<AppState>,
    prepared: &Prepared,
    sample_rate: u32,
    tx: &tokio::sync::mpsc::Sender<Result<bytes::Bytes, std::io::Error>>,
) {
    let send = |bytes: Vec<u8>| tx.blocking_send(Ok(bytes::Bytes::from(bytes))).is_ok();

    let Some(model) = model else {
        let _ = send(frames::error_frame("model not loaded"));
        return;
    };

    let mut samples = 0_u64;
    let outcome = model.synthesize(
        prepared,
        || !state.cancelled.load(Ordering::SeqCst),
        |pcm| {
            samples += pcm.len() as u64;
            // A false here stops generation: either the daemon hung up, or the
            // channel is gone because the request was abandoned.
            frames::audio_frames(pcm).into_iter().all(&send)
        },
    );

    match outcome {
        Ok(Outcome::Finished) => {}
        // `done`, not `error`. The contract describes a cancel as terminating
        // with an error frame, and that reading fits a failure; this is not
        // one. The audio already sent is real speech the user heard, and the
        // stop was asked for. An error frame here would put a synthesis failure
        // in the log every time someone cancels. The sibling Kokoro backend
        // ends a cancelled utterance the same way.
        Ok(Outcome::Stopped) => {
            log::info!("synthesis stopped after {samples} samples");
            let _ = send(frames::done_frame());
            return;
        }
        Err(e) => {
            log::error!("synthesis failed: {e:#}");
            let _ = send(frames::error_frame(&format!("{e:#}")));
            return;
        }
    }

    // Nothing pronounceable in the request. A `done` here would report a
    // successful utterance the user never heard.
    if samples == 0 {
        let _ = send(frames::error_frame("the model produced no audio"));
        return;
    }

    // One mark for the whole utterance. The daemon has already split the text
    // into prosodic units, and the talker reports no alignment inside one, so
    // a finer span would be invented rather than measured.
    let duration_ms = samples * 1000 / u64::from(sample_rate);
    if !send(frames::mark_frame(0, duration_ms, 0, prepared.text_chars())) {
        return;
    }
    let _ = send(frames::done_frame());
}

/// Map a rejected request onto the status and code the contract names for it.
fn request_error(e: &RequestError) -> Response {
    let (status, code) = match e {
        RequestError::UnknownVoice(_) => (StatusCode::BAD_REQUEST, "unknown_voice"),
        RequestError::UnsupportedLanguage(_) => (StatusCode::BAD_REQUEST, "unsupported_language"),
        RequestError::Tokenize(_) => (StatusCode::INTERNAL_SERVER_ERROR, "inference_failed"),
    };
    json_error(status, code, &e.to_string())
}

/// The JSON error envelope: `message` is the contract's code, `detail` is for
/// the person reading the log.
fn json_error(status: StatusCode, code: &str, detail: &str) -> Response {
    (
        status,
        Json(json!({ "status": "error", "message": code, "detail": detail })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    /// A router over an empty backend directory: enough to exercise every
    /// route that does not need weights, which is every route but a successful
    /// load.
    fn app() -> (axum::Router, Arc<AppState>) {
        let state = Arc::new(AppState::new(PathBuf::from("/nonexistent")));
        (router(Arc::clone(&state)), state)
    }

    async fn call(app: axum::Router, req: Request<Body>) -> (StatusCode, Value) {
        let res = app.oneshot(req).await.expect("the router must answer");
        let status = res.status();
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .expect("a JSON body must be readable");
        let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, value)
    }

    fn post(path: &str, body: &Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn ping_answers_pong() {
        let (app, _) = app();
        let req = Request::builder()
            .uri("/v1/ping")
            .body(Body::empty())
            .unwrap();
        let (status, body) = call(app, req).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "success");
        assert_eq!(body["message"], "pong");
    }

    /// A freshly spawned backend has been asked for nothing yet, and the daemon
    /// reads that state to decide whether to send a load.
    #[tokio::test]
    async fn status_starts_before_any_load() {
        let (app, _) = app();
        let req = Request::builder()
            .uri("/v1/status")
            .body(Body::empty())
            .unwrap();
        let (status, body) = call(app, req).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "success");
        assert_eq!(body["state"], "starting");
        assert!(body.get("model").is_none(), "no model has been asked for");
    }

    /// The daemon gates synthesis on `ready`, but a race is still possible, and
    /// the contract names the code it expects to see.
    #[tokio::test]
    async fn synthesize_before_a_load_is_not_ready() {
        let (app, _) = app();
        let (status, body) = call(app, post("/v1/synthesize", &json!({ "text": "hi" }))).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["message"], "not_ready");
    }

    #[tokio::test]
    async fn empty_text_is_refused_as_invalid_text() {
        let (app, _) = app();
        let (status, body) = call(app, post("/v1/synthesize", &json!({ "text": "   " }))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["message"], "invalid_text");
    }

    /// s16le is little-endian pairs scaled by 32768, and the full-scale
    /// negative sample is the one that reaches exactly -1.
    #[test]
    fn a_reference_clip_decodes_as_signed_little_endian_pairs() {
        use base64::Engine as _;
        let pcm: Vec<u8> = [0i16, 32767, -32768, -1]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let encoded = base64::engine::general_purpose::STANDARD.encode(&pcm);
        let samples = decode_reference(&encoded).expect("valid base64 of whole samples");
        assert_eq!(samples.len(), 4);
        assert!((samples[0] - 0.0).abs() < 1e-6);
        assert!((samples[1] - 0.999_969_5).abs() < 1e-6);
        assert!((samples[2] + 1.0).abs() < 1e-6);
        assert!((samples[3] + 0.000_030_5).abs() < 1e-6);
    }

    /// A clip that is not whole samples is a framing error, and reading it as
    /// one sample short would shift every sample after the first byte.
    #[test]
    fn a_clip_that_is_not_whole_samples_is_refused() {
        use base64::Engine as _;
        let odd = base64::engine::general_purpose::STANDARD.encode([1u8, 2, 3]);
        assert!(decode_reference(&odd).is_err());
        assert!(decode_reference("").is_err());
        assert!(decode_reference("not base64!!").is_err());
    }

    /// The voice id is the key the later synthesis sends, so a body without one
    /// is refused before any audio is decoded.
    #[tokio::test]
    async fn registering_without_a_voice_id_is_refused() {
        let (app, _) = app();
        let (status, body) = call(
            app,
            post(
                "/v1/voices",
                &json!({ "voice": "ryan", "channels": 1, "format": "s16le", "audio": "AAA=" }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["message"], "unknown_voice");
    }

    /// The daemon has already downmixed and converted, so a mismatch is a bug
    /// upstream rather than audio to reinterpret.
    #[tokio::test]
    async fn registering_stereo_or_float_audio_is_refused() {
        for body in [
            json!({ "voice": "voice:abc", "channels": 2, "format": "s16le", "audio": "AAA=" }),
            json!({ "voice": "voice:abc", "channels": 1, "format": "f32le", "audio": "AAA=" }),
        ] {
            let (app, _) = app();
            let (status, answer) = call(app, post("/v1/voices", &body)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(answer["message"], "unknown_voice", "{body}");
        }
    }

    /// Registering before a load has nowhere to put the voice.
    #[tokio::test]
    async fn registering_before_a_load_is_not_ready() {
        let (app, _) = app();
        let (status, body) = call(
            app,
            post(
                "/v1/voices",
                &json!({
                    "voice": "voice:abc",
                    "sample_rate": 24000,
                    "channels": 1,
                    "format": "s16le",
                    "audio": "AAAAAA==",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["message"], "not_ready");
    }

    /// The daemon's goal is that the backend not hold the voice, so releasing
    /// one it never had is success — including with no model loaded at all.
    #[tokio::test]
    async fn releasing_a_voice_that_was_never_registered_succeeds() {
        let (app, _) = app();
        let req = Request::builder()
            .method("DELETE")
            .uri("/v1/voices/voice%3Aabc")
            .body(Body::empty())
            .expect("a DELETE must build");
        let (status, body) = call(app, req).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "success");
    }

    /// A path segment that is not a cloned id never named a voice this backend
    /// could hold, and saying so beats a success that did nothing.
    #[tokio::test]
    async fn releasing_something_that_is_not_a_voice_id_is_refused() {
        let (app, _) = app();
        let req = Request::builder()
            .method("DELETE")
            .uri("/v1/voices/ryan")
            .body(Body::empty())
            .expect("a DELETE must build");
        let (status, body) = call(app, req).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["message"], "unknown_voice");
    }

    #[tokio::test]
    async fn a_body_without_text_is_refused() {
        let (app, _) = app();
        let (status, body) = call(app, post("/v1/synthesize", &json!({ "voice": "ryan" }))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["message"], "invalid_text");
    }

    /// Cancel is answered whether or not anything is running: the daemon sends
    /// it on any abandoned utterance and must not have to know which.
    #[tokio::test]
    async fn cancel_sets_the_flag_the_next_synthesis_clears() {
        let (app, state) = app();
        let (status, body) = call(app, post("/v1/cancel", &json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "success");
        assert!(state.cancelled.load(Ordering::SeqCst));
    }

    /// A load answers immediately and reports through `GET /v1/status`, so a
    /// backend directory with no weights in it still answers `202` here and
    /// fails afterwards.
    #[tokio::test]
    async fn a_load_is_accepted_before_the_weights_are_read() {
        let (app, state) = app();
        let (status, body) = call(
            app,
            post(
                "/v1/load",
                &json!({ "name": "qwen3-tts-0.6b-custom-voice" }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(body["status"], "success");
        assert!(matches!(
            state.state(),
            LoadState::Loading | LoadState::Failed { .. }
        ));
    }

    /// The failure has to name the missing file: "load failed" alone leaves the
    /// user with nothing to act on, and this is the message the daemon shows.
    #[tokio::test]
    async fn a_load_with_no_weights_fails_with_the_missing_path() {
        let dir = tempfile::tempdir().unwrap();
        let state = Arc::new(AppState::new(dir.path().to_path_buf()));
        let app = router(Arc::clone(&state));
        let (status, _) = call(app, post("/v1/load", &json!({ "name": "missing-model" }))).await;
        assert_eq!(status, StatusCode::ACCEPTED);

        // The load runs on a blocking thread; give it a moment to fail.
        for _ in 0..200 {
            if let LoadState::Failed { reason } = state.state() {
                assert!(reason.contains("config.json"), "{reason}");
                assert!(reason.contains("models.files"), "{reason}");
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("the load never reported a failure: {:?}", state.state());
    }

    /// The headers and the option names are two spellings of one contract, and
    /// nothing but this holds them together: the daemon forms the header from
    /// the manifest's `name`, so a rename here that stopped short of the
    /// manifest would read an option nobody sets.
    #[test]
    fn the_option_headers_are_the_declared_options() {
        assert_eq!(
            PRESET_HEADER,
            format!("x-tts-option-{}", voices::PRESET_OPTION)
        );
        assert_eq!(
            DESCRIPTION_HEADER,
            format!("x-tts-option-{}", voices::DESCRIPTION_OPTION)
        );
    }

    fn with_options(preset: Option<&str>, description: Option<&str>) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in [(PRESET_HEADER, preset), (DESCRIPTION_HEADER, description)] {
            if let Some(value) = value {
                headers.insert(
                    HeaderName::from_static(name),
                    value.parse().expect("a test header value must be valid"),
                );
            }
        }
        headers
    }

    #[test]
    fn a_picked_design_arrives_as_its_description() {
        let configured = configured_voice(&with_options(Some("Deep narrator (male)"), None));
        assert_eq!(
            configured.as_deref(),
            voices::design("Deep narrator (male)")
        );
    }

    #[test]
    fn a_written_description_beats_the_picked_design() {
        let headers = with_options(Some("Deep narrator (male)"), Some("A hoarse pirate."));
        assert_eq!(
            configured_voice(&headers).as_deref(),
            Some("A hoarse pirate.")
        );
    }

    /// The daemon omits the header for an option the user has not set, which is
    /// most requests to most backends.
    #[test]
    fn no_option_headers_configure_no_voice() {
        assert_eq!(configured_voice(&HeaderMap::new()), None);
    }

    /// A header this backend does not read must not disturb the two it does.
    #[test]
    fn an_unrelated_option_header_is_left_alone() {
        let mut headers = with_options(Some("Soft whisper (female)"), None);
        headers.insert(
            HeaderName::from_static("x-tts-option-request_timeout_seconds"),
            "30".parse().unwrap(),
        );
        assert_eq!(
            configured_voice(&headers).as_deref(),
            voices::design("Soft whisper (female)")
        );
    }

    /// Bytes that are not text are read as no value at all, so the design the
    /// user picked still speaks rather than the request failing over a header.
    #[test]
    fn a_header_that_is_not_text_is_read_as_unset() {
        let mut headers = with_options(Some("Soft whisper (female)"), None);
        headers.insert(
            HeaderName::from_static(DESCRIPTION_HEADER),
            axum::http::HeaderValue::from_bytes(&[0xff, 0xfe]).unwrap(),
        );
        assert_eq!(
            configured_voice(&headers).as_deref(),
            voices::design("Soft whisper (female)")
        );
    }

    /// The option only reaches the model through a request, so a synthesize
    /// carrying it must still be refused before a model is loaded rather than
    /// failing somewhere further in.
    #[tokio::test]
    async fn a_configured_voice_does_not_make_an_unloaded_backend_ready() {
        let (app, _) = app();
        let mut req = post("/v1/synthesize", &json!({ "text": "Hello." }));
        req.headers_mut().insert(
            HeaderName::from_static(PRESET_HEADER),
            "Deep narrator (male)".parse().unwrap(),
        );
        let (status, body) = call(app, req).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["message"], "not_ready");
    }

    #[tokio::test]
    async fn a_malformed_load_body_is_refused() {
        let (app, _) = app();
        let req = Request::builder()
            .method("POST")
            .uri("/v1/load")
            .header("content-type", "application/json")
            .body(Body::from("{ not json"))
            .unwrap();
        let (status, _) = call(app, req).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
}
