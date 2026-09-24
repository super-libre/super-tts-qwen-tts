// SPDX-License-Identifier: GPL-3.0-only
//! Qwen text-to-speech backend for Super TTS.
//!
//! Named for the family rather than a generation: the manifest declares which
//! checkpoints it serves, and today those are Qwen3-TTS. [`model`] documents
//! what a later generation would have to agree with to be served here too.
//!
//! A subprocess backend: the daemon spawns this binary inside a hardened
//! `systemd-run --user` transient unit and drives the `/v1` contract over a
//! Unix socket. Three environment variables are the whole interface —
//! `SUPER_TTS_BACKEND_SOCKET` names the socket to bind, `SUPER_TTS_BACKEND_DIR`
//! the directory holding `backend.toml` and the downloaded weights, and
//! `SUPER_TTS_BACKEND_CACHE_DIR` the one writable place anything may be kept
//! between runs.
//!
//! The sandbox shapes the design more than anything else: there is no network
//! (`PrivateNetwork=yes`) and the backend directory is mounted read-only, so
//! this process never downloads or writes a model file. The daemon fetches
//! everything named in `[[models.files]]` before the first `POST /v1/load`.

mod frames;
mod lang;
#[cfg(test)]
mod manifest_probe;
mod model;
mod model_thread;
mod progress;
mod prompt;
mod qwen3;
mod server;
mod voice_cache;
mod voices;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use tokio::net::UnixListener;

/// Names the Unix socket to bind.
const ENV_SOCKET: &str = "SUPER_TTS_BACKEND_SOCKET";
/// Names the backend's own directory.
const ENV_DIR: &str = "SUPER_TTS_BACKEND_DIR";
/// Names the one writable directory the sandbox grants, where the kernel cache
/// goes. Absent when an older daemon spawned this backend.
const ENV_CACHE_DIR: &str = "SUPER_TTS_BACKEND_CACHE_DIR";

/// The argument that runs the exporter instead of the server.
const EXPORT_KERNELS: &str = "export-kernels";

/// Produce the kernel bundle a release ships, then exit.
///
/// The daemon never passes arguments, so this mode is unreachable from a
/// running install: it is a developer tool, kept in this binary rather than a
/// second one because a bundle is only valid for the exact `CubeCL` the
/// consuming binary links, and two binaries are two chances to drift.
///
/// ```text
/// export-kernels [--warm <model>] <out.bundle> ["Bundle name"]
/// ```
///
/// `--warm` loads the model first, which fills the cache by running the same
/// ladder a real first request does. That is how a release bundle is made:
/// point `SUPER_TTS_BACKEND_CACHE_DIR` at an empty directory so what comes out
/// is this model's cold load and nothing else.
///
/// Without it the cache is exported as it stands, which is for the machine
/// that has already been running the backend and wants to package what it
/// learned. Whatever else that cache accumulated ships too.
///
/// The bundle is only ever good for the GPU it was made on — declare it in
/// `backend.toml` under the `cuda_sm` (or `gfx`) this machine reports.
fn export_kernels(backend_dir: &Path, args: &[String]) -> Result<()> {
    const USAGE: &str = "usage: export-kernels [--warm <model>] <out.bundle> [name]";

    let everything = args.iter().any(|a| a == "--everything");
    let args: Vec<String> = args
        .iter()
        .filter(|a| *a != "--everything")
        .cloned()
        .collect();
    let (warm, rest) = match args.first().map(String::as_str) {
        Some("--warm") => (Some(args.get(1).context(USAGE)?.clone()), &args[2..]),
        _ => (None, args.as_slice()),
    };
    let out = PathBuf::from(rest.first().context(USAGE)?);
    // The name is what a person reads in the manifest of a bundle they are
    // about to install, so it should say which machine it came from.
    let name = rest
        .get(1)
        .cloned()
        .or_else(|| warm.clone())
        .unwrap_or_else(|| "super-tts-qwen-tts".to_string());

    if let Some(model_name) = &warm {
        log::info!(
            "warming {model_name} to export its kernels; against an empty cache this is a cold load and takes minutes"
        );
        let model = model::QwenTts::load(backend_dir, model_name, None, &|_| ())?;
        let device = model.device_name().to_string();
        // Dropped before the export so nothing is still writing to the cache.
        drop(model);
        log::info!("exporting the kernels {model_name} compiled on {device}");
    } else {
        log::info!("exporting the kernel cache as it stands, without loading a model");
    }
    model::export_kernel_bundle(&out, &name, everything)
}

#[tokio::main]
async fn main() -> Result<()> {
    // CubeCL's ROCm compiler, pliron, logs its whole IR after every pass at
    // `info`, which is the level the daemon runs backends at: one cold load
    // wrote a 10 GB log in about seven minutes on an AMD card. `RUST_LOG`
    // still turns it back on, as `pliron=info`.
    env_logger::Builder::new()
        .filter_module("pliron", log::LevelFilter::Warn)
        .parse_env(env_logger::Env::default().default_filter_or("info"))
        .init();

    // Before anything else touches a device: the GPU kernels are compiled at
    // runtime and the configuration that says where to keep them is frozen the
    // first time it is read.
    let cache_dir = std::env::var_os(ENV_CACHE_DIR).map(PathBuf::from);
    match &cache_dir {
        Some(dir) => log::info!("keeping compiled kernels in {}", dir.display()),
        None => log::warn!(
            "{ENV_CACHE_DIR} is not set, so the GPU kernels have nowhere to be kept and are              recompiled on every load; a daemon new enough to grant a cache directory fixes it"
        ),
    }
    model::configure_kernel_cache(cache_dir.as_deref());

    let backend_dir =
        PathBuf::from(std::env::var(ENV_DIR).with_context(|| format!("{ENV_DIR} is not set"))?);

    // Straight after the cache is pointed somewhere and before any device
    // exists. Not per model: nothing in the cache is keyed by one, so the whole
    // build shares a single bundle and it is imported once here rather than on
    // every load.
    model::import_kernel_bundle(&backend_dir);

    // The exporter needs the backend directory and the cache, and nothing else
    // — no socket, no server. Checked here so both are already configured.
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == EXPORT_KERNELS) {
        return export_kernels(&backend_dir, &args[1..]);
    }

    let socket_path = PathBuf::from(
        std::env::var(ENV_SOCKET).with_context(|| format!("{ENV_SOCKET} is not set"))?,
    );

    // The socket directory is the one path the sandbox leaves writable, and a
    // stale socket from a killed unit would make `bind` fail with EADDRINUSE.
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let _ = std::fs::remove_file(&socket_path);

    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("binding {}", socket_path.display()))?;
    log::info!(
        "qwen-tts backend listening on {} (dir {})",
        socket_path.display(),
        backend_dir.display()
    );

    let state = Arc::new(server::AppState::new(backend_dir));
    let app = server::router(state);

    loop {
        let (stream, _) = match listener.accept().await {
            Ok(pair) => pair,
            Err(e) => {
                log::error!("accept failed: {e}");
                continue;
            }
        };
        let service = TowerToHyperService::new(app.clone());
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await
            {
                log::debug!("connection ended: {e}");
            }
        });
    }
}
