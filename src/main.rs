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

use std::path::PathBuf;
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
