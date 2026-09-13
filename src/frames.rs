// SPDX-License-Identifier: GPL-3.0-only
//! The `POST /v1/synthesize` response framing.
//!
//! `[u8 kind][u32 len little-endian][payload]`, per the Super TTS backend
//! contract. Written here rather than taken from a shared crate on purpose: a
//! backend is third-party code to the daemon, and encoding the wire format
//! independently is what makes the daemon's decoder tests a check of the
//! format instead of a round-trip against itself.

/// Frame discriminants, as they appear in the leading byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    /// Raw PCM in the format named by the response headers.
    Audio = 0x01,
    /// A JSON mark aligning output audio to input text.
    Mark = 0x02,
    /// Terminal success.
    Done = 0x03,
    /// Terminal failure.
    Error = 0x04,
}

/// Largest payload the daemon will accept in one frame.
pub const MAX_FRAME_LEN: usize = 1 << 20;

/// Audio bytes per frame.
///
/// Comfortably under [`MAX_FRAME_LEN`], and small enough that the daemon can
/// start playing early: at 24 kHz mono s16le this is about 680 ms.
pub const AUDIO_CHUNK_BYTES: usize = 32 * 1024;

/// Encode one frame.
///
/// # Panics
/// Panics if `payload` exceeds [`MAX_FRAME_LEN`]. Callers chunk audio to
/// [`AUDIO_CHUNK_BYTES`] and every other payload is small fixed JSON, so an
/// oversized frame is a bug here rather than input to tolerate.
#[must_use]
pub fn encode(kind: Kind, payload: &[u8]) -> Vec<u8> {
    assert!(
        payload.len() <= MAX_FRAME_LEN,
        "frame payload {} exceeds the {MAX_FRAME_LEN}-byte cap",
        payload.len()
    );
    let len = u32::try_from(payload.len()).expect("checked against MAX_FRAME_LEN");
    let mut out = Vec::with_capacity(5 + payload.len());
    out.push(kind as u8);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Encode `samples` as s16le audio frames, split to [`AUDIO_CHUNK_BYTES`].
///
/// The codec decoder emits `f32` in roughly `[-1, 1]`, but the daemon accepts `s16le`
/// as a first-class format and it halves the bytes on the socket, so samples are
/// narrowed here. Values are clamped before scaling: the decoder occasionally
/// overshoots `1.0`, and letting that wrap would turn a loud sample into a
/// full-scale click of the opposite sign.
#[must_use]
pub fn audio_frames(samples: &[f32]) -> Vec<Vec<u8>> {
    let mut pcm = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        #[allow(clippy::cast_possible_truncation)]
        let v = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
        pcm.extend_from_slice(&v.to_le_bytes());
    }
    pcm.chunks(AUDIO_CHUNK_BYTES)
        .map(|c| encode(Kind::Audio, c))
        .collect()
}

/// Encode a terminal `error` frame carrying a human-readable message.
#[must_use]
pub fn error_frame(message: &str) -> Vec<u8> {
    let payload = serde_json::json!({ "message": message }).to_string();
    encode(Kind::Error, payload.as_bytes())
}

/// Encode the terminal `done` frame.
#[must_use]
pub fn done_frame() -> Vec<u8> {
    encode(Kind::Done, b"{}")
}

/// Encode a `mark` aligning an audio span to a span of the request text.
#[must_use]
pub fn mark_frame(start_ms: u64, end_ms: u64, start_char: u32, end_char: u32) -> Vec<u8> {
    let payload = serde_json::json!({
        "start_ms": start_ms,
        "end_ms": end_ms,
        "start_char": start_char,
        "end_char": end_char,
    })
    .to_string();
    encode(Kind::Mark, payload.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_one(bytes: &[u8]) -> (u8, Vec<u8>) {
        let len = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
        assert_eq!(bytes.len(), 5 + len, "frame length prefix must be exact");
        (bytes[0], bytes[5..].to_vec())
    }

    #[test]
    fn a_frame_is_a_kind_a_little_endian_length_and_its_payload() {
        let (kind, payload) = parse_one(&encode(Kind::Mark, b"hi"));
        assert_eq!(kind, 0x02);
        assert_eq!(payload, b"hi");
    }

    #[test]
    fn every_kind_uses_its_contract_discriminant() {
        assert_eq!(encode(Kind::Audio, b"")[0], 0x01);
        assert_eq!(encode(Kind::Mark, b"")[0], 0x02);
        assert_eq!(encode(Kind::Done, b"")[0], 0x03);
        assert_eq!(encode(Kind::Error, b"")[0], 0x04);
    }

    #[test]
    fn an_empty_payload_still_carries_a_length_prefix() {
        assert_eq!(encode(Kind::Done, b""), vec![0x03, 0, 0, 0, 0]);
    }

    #[test]
    fn samples_narrow_to_little_endian_s16() {
        let frames = audio_frames(&[0.0, 0.5, -0.5]);
        assert_eq!(frames.len(), 1);
        let (kind, pcm) = parse_one(&frames[0]);
        assert_eq!(kind, 0x01);
        let got: Vec<i16> = pcm
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| i16::from_le_bytes(*c))
            .collect();
        assert_eq!(got, vec![0, 16383, -16383]);
    }

    /// The model overshoots `[-1, 1]` often enough that this is the difference
    /// between a loud sample and a full-scale click of the opposite sign.
    #[test]
    fn samples_past_full_scale_clamp_instead_of_wrapping() {
        let frames = audio_frames(&[9.0, -9.0]);
        let (_, pcm) = parse_one(&frames[0]);
        let got: Vec<i16> = pcm
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| i16::from_le_bytes(*c))
            .collect();
        assert_eq!(got, vec![i16::MAX, -i16::MAX]);
    }

    #[test]
    fn long_audio_splits_into_capped_frames() {
        // Three chunks' worth of samples, plus a remainder.
        let n = (AUDIO_CHUNK_BYTES / 2) * 3 + 17;
        let frames = audio_frames(&vec![0.0; n]);
        assert_eq!(frames.len(), 4);
        for f in &frames {
            assert!(f.len() - 5 <= AUDIO_CHUNK_BYTES);
            assert!(f.len() - 5 <= MAX_FRAME_LEN);
        }
        let total: usize = frames.iter().map(|f| f.len() - 5).sum();
        assert_eq!(total, n * 2, "no samples lost across the split");
    }

    #[test]
    fn no_audio_produces_no_frames() {
        assert!(audio_frames(&[]).is_empty());
    }

    #[test]
    fn an_error_frame_carries_its_message_where_the_daemon_reads_it() {
        let (kind, payload) = parse_one(&error_frame("model not loaded"));
        assert_eq!(kind, 0x04);
        let v: serde_json::Value = serde_json::from_slice(&payload).unwrap();
        assert_eq!(v["message"], "model not loaded");
    }

    #[test]
    fn a_mark_carries_both_spans() {
        let (kind, payload) = parse_one(&mark_frame(0, 250, 0, 5));
        assert_eq!(kind, 0x02);
        let v: serde_json::Value = serde_json::from_slice(&payload).unwrap();
        assert_eq!(v["start_ms"], 0);
        assert_eq!(v["end_ms"], 250);
        assert_eq!(v["start_char"], 0);
        assert_eq!(v["end_char"], 5);
    }
}
