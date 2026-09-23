// SPDX-License-Identifier: GPL-3.0-only
//! Derived cloning artifacts, kept on disk between loads.
//!
//! Registering a cloned voice derives two things from its reference recording:
//! a speaker embedding, and — when the clip came with a transcript — the codec
//! codes that make it an in-context example. Deriving them is expensive in a
//! way that has nothing to do with their size. The codec encoder runs over the
//! whole clip as one tensor, so every clip *length* is a shape `CubeCL` has not
//! tuned yet, and tuning is the slow half of meeting any new shape: a first
//! registration of a twenty-second clip can hold the request open for the
//! better part of a minute, with the user's finger on a Preview button that
//! looks broken.
//!
//! What comes out the other side is tiny. At 12.5 frames a second and sixteen
//! codes a frame, twenty seconds of reference is 4000 codes — around 17 KB
//! beside the embedding, against a checkpoint of several gigabytes. So the cost
//! is in the producing, never in the keeping, and the answer is to produce it
//! once and write it down: a hit here turns the second registration of a voice,
//! and every registration after the next daemon restart, into a file read.
//!
//! The cache lives under the one writable directory the sandbox grants, beside
//! the kernel cache that is there for the same reason. Every failure is a
//! silent fall back to deriving: a cache that cannot be read or written must
//! cost a slow registration, never a failed one.

use std::path::{Path, PathBuf};

/// Magic at the head of every entry, so a file that is not one of ours — or is
/// a truncated one of ours — is rejected rather than parsed into nonsense.
const MAGIC: &[u8; 4] = b"QTVC";

/// Entry format version. Bumping it makes every existing entry a miss, which
/// is the whole invalidation story: entries are derived data and re-deriving
/// them is always correct.
const VERSION: u16 = 1;

/// Names the writable directory the sandbox grants. Kept in step with
/// `main.rs`, which reads the same variable to place the kernel cache.
const ENV_CACHE_DIR: &str = "SUPER_TTS_BACKEND_CACHE_DIR";

/// What registering a voice derives, in the form it is written down.
#[derive(Debug, Clone, PartialEq)]
pub struct Derived {
    /// The speaker embedding, as its raw values.
    pub embedding: Vec<f32>,
    /// The transcript's token ids and the clip's codec codes, when the clip was
    /// registered with a transcript.
    pub icl: Option<(Vec<u32>, Vec<u32>)>,
}

/// Where entries are kept, or `None` when no directory was granted.
///
/// An older daemon spawns this backend without one; the backend then simply
/// derives every time, as it always did.
#[must_use]
pub fn dir() -> Option<PathBuf> {
    let dir = std::env::var_os(ENV_CACHE_DIR).map(PathBuf::from)?;
    Some(dir.join("voices"))
}

/// The filename an entry is stored under.
///
/// Everything the artifact depends on goes into the key: the checkpoint that
/// derived it, the transcript that shaped the in-context half, and the samples
/// themselves. The samples matter beyond identifying the voice — the daemon
/// trims a stored clip to the model's `clone_ref_seconds` before sending it, so
/// raising that budget hands over a *different* recording under the same voice
/// id, and keying on the id alone would answer with codes for audio the model
/// is no longer being given.
#[must_use]
pub fn key(model: &str, transcript: Option<&str>, pcm: &[f32]) -> String {
    let mut h = Fnv::new();
    h.write(model.as_bytes());
    // Length-prefixed rather than concatenated, so a transcript that ends where
    // the next field begins cannot be confused with one that does not.
    h.write(&(model.len() as u64).to_le_bytes());
    match transcript {
        Some(t) => {
            h.write(b"t");
            h.write(t.as_bytes());
            h.write(&(t.len() as u64).to_le_bytes());
        }
        None => h.write(b"-"),
    }
    h.write(&(pcm.len() as u64).to_le_bytes());
    for sample in pcm {
        h.write(&sample.to_le_bytes());
    }
    format!("{:016x}-{}.voice", h.finish(), pcm.len())
}

/// Read the entry for `key`, or `None` when there is not a usable one.
#[must_use]
pub fn load(dir: &Path, key: &str) -> Option<Derived> {
    let bytes = std::fs::read(dir.join(key)).ok()?;
    match decode(&bytes) {
        Ok(derived) => Some(derived),
        Err(e) => {
            log::warn!("ignoring the cached voice {key}: {e}");
            None
        }
    }
}

/// Write the entry for `key`. Best-effort: a failure is logged and the caller
/// carries on with what it derived.
pub fn store(dir: &Path, key: &str, derived: &Derived) {
    if let Err(e) = write(dir, key, derived) {
        log::warn!("not caching the derived voice {key}: {e}");
    }
}

fn write(dir: &Path, key: &str, derived: &Derived) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    // Written beside the entry and renamed over it, so a load that races a
    // write — or an interrupted one — never reads half a file.
    let tmp = dir.join(format!("{key}.partial"));
    std::fs::write(&tmp, encode(derived))?;
    std::fs::rename(&tmp, dir.join(key))
}

/// The entry's bytes: the magic, the version, then each vector length-prefixed.
fn encode(derived: &Derived) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    push_f32s(&mut out, &derived.embedding);
    match &derived.icl {
        Some((ids, codes)) => {
            out.push(1);
            push_u32s(&mut out, ids);
            push_u32s(&mut out, codes);
        }
        None => out.push(0),
    }
    out
}

fn decode(bytes: &[u8]) -> Result<Derived, String> {
    let mut r = Reader { bytes, at: 0 };
    if r.take(4)? != MAGIC {
        return Err("not a voice cache entry".into());
    }
    let version = u16::from_le_bytes(r.take(2)?.try_into().map_err(|_| "truncated version")?);
    if version != VERSION {
        return Err(format!("version {version}, expected {VERSION}"));
    }
    let embedding = r.f32s()?;
    let icl = match r.take(1)?[0] {
        0 => None,
        1 => Some((r.u32s()?, r.u32s()?)),
        other => return Err(format!("unknown in-context flag {other}")),
    };
    if r.at != bytes.len() {
        return Err("trailing bytes".into());
    }
    Ok(Derived { embedding, icl })
}

fn push_f32s(out: &mut Vec<u8>, values: &[f32]) {
    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn push_u32s(out: &mut Vec<u8>, values: &[u32]) {
    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// A bounds-checked walk over an entry's bytes.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).ok_or("length overflow")?;
        let slice = self.bytes.get(self.at..end).ok_or("truncated entry")?;
        self.at = end;
        Ok(slice)
    }

    fn len(&mut self) -> Result<usize, String> {
        let bytes: [u8; 8] = self.take(8)?.try_into().map_err(|_| "truncated length")?;
        usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| "length out of range".to_string())
    }

    fn f32s(&mut self) -> Result<Vec<f32>, String> {
        let n = self.len()?;
        let bytes = self.take(n.checked_mul(4).ok_or("length overflow")?)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect())
    }

    fn u32s(&mut self) -> Result<Vec<u32>, String> {
        let n = self.len()?;
        let bytes = self.take(n.checked_mul(4).ok_or("length overflow")?)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(u32::from_le_bytes)
            .collect())
    }
}

/// FNV-1a, 64-bit.
///
/// Hand-rolled rather than pulled in: the standard library's hasher is
/// explicitly not stable across releases, and a key that changes with the
/// compiler turns every entry into a miss after a toolchain bump. A dependency
/// would do, but sixteen lines that are fixed forever cost less. Collisions
/// only ever cost a wrong *cache* answer, never a wrong voice — see the note on
/// [`key`] for everything folded in.
struct Fnv(u64);

impl Fnv {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x1000_0000_01b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::{Derived, decode, encode, key, load, store};

    fn sample() -> Derived {
        Derived {
            embedding: vec![0.5, -0.25, 1.0],
            icl: Some((vec![1, 2, 3], vec![9, 8, 7, 6])),
        }
    }

    #[test]
    fn an_entry_survives_a_round_trip() {
        let with_icl = sample();
        assert_eq!(decode(&encode(&with_icl)), Ok(with_icl));

        // A clip registered without a transcript clones from the embedding
        // alone, and its entry has to say so rather than storing empty codes.
        let embedding_only = Derived {
            embedding: vec![1.0, 2.0],
            icl: None,
        };
        assert_eq!(decode(&encode(&embedding_only)), Ok(embedding_only));
    }

    #[test]
    fn a_damaged_entry_is_rejected_rather_than_parsed() {
        let good = encode(&sample());
        assert!(decode(b"").is_err(), "empty");
        assert!(decode(b"nope").is_err(), "wrong magic");
        assert!(decode(&good[..good.len() - 3]).is_err(), "truncated");

        let mut trailing = good.clone();
        trailing.push(0);
        assert!(decode(&trailing).is_err(), "trailing bytes");

        let mut version = good.clone();
        version[4] = 99;
        assert!(decode(&version).is_err(), "unknown version");
    }

    /// The key has to move when anything the artifact was derived from moves —
    /// most of all the samples, since raising `clone_ref_seconds` sends a longer
    /// trim of the same voice and must not answer with the old codes.
    #[test]
    fn the_key_covers_every_input() {
        let pcm = [0.1, 0.2, 0.3];
        let base = key("qwen3-tts-0.6b-base", Some("hello"), &pcm);

        assert_ne!(base, key("qwen3-tts-1.7b-base", Some("hello"), &pcm));
        assert_ne!(base, key("qwen3-tts-0.6b-base", Some("hell"), &pcm));
        assert_ne!(base, key("qwen3-tts-0.6b-base", None, &pcm));
        assert_ne!(base, key("qwen3-tts-0.6b-base", Some("hello"), &pcm[..2]));
        assert_ne!(
            base,
            key("qwen3-tts-0.6b-base", Some("hello"), &[0.1, 0.2, 0.4])
        );
        // And it must not move on its own, or every entry is a miss.
        assert_eq!(base, key("qwen3-tts-0.6b-base", Some("hello"), &pcm));
    }

    #[test]
    fn a_stored_entry_reads_back_and_a_missing_one_is_none() {
        let dir = std::env::temp_dir().join(format!("qwen-voice-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let k = key("m", Some("t"), &[0.0, 1.0]);

        assert!(load(&dir, &k).is_none(), "nothing stored yet");
        store(&dir, &k, &sample());
        assert_eq!(load(&dir, &k), Some(sample()));
        assert!(load(&dir, "absent.voice").is_none());

        // A partial file left by an interrupted write is not an entry.
        assert!(!dir.join(format!("{k}.partial")).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
