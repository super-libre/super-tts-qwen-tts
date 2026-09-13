// SPDX-License-Identifier: GPL-3.0-only
//! Language codes, from what the daemon sends to what the model was trained on.
//!
//! The daemon resolves a request to a BCP-47 tag before it reaches a backend
//! (`en`, `zh-CN`, …) and guarantees it is one of the model's declared
//! `supported_languages`. Qwen3-TTS instead names its languages in English —
//! `english`, `chinese` — and the talker turns that name into one of the
//! `codec_language_id` entries of its config. This module is the join between
//! the two, and it is the reason `backend.toml` can declare ordinary language
//! codes rather than leaking the model's vocabulary into the settings UI.
//!
//! Regional subtags are dropped: the model has one Chinese, not a mainland and
//! a Taiwanese one, so `zh-CN` and `zh-TW` both resolve to `chinese`. Its two
//! dialects, Beijing and Sichuan, are not languages here — the reference
//! implementation selects them through the speaker (`dylan`, `eric`), and
//! [`crate::voices`] leaves that to the model for the same reason.

/// The ten languages Qwen3-TTS speaks, as `(BCP-47 primary subtag, model name)`.
///
/// Sorted by code so the table reads as a lookup rather than a history of
/// which language was added when. A checkpoint that speaks a language missing
/// here is refused it: [`QwenTts::resolve_language`] checks this table first
/// and the talker's own `codec_language_id` second, so both have to be extended
/// together.
///
/// [`QwenTts::resolve_language`]: crate::model::QwenTts
const LANGUAGES: &[(&str, &str)] = &[
    ("de", "german"),
    ("en", "english"),
    ("es", "spanish"),
    ("fr", "french"),
    ("it", "italian"),
    ("ja", "japanese"),
    ("ko", "korean"),
    ("pt", "portuguese"),
    ("ru", "russian"),
    ("zh", "chinese"),
];

/// The language the model uses when a request names none.
///
/// `auto` is the model's own detection, driven by the script and words of the
/// text. The daemon never sends `auto` — it owns detection, because it owns the
/// text — so this is only reached when a request omits `language` entirely.
pub const AUTO: &str = "auto";

/// Translate a BCP-47 tag to the language name Qwen3-TTS expects.
///
/// Returns `None` for a tag the model does not speak, which the caller reports
/// as `unsupported_language`. Matching is on the primary subtag alone and is
/// case-insensitive, so `EN`, `en` and `en-GB` are one language.
#[must_use]
pub fn to_model_name(tag: &str) -> Option<&'static str> {
    let primary = tag.split(['-', '_']).next()?.trim();
    if primary.eq_ignore_ascii_case(AUTO) {
        return Some(AUTO);
    }
    LANGUAGES
        .iter()
        .find(|(code, _)| primary.eq_ignore_ascii_case(code))
        .map(|(_, name)| *name)
}

/// Every BCP-47 code this backend accepts.
///
/// Only the tests read it — the manifest carries its own copy of the list, and
/// the test below is what holds the two together.
#[cfg(test)]
fn supported_codes() -> impl Iterator<Item = &'static str> {
    LANGUAGES.iter().map(|(code, _)| *code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_declared_code_resolves_to_a_model_name() {
        for code in supported_codes() {
            assert!(to_model_name(code).is_some(), "{code} did not resolve");
        }
    }

    /// The daemon sends a resolved tag, which may carry a region the model has
    /// no separate voice for.
    #[test]
    fn a_regional_subtag_resolves_to_its_language() {
        assert_eq!(to_model_name("en-GB"), Some("english"));
        assert_eq!(to_model_name("en-US"), Some("english"));
        assert_eq!(to_model_name("zh-CN"), Some("chinese"));
        assert_eq!(to_model_name("zh-Hant-TW"), Some("chinese"));
        assert_eq!(to_model_name("pt_BR"), Some("portuguese"));
    }

    #[test]
    fn matching_ignores_case() {
        assert_eq!(to_model_name("JA"), Some("japanese"));
        assert_eq!(to_model_name("Ko"), Some("korean"));
    }

    #[test]
    fn auto_is_passed_through() {
        assert_eq!(to_model_name("auto"), Some(AUTO));
        assert_eq!(to_model_name("AUTO"), Some(AUTO));
    }

    /// A language the model cannot speak must be refused rather than guessed
    /// at: synthesizing Hindi text with the English phoneme set produces
    /// confident nonsense, which is worse than an error the user can read.
    #[test]
    fn an_unsupported_language_is_rejected() {
        assert_eq!(to_model_name("hi"), None);
        assert_eq!(to_model_name("ar"), None);
        assert_eq!(to_model_name(""), None);
    }

    /// The manifest and this table have to agree, or the daemon will forward a
    /// language the backend then refuses.
    #[test]
    fn the_table_matches_what_the_manifest_declares() {
        let manifest = include_str!("../backend.toml");
        let declared: Vec<&str> = manifest
            .lines()
            .find_map(|l| {
                let rest = l.trim().strip_prefix("supported_languages")?;
                let list = rest.split_once('[')?.1.split_once(']')?.0;
                Some(
                    list.split(',')
                        .map(|s| s.trim().trim_matches('"'))
                        .filter(|s| !s.is_empty())
                        .collect(),
                )
            })
            .expect("backend.toml declares supported_languages");
        let mut ours: Vec<&str> = supported_codes().collect();
        let mut theirs = declared;
        ours.sort_unstable();
        theirs.sort_unstable();
        assert_eq!(ours, theirs);
    }
}
