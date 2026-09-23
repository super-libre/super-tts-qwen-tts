// SPDX-License-Identifier: GPL-3.0-only
//! Voice ids, from the three shapes the protocol defines to what a checkpoint
//! can actually condition on.
//!
//! `docs/protocol/backend/config.md` gives a `voice` one of three shapes: a
//! bare id is a *preset*, `voice:<uuid>` is a *cloned* voice, and `desc:<text>`
//! is a *described* one. Which shapes reach a backend is set by the
//! `voice_kinds` its manifest declares — the daemon refuses the others before
//! they get here.
//!
//! The three Qwen3-TTS checkpoint families — the generation served today — take
//! one shape each. CustomVoice conditions on one of nine predefined speakers,
//! so its voices are presets. VoiceDesign has no speakers at all and builds a
//! voice from a natural-language description, so its voices are described. The
//! Base checkpoints clone from a recording, so their voices are cloned ones,
//! registered over `POST /v1/voices` before the first synthesis that names
//! them.
//!
//! A described voice is the one shape no request from the settings app can
//! carry: its voice picker offers a model's `[[models.voices]]`, and VoiceDesign
//! declares none, so there is nothing there to pick. The description comes from
//! the backend's own configuration instead — the two `[[options]]` below, which
//! the daemon injects as headers on every request. [`configured`] is where the
//! two become one voice, and [`crate::model::QwenTts::prepare`] is where a
//! request that named its own still wins.

/// A `voice` from a request, classified by its shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Requested<'a> {
    /// The request named no voice. The model picks its own default.
    Default,
    /// A bare id: one of the checkpoint's predefined speakers.
    Speaker(&'a str),
    /// `desc:<text>`: a natural-language description of the voice to build.
    Description(&'a str),
    /// `voice:<uuid>`: a user-cloned voice, carrying the uuid alone.
    ///
    /// The contract suggests keying a backend's cache by the whole wire id.
    /// Keying by the uuid is the same thing once both sides parse through here,
    /// and it means a malformed id is refused in one place rather than stored
    /// under a key no synthesis could ever ask for.
    Cloned(&'a str),
}

/// The `desc:` prefix marking a described voice.
const DESCRIBED_PREFIX: &str = "desc:";
/// The `voice:` prefix marking a cloned voice.
const CLONED_PREFIX: &str = "voice:";

/// Classify the `voice` field of a synthesis request.
///
/// A voice that is present but blank, or a `desc:` with nothing after it, is
/// read as no voice at all rather than as an empty description: an empty
/// instruction steers the model nowhere, so the default is the better answer.
#[must_use]
pub fn parse(voice: Option<&str>) -> Requested<'_> {
    let Some(voice) = voice.map(str::trim).filter(|v| !v.is_empty()) else {
        return Requested::Default;
    };
    if let Some(uuid) = voice.strip_prefix(CLONED_PREFIX) {
        return Requested::Cloned(uuid);
    }
    if let Some(text) = voice.strip_prefix(DESCRIBED_PREFIX) {
        let text = text.trim();
        return if text.is_empty() {
            Requested::Default
        } else {
            Requested::Description(text)
        };
    }
    Requested::Speaker(voice)
}

/// The voice description used by a VoiceDesign model when a request names none
/// and the backend is configured with none either.
///
/// VoiceDesign has no speakers, so it has no `default_voice` in the manifest to
/// fall back on, and a prompt with no instruction at all leaves the voice to
/// whatever the sampler wanders into — it varies between requests, which is the
/// one thing a default must not do. The delivery is deliberately plain: a
/// neutral reading is what a caller who expressed no preference is asking for.
///
/// It names a gender all the same, because "neutral" is a description of tone
/// and pace and the model does not read it as one of gender: leave gender out
/// and the sampler picks, so the default would be a man on one request and a
/// woman on the next. Which one it names is arbitrary — the point is that it
/// names the same one every time, and every entry in [`DESIGNS`] says which it
/// is so a caller who does have a preference can act on it.
pub const DEFAULT_DESCRIPTION: &str =
    "A clear, neutral female voice speaking at a natural pace, with even tone.";

/// The voice id standing for "whatever the description option says".
///
/// The twelve designs are voices with fixed wording; this one is the voice
/// whose wording the user writes. It is a declared `[[models.voices]]` id like
/// any other, so picking it is a per-model voice preference the daemon
/// validates and stores — where the description it stands for stays a backend
/// option, because it is free text and an option is what this backend already
/// has a field for.
///
/// With the field empty it resolves to [`DEFAULT_DESCRIPTION`], which is also
/// what the manifest's `default_voice` names: choosing Custom and writing
/// nothing is the same voice as choosing nothing at all, rather than a prompt
/// with no instruction in it.
pub const CUSTOM_VOICE_ID: &str = "custom";

/// The `[[options]]` name carrying a description written out by hand.
///
/// Only the tests read it — at runtime the name reaches this backend already
/// spelled as a header, and `server.rs` holds that spelling so a lookup costs
/// no allocation. The tests either side are what hold the two together, the
/// way [`crate::lang`] holds its table to the manifest's language list.
#[cfg(test)]
pub const DESCRIPTION_OPTION: &str = "voice_design_description";

/// The pre-made voices a VoiceDesign checkpoint offers, as `(id, label,
/// description)`.
///
/// The id is the voice id a request carries and the daemon stores; the label
/// is what the picker shows; the description is what the model is actually
/// conditioned on. Writing the descriptions here rather than into the manifest
/// keeps the picker readable — twelve sentences in a picker is not a picker —
/// and keeps the wording that steers the model in the same file as the default
/// it falls back to.
///
/// `backend.toml` carries the ids and labels as this model's
/// `[[models.voices]]`, which is what makes them ordinary voices: the daemon
/// lists them beside a CustomVoice checkpoint's speakers and a Base one's
/// clones, validates a pick against them, and stores it per model. A test below
/// holds the two lists together: an id only this table knows is one the daemon
/// would refuse before it ever arrived, and one only the manifest knows is one
/// that resolves to nothing.
///
/// The spread is deliberate — gender, pitch, pace, age and setting each vary
/// across the twelve, six voices to a gender — because a list of twelve
/// near-neighbours would leave a user with nothing to pick between and send
/// them to the description field anyway.
///
/// Every name carries its gender in parentheses, the way the nine CustomVoice
/// speakers do in the manifest, and every description repeats it. Both halves
/// are load-bearing and a test below asserts both: the name is the whole of
/// what the dropdown shows, so a *Deep narrator* that did not say would leave
/// the user picking blind, and the description is the whole of what the model
/// reads, so one that did not say would leave the choice to the sampler and
/// hand back a different gender from one request to the next.
pub const DESIGNS: [(&str, &str, &str); 12] = [
    (
        "neutral-narrator-female",
        "Neutral narrator (female)",
        DEFAULT_DESCRIPTION,
    ),
    (
        "warm-storyteller-female",
        "Warm storyteller (female)",
        "A warm, gentle female voice reading unhurriedly, with soft rounded \
         vowels and a kind, inviting tone.",
    ),
    (
        "deep-narrator-male",
        "Deep narrator (male)",
        "A deep, resonant male voice speaking slowly and calmly, with steady \
         breath and a measured, cinematic delivery.",
    ),
    (
        "bright-presenter-female",
        "Bright presenter (female)",
        "A bright, energetic young female voice speaking quickly and clearly, \
         with lively pitch and an upbeat, confident tone.",
    ),
    (
        "calm-documentary-male",
        "Calm documentary (male)",
        "A composed male voice speaking quietly and precisely, with careful \
         diction and long, even phrases.",
    ),
    (
        "friendly-assistant-male",
        "Friendly assistant (male)",
        "A friendly, conversational male voice at a moderate pace, relaxed and \
         approachable, as if speaking to someone across a table.",
    ),
    (
        "news-anchor-male",
        "News anchor (male)",
        "A crisp, authoritative male voice with clean articulation and a level, \
         professional cadence, projecting without strain.",
    ),
    (
        "gravelly-veteran-male",
        "Gravelly veteran (male)",
        "A rough, gravelly older male voice speaking slowly, weathered and low, \
         with a dry, unhurried delivery.",
    ),
    (
        "soft-whisper-female",
        "Soft whisper (female)",
        "A soft, breathy female voice speaking barely above a whisper, close to \
         the microphone, intimate and hushed.",
    ),
    (
        "cheerful-child-female",
        "Cheerful child (female)",
        "A light, high-pitched young girl's voice, playful and quick, full of \
         curiosity and bounce.",
    ),
    (
        "elegant-host-female",
        "Elegant host (female)",
        "A poised, refined female voice speaking smoothly and unhurriedly, with \
         polished diction and a hint of warmth.",
    ),
    (
        "dramatic-trailer-male",
        "Dramatic trailer (male)",
        "A powerful, theatrical male voice speaking with heavy emphasis and long \
         pauses, dark and intense.",
    ),
];

/// The description a pre-made voice's id stands for.
///
/// Matched exactly, bar surrounding whitespace: the daemon refuses a voice the
/// model does not declare, so the id that arrives is one of these or the
/// request came from something that bypassed that check.
#[must_use]
pub fn design(id: &str) -> Option<&'static str> {
    let id = id.trim();
    DESIGNS
        .iter()
        .find(|(offered, _, _)| *offered == id)
        .map(|(_, _, description)| *description)
}

/// The description the backend is configured with, if any.
///
/// One option rather than the two this backend used to declare: which voice a
/// model speaks in is now a voice id the daemon stores per model, and only the
/// free-text half — the sentence behind [`CUSTOM_VOICE_ID`] — is left with
/// nowhere else to live. A picker's worth of fixed choices is a voice list; a
/// sentence the user writes is not.
///
/// A blank field is the field left alone, not an instruction to say nothing:
/// an empty description steers the model nowhere, so it resolves to `None` and
/// the caller falls back to [`DEFAULT_DESCRIPTION`].
#[must_use]
pub fn configured(description: Option<&str>) -> Option<&str> {
    description.map(str::trim).filter(|d| !d.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest_probe::{declared_choices, option_body, option_value};

    #[test]
    fn a_bare_id_is_a_preset_speaker() {
        assert_eq!(parse(Some("ryan")), Requested::Speaker("ryan"));
        assert_eq!(parse(Some("ono_anna")), Requested::Speaker("ono_anna"));
    }

    #[test]
    fn a_desc_prefix_is_a_description() {
        assert_eq!(
            parse(Some("desc:A calm, deep male voice.")),
            Requested::Description("A calm, deep male voice.")
        );
    }

    #[test]
    fn a_voice_prefix_is_a_clone_reference() {
        assert_eq!(
            parse(Some("voice:0b2f8c1e-1111-2222-3333-444455556666")),
            Requested::Cloned("0b2f8c1e-1111-2222-3333-444455556666")
        );
    }

    #[test]
    fn no_voice_is_the_default() {
        assert_eq!(parse(None), Requested::Default);
    }

    /// Whitespace on either side of a speaker id is the caller's, not part of
    /// the id, and a voice of nothing but whitespace is no voice at all.
    #[test]
    fn blank_and_padded_voices_are_normalized() {
        assert_eq!(parse(Some("")), Requested::Default);
        assert_eq!(parse(Some("   ")), Requested::Default);
        assert_eq!(parse(Some("  ryan  ")), Requested::Speaker("ryan"));
    }

    /// An empty description would condition the model on an empty instruction,
    /// which steers it nowhere; the default description is the honest reading.
    #[test]
    fn an_empty_description_falls_back_to_the_default() {
        assert_eq!(parse(Some("desc:")), Requested::Default);
        assert_eq!(parse(Some("desc:   ")), Requested::Default);
    }

    /// The prefix is only special at the start — a speaker whose name merely
    /// contains it stays a speaker.
    #[test]
    fn a_prefix_matches_only_at_the_start() {
        assert_eq!(
            parse(Some("my_desc:voice")),
            Requested::Speaker("my_desc:voice")
        );
    }

    #[test]
    fn a_design_resolves_to_its_description() {
        assert_eq!(design("neutral-narrator-female"), Some(DEFAULT_DESCRIPTION));
        assert_eq!(
            design("  deep-narrator-male  "),
            design("deep-narrator-male")
        );
        assert!(design("deep-narrator-male").is_some_and(|d| d.contains("resonant")));
        assert_eq!(design("ryan"), None);
        assert_eq!(design(""), None);
    }

    /// Custom is deliberately not in the table: it is the id whose description
    /// comes from the option, so resolving it here would give it fixed wording
    /// and make the field it reads do nothing.
    #[test]
    fn the_custom_voice_resolves_to_no_fixed_design() {
        assert_eq!(design(CUSTOM_VOICE_ID), None);
    }

    /// Both halves of a design have to name its gender: the name is all the
    /// dropdown shows, and the description is all the model reads. A
    /// description that left it out would hand back a different gender from
    /// one request to the next, which is the same failure the default guards
    /// against.
    #[test]
    fn every_design_names_its_gender() {
        for (_, name, description) in DESIGNS {
            let gender = if name.ends_with("(female)") {
                "female"
            } else if name.ends_with("(male)") {
                "male"
            } else {
                panic!("{name} does not say whether it is female or male");
            };
            // A girl's or a boy's voice is the gendered word for a child, so
            // the child voice satisfies this without the adult noun.
            assert!(
                description.contains(gender)
                    || description.contains(if gender == "female" { "girl" } else { "boy" }),
                "{name} is named {gender} but its description does not say so"
            );
        }
    }

    /// Neither gender is the only one on offer. Not a balance the model cares
    /// about — it is what keeps the list useful to whoever opens the dropdown.
    #[test]
    fn the_designs_offer_both_genders() {
        let women = DESIGNS
            .iter()
            .filter(|(_, name, _)| name.ends_with("(female)"))
            .count();
        assert_eq!(women, DESIGNS.len() - women, "{women} of {}", DESIGNS.len());
    }

    /// Every design has to steer the model somewhere, and two designs sharing
    /// a name would make the dropdown's second copy unreachable.
    #[test]
    fn the_designs_are_distinct_and_populated() {
        for (id, name, description) in DESIGNS {
            assert!(!name.trim().is_empty(), "a design has no name");
            // The id is a wire value now — it travels in a request and is
            // stored in the daemon's config — so it has to be the kind of
            // token that survives both, not a sentence with spaces in it.
            assert!(
                !id.is_empty()
                    && id
                        .bytes()
                        .all(|b| b.is_ascii_lowercase() || b == b'-' || b.is_ascii_digit()),
                "{id} is not a plain lowercase id"
            );
            assert_ne!(id, CUSTOM_VOICE_ID, "a design may not shadow Custom");
            assert_eq!(
                DESIGNS.iter().filter(|(i, _, _)| *i == id).count(),
                1,
                "{id} is declared twice"
            );
            assert!(
                description.len() > 30,
                "{name} is described in too few words to steer the model"
            );
            assert_eq!(
                DESIGNS.iter().filter(|(_, n, _)| *n == name).count(),
                1,
                "{name} is offered twice"
            );
        }
    }

    #[test]
    fn a_written_description_is_the_configured_voice() {
        assert_eq!(
            configured(Some("A hoarse pirate.")),
            Some("A hoarse pirate.")
        );
    }

    /// An empty field is the field left alone, not an instruction to say
    /// nothing — Custom has to fall back to the default rather than prompt the
    /// model with a blank.
    #[test]
    fn a_blank_description_asks_for_nothing() {
        assert_eq!(configured(Some("   ")), None);
        assert_eq!(configured(Some("")), None);
    }

    #[test]
    fn configuring_nothing_asks_for_nothing() {
        assert_eq!(configured(None), None);
    }

    /// A description survives whitespace the settings field picked up.
    #[test]
    fn a_written_description_is_trimmed() {
        assert_eq!(
            configured(Some("  A hoarse pirate.  ")),
            Some("A hoarse pirate.")
        );
    }

    /// The manifest and this table have to agree, or the daemon will refuse an
    /// id only this table knows, and store one only the manifest knows — and
    /// the second is the worse half, since it is stored successfully and then
    /// resolves to nothing on every request.
    ///
    /// Ids and labels both: the id is what travels, and the label is the whole
    /// of what the picker shows, so a table that held the gender and a manifest
    /// that did not would leave the user picking blind.
    #[test]
    fn the_designs_match_what_the_manifest_declares() {
        let manifest = include_str!("../backend.toml");
        let declared = declared_voices(manifest, DESIGN_MODEL);
        let ours: Vec<(&str, &str)> = DESIGNS.iter().map(|(id, l, _)| (*id, *l)).collect();
        let (custom, designs) = declared
            .split_last()
            .expect("backend.toml declares the design model's voices");
        assert_eq!(ours, designs);
        // Custom is declared alongside them and deliberately absent from the
        // table: it is the one whose description the option carries.
        assert_eq!(custom.0, CUSTOM_VOICE_ID);
    }

    /// The manifest's `default_voice` is what a request naming none resolves
    /// to, so it has to be a design — and the same one Custom falls back to
    /// with an empty field, or clearing the field would change the voice
    /// rather than restore it.
    #[test]
    fn the_manifests_default_design_is_the_default_description() {
        let manifest = include_str!("../backend.toml");
        let default = model_field(manifest, DESIGN_MODEL, "default_voice")
            .expect("backend.toml gives the design model a default_voice");
        assert_eq!(design(default), Some(DEFAULT_DESCRIPTION));
    }

    /// The free-text field must not declare choices — declaring any would turn
    /// it into a dropdown and refuse every description a user wrote.
    #[test]
    fn the_description_option_is_a_free_text_field() {
        let manifest = include_str!("../backend.toml");
        assert!(option_body(manifest, DESCRIPTION_OPTION).is_some());
        assert_eq!(declared_choices(manifest, DESCRIPTION_OPTION), None);
    }

    /// The checkpoint the designs belong to. Only the tests name it: at
    /// runtime a request is already addressed to one model, and this crate
    /// tells the three families apart by what the checkpoint is, not by what
    /// the manifest called it.
    const DESIGN_MODEL: &str = "qwen3-tts-1.7b-voice-design";

    /// The `[[models]]` block declaring `name`, up to the next top-level table.
    ///
    /// Sub-tables are kept: `[[models.voices]]` is part of the model, and
    /// stopping at the first `[` would cut every voice off the entry.
    fn model_body<'a>(manifest: &'a str, name: &str) -> Option<&'a str> {
        manifest
            .split("\n[[models]]")
            .skip(1)
            .map(|block| block.split("\n[[").next().unwrap_or(block))
            .find(|block| {
                block
                    .lines()
                    .any(|l| option_value(l, "name").is_some_and(|v| v == name))
            })
    }

    /// One scalar field of the model named `name`, from its own lines rather
    /// than a sub-table's.
    fn model_field<'a>(manifest: &'a str, name: &str, key: &str) -> Option<&'a str> {
        model_body(manifest, name)?
            .split("[[models.voices]]")
            .next()?
            .lines()
            .find_map(|l| option_value(l, key))
    }

    /// The `(id, label)` of every `[[models.voices]]` the model declares, in
    /// the order it declares them.
    fn declared_voices<'a>(manifest: &'a str, name: &str) -> Vec<(&'a str, &'a str)> {
        let Some(body) = model_body(manifest, name) else {
            return Vec::new();
        };
        body.split("[[models.voices]]")
            .skip(1)
            .filter_map(|entry| {
                let lines: Vec<&str> = entry.lines().collect();
                let id = lines.iter().find_map(|l| option_value(l, "id"))?;
                let label = lines.iter().find_map(|l| option_value(l, "label"))?;
                Some((id, label))
            })
            .collect()
    }
}
