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

/// The `[[options]]` name carrying a pick from [`DESIGNS`].
///
/// Only the tests read it — at runtime the name reaches this backend already
/// spelled as a header, and `server.rs` holds that spelling so a lookup costs
/// no allocation. The tests either side are what hold the two together, the
/// way [`crate::lang`] holds its table to the manifest's language list.
#[cfg(test)]
pub const PRESET_OPTION: &str = "voice_design_preset";

/// The `[[options]]` name carrying a description written out by hand.
/// Read only by the tests, like [`PRESET_OPTION`].
#[cfg(test)]
pub const DESCRIPTION_OPTION: &str = "voice_design_description";

/// The pre-made voices a VoiceDesign checkpoint offers, as `(name,
/// description)`.
///
/// The name is what the settings dropdown shows and what the daemon stores and
/// injects, since an option's `choices` are values and not labels; the
/// description is what the model is actually conditioned on. Writing the
/// descriptions here rather than into the manifest keeps the dropdown readable
/// — twelve sentences in a picker is not a picker — and keeps the wording that
/// steers the model in the same file as the default it falls back to.
///
/// `backend.toml` carries the names as the `voice_design_preset` option's
/// `choices`, and a test below holds the two lists together: a name only this
/// table knows is one the daemon would refuse before it ever arrived, and a
/// choice only the manifest knows is one that resolves to nothing.
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
pub const DESIGNS: [(&str, &str); 12] = [
    ("Neutral narrator (female)", DEFAULT_DESCRIPTION),
    (
        "Warm storyteller (female)",
        "A warm, gentle female voice reading unhurriedly, with soft rounded \
         vowels and a kind, inviting tone.",
    ),
    (
        "Deep narrator (male)",
        "A deep, resonant male voice speaking slowly and calmly, with steady \
         breath and a measured, cinematic delivery.",
    ),
    (
        "Bright presenter (female)",
        "A bright, energetic young female voice speaking quickly and clearly, \
         with lively pitch and an upbeat, confident tone.",
    ),
    (
        "Calm documentary (male)",
        "A composed male voice speaking quietly and precisely, with careful \
         diction and long, even phrases.",
    ),
    (
        "Friendly assistant (male)",
        "A friendly, conversational male voice at a moderate pace, relaxed and \
         approachable, as if speaking to someone across a table.",
    ),
    (
        "News anchor (male)",
        "A crisp, authoritative male voice with clean articulation and a level, \
         professional cadence, projecting without strain.",
    ),
    (
        "Gravelly veteran (male)",
        "A rough, gravelly older male voice speaking slowly, weathered and low, \
         with a dry, unhurried delivery.",
    ),
    (
        "Soft whisper (female)",
        "A soft, breathy female voice speaking barely above a whisper, close to \
         the microphone, intimate and hushed.",
    ),
    (
        "Cheerful child (female)",
        "A light, high-pitched young girl's voice, playful and quick, full of \
         curiosity and bounce.",
    ),
    (
        "Elegant host (female)",
        "A poised, refined female voice speaking smoothly and unhurriedly, with \
         polished diction and a hint of warmth.",
    ),
    (
        "Dramatic trailer (male)",
        "A powerful, theatrical male voice speaking with heavy emphasis and long \
         pauses, dark and intense.",
    ),
];

/// The description a pre-made voice's name stands for.
///
/// Matched exactly, bar surrounding whitespace: the daemon refuses a write of
/// any value the manifest's `choices` do not offer, so the name that arrives is
/// one of these or the option was set by something that bypassed that check.
#[must_use]
pub fn design(name: &str) -> Option<&'static str> {
    let name = name.trim();
    DESIGNS
        .iter()
        .find(|(offered, _)| *offered == name)
        .map(|(_, description)| *description)
}

/// The voice the backend's configuration asks for, from the two options it
/// declares.
///
/// `description` wins whenever it holds anything. It is the more specific of
/// the two — a sentence the user wrote against a name they merely picked — and
/// the manifest says so where they set it, so a description left in the field
/// overriding a dropdown they later moved is what they were told to expect.
///
/// A preset naming no known design resolves to nothing rather than to the
/// default, so the model's own fallback is what fills in. Both answers sound
/// the same today, since the default *is* the first design; they stop being the
/// same the moment a request carries its own description, which the caller
/// would then lose to a stale pick.
#[must_use]
pub fn configured<'a>(preset: Option<&str>, description: Option<&'a str>) -> Option<&'a str> {
    if let Some(written) = description.map(str::trim).filter(|d| !d.is_empty()) {
        return Some(written);
    }
    let preset = preset.map(str::trim).filter(|p| !p.is_empty())?;
    let found = design(preset);
    if found.is_none() {
        log::warn!("ignoring the configured voice {preset:?}, which is not one this backend has");
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            design("Neutral narrator (female)"),
            Some(DEFAULT_DESCRIPTION)
        );
        assert_eq!(
            design("  Deep narrator (male)  "),
            design("Deep narrator (male)")
        );
        assert!(design("Deep narrator (male)").is_some_and(|d| d.contains("resonant")));
        assert_eq!(design("Ryan"), None);
        assert_eq!(design(""), None);
    }

    /// Both halves of a design have to name its gender: the name is all the
    /// dropdown shows, and the description is all the model reads. A
    /// description that left it out would hand back a different gender from
    /// one request to the next, which is the same failure the default guards
    /// against.
    #[test]
    fn every_design_names_its_gender() {
        for (name, description) in DESIGNS {
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
            .filter(|(name, _)| name.ends_with("(female)"))
            .count();
        assert_eq!(women, DESIGNS.len() - women, "{women} of {}", DESIGNS.len());
    }

    /// Every design has to steer the model somewhere, and two designs sharing
    /// a name would make the dropdown's second copy unreachable.
    #[test]
    fn the_designs_are_distinct_and_populated() {
        for (name, description) in DESIGNS {
            assert!(!name.trim().is_empty(), "a design has no name");
            assert!(
                description.len() > 30,
                "{name} is described in too few words to steer the model"
            );
            assert_eq!(
                DESIGNS.iter().filter(|(n, _)| *n == name).count(),
                1,
                "{name} is offered twice"
            );
        }
    }

    #[test]
    fn a_picked_design_is_the_configured_voice() {
        assert_eq!(
            configured(Some("Deep narrator (male)"), None),
            design("Deep narrator (male)")
        );
    }

    /// The field the manifest calls an override has to override.
    #[test]
    fn a_written_description_beats_the_picked_design() {
        assert_eq!(
            configured(Some("Deep narrator (male)"), Some("A hoarse pirate.")),
            Some("A hoarse pirate.")
        );
    }

    /// An empty field is the field left alone, not an instruction to say
    /// nothing — clearing it has to give the dropdown back.
    #[test]
    fn a_blank_description_leaves_the_design_in_charge() {
        assert_eq!(
            configured(Some("Deep narrator (male)"), Some("   ")),
            design("Deep narrator (male)")
        );
        assert_eq!(
            configured(Some("Deep narrator (male)"), Some("")),
            design("Deep narrator (male)")
        );
    }

    #[test]
    fn configuring_nothing_asks_for_nothing() {
        assert_eq!(configured(None, None), None);
        assert_eq!(configured(Some("  "), Some("  ")), None);
    }

    /// A pick this backend does not have resolves to nothing, which leaves the
    /// model's own default in charge rather than a design chosen at random.
    #[test]
    fn an_unknown_design_is_ignored() {
        assert_eq!(configured(Some("Sea captain"), None), None);
    }

    /// A description survives whitespace the settings field picked up.
    #[test]
    fn a_written_description_is_trimmed() {
        assert_eq!(
            configured(None, Some("  A hoarse pirate.  ")),
            Some("A hoarse pirate.")
        );
    }

    /// The manifest and this table have to agree, or the daemon will refuse a
    /// name only this table knows, and store a choice only the manifest knows.
    #[test]
    fn the_designs_match_what_the_manifest_offers() {
        let manifest = include_str!("../backend.toml");
        let declared = declared_choices(manifest, PRESET_OPTION)
            .expect("backend.toml declares the preset option's choices");
        let ours: Vec<&str> = DESIGNS.iter().map(|(name, _)| *name).collect();
        assert_eq!(ours, declared);
    }

    /// The manifest's default is what the daemon injects for a user who has
    /// picked nothing, so it has to be a design — and the same one the model
    /// falls back to when no option arrives at all, or the voice would change
    /// depending on whether the daemon happened to send the header.
    #[test]
    fn the_manifests_default_design_is_the_default_description() {
        let manifest = include_str!("../backend.toml");
        let default = option_field(manifest, PRESET_OPTION, "default")
            .expect("backend.toml gives the preset option a default");
        assert_eq!(design(default), Some(DEFAULT_DESCRIPTION));
    }

    /// The free-text field must not declare choices — declaring any would turn
    /// it into a second dropdown and refuse every description a user wrote.
    #[test]
    fn the_description_option_is_a_free_text_field() {
        let manifest = include_str!("../backend.toml");
        assert!(option_body(manifest, DESCRIPTION_OPTION).is_some());
        assert_eq!(declared_choices(manifest, DESCRIPTION_OPTION), None);
    }

    /// The `[[options]]` block declaring `name`, as its own lines.
    ///
    /// Parsed rather than deserialized because the manifest is the schema's
    /// shape and not this crate's: pulling in a TOML dependency to read two
    /// fields would make the test depend on a type it does not own.
    fn option_body<'a>(manifest: &'a str, name: &str) -> Option<Vec<&'a str>> {
        let mut blocks = manifest.split("[[options]]").skip(1);
        blocks.find_map(|block| {
            let lines: Vec<&str> = block
                .lines()
                .take_while(|l| !l.trim_start().starts_with('['))
                .collect();
            lines
                .iter()
                .any(|l| option_value(l, "name").is_some_and(|v| v == name))
                .then_some(lines)
        })
    }

    /// One `key = "value"` from an option's block, unquoted.
    fn option_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
        let (found, value) = line.trim().split_once('=')?;
        (found.trim() == key).then(|| value.trim().trim_matches('"'))
    }

    /// One scalar field of the option named `name`.
    fn option_field<'a>(manifest: &'a str, name: &str, key: &str) -> Option<&'a str> {
        option_body(manifest, name)?
            .into_iter()
            .find_map(|l| option_value(l, key))
    }

    /// The `choices` the option named `name` offers, in the order it lists
    /// them, or `None` when it declares none.
    ///
    /// Walks the lines rather than joining them: the list is written one entry
    /// to a line, and a joined copy would be a `String` this cannot hand
    /// slices of back to its caller.
    fn declared_choices<'a>(manifest: &'a str, name: &str) -> Option<Vec<&'a str>> {
        let body = option_body(manifest, name)?;
        let opens = body
            .iter()
            .position(|l| option_value(l, "choices").is_some())?;
        let mut choices = Vec::new();
        for line in &body[opens..] {
            let line = line.split_once('[').map_or(*line, |(_, rest)| rest);
            let (line, closes) = line
                .split_once(']')
                .map_or((line, false), |(entries, _)| (entries, true));
            choices.extend(
                line.split(',')
                    .map(|c| c.trim().trim_matches('"'))
                    .filter(|c| !c.is_empty()),
            );
            if closes {
                return Some(choices);
            }
        }
        None
    }
}
