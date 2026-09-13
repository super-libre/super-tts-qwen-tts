// SPDX-License-Identifier: GPL-3.0-only
//! The chat template the served checkpoints were trained with.
//!
//! The talker is a Qwen3 transformer, so its text input is a chat transcript
//! rather than a bare sentence: the text to speak arrives as an assistant turn
//! and a voice instruction as a user turn, each wrapped in the `<|im_start|>` /
//! `<|im_end|>` markers of the Qwen chat format. `candle_transformers` takes
//! the already-tokenized ids and documents the exact strings they must come
//! from, so getting these wrong does not fail loudly — it shifts the prompt out
//! of the distribution the model was trained on and degrades the voice.
//!
//! Building the strings is kept apart from tokenizing them so the templates can
//! be asserted without an 11 MB `tokenizer.json` on disk.
//!
//! These are Qwen3-TTS's templates. A later generation trained on a different
//! transcript shape would need its own, and the reference tokenization test
//! below is what would catch the difference.

/// Wrap the text to speak in the turn the talker expects.
///
/// The trailing `<|im_start|>assistant\n` is the open turn the model generates
/// into; without it the model has no cue that it is its turn to speak.
#[must_use]
pub fn input_text(text: &str) -> String {
    format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
}

/// Wrap a voice or delivery instruction in the user turn the talker expects.
///
/// This is what steers a VoiceDesign checkpoint to a described voice, and what
/// carries `instructions` to a 1.7B CustomVoice one.
#[must_use]
pub fn instruct_text(instruct: &str) -> String {
    format!("<|im_start|>user\n{instruct}<|im_end|>\n")
}

/// Wrap the transcript of a cloning reference in the assistant turn the talker
/// reads it as.
///
/// This is the *said* half of an in-context example: the words the reference
/// recording speaks, in the same turn shape a generated utterance occupies, so
/// the codes of that recording read as something the model itself just said.
/// Unlike [`input_text`] it opens no new turn — the example is complete, and
/// the turn the model generates into is opened by the request that follows it.
#[must_use]
pub fn reference_text(transcript: &str) -> String {
    format!("<|im_start|>assistant\n{transcript}<|im_end|>\n")
}

/// Join a voice description with any separate delivery guidance.
///
/// A VoiceDesign request can carry both: the `desc:` voice says who is
/// speaking, and `instructions` says how. The model takes one instruction, so
/// they are concatenated into one sentence-separated string rather than one of
/// them being dropped — dropping the second would silently ignore something the
/// caller asked for.
#[must_use]
pub fn join_instructions(description: &str, instructions: Option<&str>) -> String {
    let extra = instructions.map(str::trim).filter(|s| !s.is_empty());
    match extra {
        None => description.trim().to_string(),
        Some(extra) => {
            let description = description.trim();
            if description.is_empty() {
                return extra.to_string();
            }
            let sep = if description.ends_with(['.', '!', '?', '。', '！', '？']) {
                " "
            } else {
                ". "
            };
            format!("{description}{sep}{extra}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_text_turn_opens_an_assistant_turn_to_generate_into() {
        assert_eq!(
            input_text("Hello."),
            "<|im_start|>assistant\nHello.<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// The reference turn is closed: the example is something already said, and
    /// opening a turn there would tell the model to speak the transcript again.
    #[test]
    fn the_reference_turn_is_closed() {
        assert_eq!(
            reference_text("Hello."),
            "<|im_start|>assistant\nHello.<|im_end|>\n"
        );
        assert!(!reference_text("Hello.").ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn the_instruction_turn_is_a_closed_user_turn() {
        assert_eq!(
            instruct_text("Speak slowly."),
            "<|im_start|>user\nSpeak slowly.<|im_end|>\n"
        );
    }

    /// Text is placed verbatim. Markup normalization is the daemon's job, and
    /// re-escaping here would change what the user asked to have spoken.
    #[test]
    fn text_is_not_altered() {
        let text = "  Ünicode, 中文, and \"quotes\" — kept.  ";
        assert!(input_text(text).contains(text));
    }

    #[test]
    fn a_description_alone_is_the_whole_instruction() {
        assert_eq!(join_instructions("A calm voice.", None), "A calm voice.");
        assert_eq!(
            join_instructions("A calm voice.", Some("  ")),
            "A calm voice."
        );
    }

    /// Both halves survive: the voice and the delivery are different requests
    /// and the model takes one string.
    #[test]
    fn a_description_and_instructions_are_joined_into_one_instruction() {
        assert_eq!(
            join_instructions("A calm, deep male voice.", Some("Read it slowly.")),
            "A calm, deep male voice. Read it slowly."
        );
    }

    /// A description that does not end a sentence gets one, so the two halves
    /// do not run together into a single garbled clause.
    #[test]
    fn an_unterminated_description_gains_a_separator() {
        assert_eq!(
            join_instructions("A calm, deep male voice", Some("Read it slowly.")),
            "A calm, deep male voice. Read it slowly."
        );
    }

    #[test]
    fn a_full_width_terminator_counts_as_a_terminator() {
        assert_eq!(
            join_instructions("平静的男声。", Some("慢慢读。")),
            "平静的男声。 慢慢读。"
        );
    }

    #[test]
    fn instructions_alone_stand_on_their_own() {
        assert_eq!(
            join_instructions("  ", Some("Read it slowly.")),
            "Read it slowly."
        );
    }
}

/// Cross-checks the templates and the tokenizer against the reference.
///
/// The templates only pay off if they tokenize the way the reference
/// implementation does, and this crate reads `tokenizer.json` through
/// `tokenizers` with the `fancy-regex` engine rather than the default `onig`
/// one — Qwen's pre-tokenizer pattern uses a lookahead, and `fancy-regex`
/// supplies it without a C library. The two engines are different regex
/// implementations of the same pattern, so "they agree" is an assertion, not an
/// assumption.
///
/// The ids below were produced by the Python `tokenizers` 0.22.2 package, which
/// is the `onig` build, over `Qwen/Qwen3-0.6B`'s `tokenizer.json`. The test
/// skips when that file is absent — it is 11 MB and the daemon fetches it in
/// production; `just test-tokenizer` fetches it and runs the whole suite.
#[cfg(test)]
mod tokenizer_tests {
    use super::*;

    /// Where `just fetch-tokenizer` puts the file, overridable for a checkout
    /// that keeps it elsewhere.
    fn tokenizer_path() -> std::path::PathBuf {
        std::env::var_os("SUPER_TTS_TEST_TOKENIZER").map_or_else(
            || {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("target/test-backend/tokenizer.json")
            },
            std::path::PathBuf::from,
        )
    }

    #[test]
    fn the_templates_tokenize_the_way_the_reference_does() {
        let path = tokenizer_path();
        let Ok(tokenizer) = tokenizers::Tokenizer::from_file(&path) else {
            eprintln!(
                "skipping: no tokenizer at {} — run `just test-tokenizer`",
                path.display()
            );
            return;
        };
        let encode = |s: &str| -> Vec<u32> {
            tokenizer
                .encode(s, false)
                .expect("encoding must succeed")
                .get_ids()
                .to_vec()
        };

        // Plain English: the ordinary case, and the one that pins the chat
        // markers themselves (151_644 is `<|im_start|>`, 151_645 `<|im_end|>`).
        assert_eq!(
            encode(&input_text("Hello there, this is a test.")),
            vec![
                151_644, 77091, 198, 9707, 1052, 11, 419, 374, 264, 1273, 13, 151_645, 198,
                151_644, 77091, 198
            ]
        );

        // The instruction turn, which is what steers a described voice.
        assert_eq!(
            encode(&instruct_text("A calm, deep male voice.")),
            vec![
                151_644, 872, 198, 32, 19300, 11, 5538, 8593, 7743, 13, 151_645, 198
            ]
        );

        // Non-Latin script with full-width punctuation: the byte-level fallback.
        assert_eq!(
            encode(&input_text("你好，世界。")),
            vec![
                151_644, 77091, 198, 108_386, 3837, 99489, 1773, 151_645, 198, 151_644, 77091, 198
            ]
        );

        // Apostrophes, a decimal, runs of spaces and a newline — every branch
        // of the pre-tokenizer pattern, including the `\s+(?!\S)` lookahead
        // that is the whole reason the engine choice matters.
        assert_eq!(
            encode(&input_text(
                "It's 3.14, isn't it?   Two  spaces.\nAnd a newline."
            )),
            vec![
                151_644, 77091, 198, 2132, 594, 220, 18, 13, 16, 19, 11, 4436, 944, 432, 30, 256,
                9043, 220, 12621, 624, 3036, 264, 39027, 13, 151_645, 198, 151_644, 77091, 198
            ]
        );
    }
}
