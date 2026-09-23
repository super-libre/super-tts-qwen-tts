// SPDX-License-Identifier: GPL-3.0-only
//! Reading fields back out of `backend.toml`, for the tests that hold this
//! crate's tables to the manifest's.
//!
//! Parsed rather than deserialized because the manifest is the schema's shape
//! and not this crate's: pulling in a TOML dependency to read a handful of
//! fields would make the tests depend on a type they do not own. Shared so that
//! the modules checking their own corner of the manifest — [`crate::voices`]
//! for the designs, [`crate::model`] for the sampling option — read it the same
//! way rather than each growing a parser that agrees with the others until it
//! does not.

/// The `[[options]]` block declaring `name`, as its own lines.
pub(crate) fn option_body<'a>(manifest: &'a str, name: &str) -> Option<Vec<&'a str>> {
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
pub(crate) fn option_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let (found, value) = line.trim().split_once('=')?;
    (found.trim() == key).then(|| value.trim().trim_matches('"'))
}

/// The `choices` the option named `name` offers, in the order it lists
/// them, or `None` when it declares none.
///
/// Walks the lines rather than joining them: the list is written one entry
/// to a line, and a joined copy would be a `String` this cannot hand
/// slices of back to its caller.
pub(crate) fn declared_choices<'a>(manifest: &'a str, name: &str) -> Option<Vec<&'a str>> {
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

/// One `key = value` from the `[[options]]` block declaring `name`, unquoted.
pub(crate) fn option_field<'a>(manifest: &'a str, name: &str, key: &str) -> Option<&'a str> {
    option_body(manifest, name)?
        .iter()
        .find_map(|line| option_value(line, key))
}
