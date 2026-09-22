//! Resolving what the user typed into a job UUID.
//!
//! The leaderboard prints short IDs (the first [`SHORT_LEN`] hex characters, as git does with
//! commits), so every command that takes a job has to accept one. A full UUID still works and
//! costs no query.

use anyhow::{bail, Result};
use proteus_storage::repository::ProteusRepository;
use uuid::Uuid;

/// How many characters of a job UUID the CLI prints. 8 hex characters is 4 billion values —
/// far past any local job table, and the same length git uses.
pub const SHORT_LEN: usize = 8;

/// Shortest prefix accepted. Below this, a match is more likely a coincidence than an intent.
const MIN_PREFIX: usize = 4;

/// The first [`SHORT_LEN`] characters of a job ID, for display.
pub fn short(id: Uuid) -> String {
    id.as_simple().to_string()[..SHORT_LEN].to_string()
}

/// Resolve a full UUID or a unique prefix of one against the job table.
///
/// Ambiguity is an error naming the candidates rather than a silent pick of the first.
pub async fn resolve(repo: &ProteusRepository, input: &str) -> Result<Uuid> {
    if let Ok(id) = Uuid::parse_str(input) {
        return Ok(id);
    }
    if input.len() < MIN_PREFIX {
        bail!(
            "'{input}' is neither a job UUID nor a prefix of one \
             (a prefix needs at least {MIN_PREFIX} characters)"
        );
    }
    // Stored IDs are hyphenated, so a bare hex prefix only matches up to the first hyphen at
    // position 8. That is exactly SHORT_LEN, so every printed short ID resolves; longer bare
    // hex prefixes are hyphenated back before matching.
    let candidates = repo.find_job_ids_by_prefix(&hyphenate(input), 4).await?;
    match candidates.len() {
        0 => bail!("No job matches '{input}'"),
        1 => Ok(candidates[0]),
        n => bail!(
            "'{input}' matches {n} jobs ({}{}) — use more characters",
            candidates
                .iter()
                .take(3)
                .map(|id| short(*id))
                .collect::<Vec<_>>()
                .join(", "),
            if n > 3 { ", …" } else { "" }
        ),
    }
}

/// Insert the UUID hyphens a bare hex prefix is missing, so `8c716e905b74` matches the stored
/// `8c716e90-5b74-…`. Input already containing hyphens is left alone.
fn hyphenate(input: &str) -> String {
    if input.contains('-') {
        return input.to_string();
    }
    const GROUPS: [usize; 4] = [8, 4, 4, 4];
    let mut out = String::with_capacity(input.len() + 4);
    let mut rest = input;
    for group in GROUPS {
        if rest.len() <= group {
            break;
        }
        out.push_str(&rest[..group]);
        out.push('-');
        rest = &rest[group..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_printed_short_id_is_a_valid_bare_prefix() {
        let id = Uuid::parse_str("8c716e90-5b74-45ed-9dc7-872408e5aaa0").unwrap();
        assert_eq!(short(id), "8c716e90");
        assert_eq!(hyphenate(&short(id)), "8c716e90");
    }

    #[test]
    fn bare_hex_longer_than_a_group_gets_its_hyphens_back() {
        assert_eq!(hyphenate("8c716e905b74"), "8c716e90-5b74");
        assert_eq!(hyphenate("8c716e905b7445ed"), "8c716e90-5b74-45ed");
        assert_eq!(
            hyphenate("8c716e905b7445ed9dc7872408e5aaa0"),
            "8c716e90-5b74-45ed-9dc7-872408e5aaa0"
        );
    }

    #[test]
    fn an_already_hyphenated_prefix_is_left_alone() {
        assert_eq!(hyphenate("8c716e90-5b74"), "8c716e90-5b74");
    }
}
