//! Pairing the residues of two structures of the same protein: a model against its deposited
//! entry, a design candidate against its parent. Shared by the viewer's superposition and by
//! [`crate::variant`], so the two cannot pair residues differently.

/// Chain ID, residue number and insertion code: what identifies a residue across two files.
pub type ResidueId = (String, isize, Option<String>);

/// Index pairs `(target, reference)` of residues with the same identity, in target order. A
/// duplicated identity (which a well-formed file does not have) pairs its first occurrence.
pub fn pair_residues(target: &[ResidueId], reference: &[ResidueId]) -> Vec<(usize, usize)> {
    let mut by_id: std::collections::HashMap<&ResidueId, usize> =
        std::collections::HashMap::with_capacity(reference.len());
    for (j, id) in reference.iter().enumerate() {
        by_id.entry(id).or_insert(j);
    }
    let mut used = vec![false; reference.len()];
    let mut pairs = Vec::new();
    for (i, id) in target.iter().enumerate() {
        if let Some(&j) = by_id.get(id) {
            if !used[j] {
                used[j] = true;
                pairs.push((i, j));
            }
        }
    }
    pairs
}

/// Global alignment (Needleman–Wunsch) of two residue-name sequences; returns the aligned
/// index pairs. Match +2, mismatch −1, gap −1: identical stretches dominate, a point mutation
/// stays aligned rather than opening two gaps.
pub fn align_by_sequence(a: &[String], b: &[String]) -> Vec<(usize, usize)> {
    let (n, m) = (a.len(), b.len());
    let mut score = vec![vec![0i32; m + 1]; n + 1];
    for (i, row) in score.iter_mut().enumerate() {
        row[0] = -(i as i32);
    }
    for (j, cell) in score[0].iter_mut().enumerate() {
        *cell = -(j as i32);
    }
    for i in 1..=n {
        for j in 1..=m {
            let s = if a[i - 1] == b[j - 1] { 2 } else { -1 };
            score[i][j] = (score[i - 1][j - 1] + s)
                .max(score[i - 1][j] - 1)
                .max(score[i][j - 1] - 1);
        }
    }
    let (mut i, mut j, mut pairs) = (n, m, Vec::new());
    while i > 0 && j > 0 {
        let s = if a[i - 1] == b[j - 1] { 2 } else { -1 };
        if score[i][j] == score[i - 1][j - 1] + s {
            pairs.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if score[i][j] == score[i - 1][j] - 1 {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    pairs.reverse();
    pairs
}

/// Pair residues of two traces the way a reader would: by identity when the files share a
/// numbering, by residue number when each is a single chain with a different chain id (a
/// predicted model against a deposited entry), and by sequence alignment when the numbering
/// differs altogether. The pairing that matches the most identical amino acids wins; ties go
/// to the more literal method.
pub fn pair_best(
    target_ids: &[ResidueId],
    target_names: &[String],
    reference_ids: &[ResidueId],
    reference_names: &[String],
) -> (Vec<(usize, usize)>, &'static str) {
    let matching = |pairs: &[(usize, usize)]| {
        pairs
            .iter()
            .filter(|&&(i, j)| target_names[i] == reference_names[j])
            .count()
    };
    let mut best = (pair_residues(target_ids, reference_ids), "residue id");
    let single_chain = |ids: &[ResidueId]| ids.windows(2).all(|w| w[0].0 == w[1].0);
    if single_chain(target_ids) && single_chain(reference_ids) {
        let strip = |ids: &[ResidueId]| -> Vec<ResidueId> {
            ids.iter()
                .map(|(_, n, ic)| (String::new(), *n, ic.clone()))
                .collect()
        };
        let by_number = pair_residues(&strip(target_ids), &strip(reference_ids));
        if matching(&by_number) > matching(&best.0) {
            best = (by_number, "residue number");
        }
    }
    // The alignment table is n×m; beyond ~4000×4000 residues it is not worth the memory.
    if target_names.len() * reference_names.len() <= 16_000_000 {
        let aligned = align_by_sequence(target_names, reference_names);
        if matching(&aligned) > matching(&best.0) {
            best = (aligned, "sequence alignment");
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(chain: &str, nums: &[isize]) -> Vec<ResidueId> {
        nums.iter().map(|&n| (chain.to_string(), n, None)).collect()
    }

    fn names(s: &[&str]) -> Vec<String> {
        s.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn identical_numbering_pairs_by_residue_id() {
        let a = ids("A", &[1, 2, 3, 4]);
        let n = names(&["ALA", "GLY", "ASP", "LYS"]);
        let (pairs, how) = pair_best(&a, &n, &a, &n);
        assert_eq!(pairs, vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
        assert_eq!(how, "residue id");
    }

    #[test]
    fn point_mutant_with_different_chain_id_pairs_by_number() {
        let parent = ids("A", &[10, 11, 12, 13]);
        let model = ids("B", &[10, 11, 12, 13]);
        let pn = names(&["ALA", "GLY", "ASP", "LYS"]);
        let mn = names(&["ALA", "GLY", "ALA", "LYS"]);
        let (pairs, how) = pair_best(&model, &mn, &parent, &pn);
        assert_eq!(pairs.len(), 4);
        assert_eq!(how, "residue number");
    }

    #[test]
    fn renumbered_model_pairs_by_alignment_and_keeps_the_mutation_aligned() {
        let parent = ids("A", &[101, 102, 103, 104, 105, 106]);
        let model = ids("A", &[1, 2, 3, 4, 5, 6]);
        let pn = names(&["MET", "LYS", "ASP", "TRP", "GLU", "LEU"]);
        let mn = names(&["MET", "LYS", "ALA", "TRP", "GLU", "LEU"]);
        let (pairs, how) = pair_best(&model, &mn, &parent, &pn);
        assert_eq!(how, "sequence alignment");
        assert!(
            pairs.contains(&(2, 2)),
            "the mutated residue stays paired: {pairs:?}"
        );
    }
}
