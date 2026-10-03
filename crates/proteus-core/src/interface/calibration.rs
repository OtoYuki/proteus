//! Which ipSAE_min cutoff to use, per predictor.
//!
//! The cutoff in common use is `ipSAE_min > 0.61`, from Overath et al. 2025's meta-analysis of
//! 3,669 wet-lab-tested designs. That value was fitted on **AlphaFold 3 scores only**. ipSAE is
//! derived from a predictor's own PAE matrix, and PAE is not calibrated across predictors, so the
//! cutoff does not transfer: fitted on one design set, the best threshold ranges from 0.585
//! (ColabFold) to 0.766 (Boltz-2), and — the part that costs a design team real screening budget —
//! **the predictor needing the highest cutoff gives the worst enrichment.**
//!
//! Numbers below are leave-one-target-out fits on the Overath dataset (CC-BY-4.0): the threshold is
//! chosen on every target but one, then applied to the held-out target, which is the same protocol
//! Overath used. `enrichment` is held-out precision divided by that target's binder rate, with a
//! 95 % CI bootstrapped over targets. AlphaFold 3's fitted 0.634 reproduces the published 0.61 from
//! an independent implementation and threshold search, which is the method check on the rest.
//!
//! These are not a substitute for calibrating on your own campaign. Treat them as the right
//! starting point for a predictor, and the evidence that keeping 0.61 for a non-AF3 predictor
//! over-selects.

use std::path::Path;

/// A structure predictor, as identified from the confidence files written beside a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Predictor {
    AlphaFold3,
    /// Boltz-1 and Boltz-2 write identical file names, so they cannot be told apart from the
    /// output alone; [`IpsaeCalibration::note`] carries both fitted values.
    Boltz,
    ColabFold,
    OpenFold3,
    Protenix,
    Chai1,
    /// AlphaFold DB entry (AlphaFold 2).
    AlphaFoldDb,
}

impl Predictor {
    pub fn name(self) -> &'static str {
        match self {
            Self::AlphaFold3 => "AlphaFold 3",
            Self::Boltz => "Boltz",
            Self::ColabFold => "ColabFold",
            Self::OpenFold3 => "OpenFold3",
            Self::Protenix => "Protenix",
            Self::Chai1 => "Chai-1",
            Self::AlphaFoldDb => "AlphaFold DB",
        }
    }
}

/// A fitted ipSAE_min cutoff and what it buys on a target it was not fitted on.
#[derive(Debug, Clone, Copy)]
pub struct IpsaeCalibration {
    pub predictor: Predictor,
    /// Median of the leave-one-target-out fits.
    pub threshold: f64,
    /// Mean precision on the held-out target at that threshold.
    pub held_out_precision: f64,
    /// Held-out precision ÷ the target's binder rate: how much better than picking at random.
    pub enrichment: f64,
    /// 95 % CI on `enrichment`, bootstrapped over targets.
    pub enrichment_ci: (f64, f64),
    /// Targets the fit used. Each needed ≥ 5 binders.
    pub targets: usize,
    pub note: Option<&'static str>,
}

/// The published, AlphaFold 3-derived cutoff (Overath et al. 2025).
pub const PUBLISHED_IPSAE_CUTOFF: f64 = 0.61;

/// The fitted cutoff for a predictor, or `None` when no fit exists for it.
///
/// Protenix and Chai-1 return `None` on purpose: neither was folded in the study behind these
/// numbers, and inventing a cutoff for them would be worse than saying nothing.
pub fn ipsae_calibration(predictor: Predictor) -> Option<IpsaeCalibration> {
    let c = |threshold, held_out_precision, enrichment, enrichment_ci, targets, note| {
        Some(IpsaeCalibration {
            predictor,
            threshold,
            held_out_precision,
            enrichment,
            enrichment_ci,
            targets,
            note,
        })
    };
    match predictor {
        Predictor::AlphaFold3 => c(
            0.634,
            0.480,
            4.21,
            (2.55, 6.12),
            11,
            Some("reproduces the published 0.61; the best-triaging predictor measured"),
        ),
        Predictor::Boltz => c(
            0.766,
            0.325,
            1.71,
            (1.36, 2.16),
            6,
            Some(
                "Boltz-2 fit; Boltz-1 on the same dataset fits 0.711 (enrichment 2.62x). \
                  The two write identical file names, so pick by what you ran",
            ),
        ),
        Predictor::ColabFold => c(0.585, 0.282, 2.11, (1.15, 3.43), 11, None),
        Predictor::OpenFold3 => c(0.683, 0.324, 1.84, (1.37, 2.30), 6, None),
        Predictor::Protenix | Predictor::Chai1 | Predictor::AlphaFoldDb => None,
    }
}

/// Which predictor wrote this model, from the confidence files beside it.
///
/// This reads the same naming conventions as [`crate::pae::sidecar_files`], so a model whose PAE
/// Proteus can parse is a model whose predictor it can name. Returns `None` when nothing beside
/// the file identifies a predictor — a bare PDB, or an experimental structure.
pub fn detect_predictor(structure: &Path) -> Option<Predictor> {
    let dir = structure.parent()?;
    let name = structure.file_name()?.to_str()?;
    let name = name.strip_suffix(".gz").unwrap_or(name);
    let stem = name.rsplit_once('.').map_or(name, |(s, _)| s);
    let has = |p: String| dir.join(p).is_file();

    // Boltz: input_model_0.pdb → pae_input_model_0.npz.
    if has(format!("pae_{stem}.npz")) || has(format!("confidence_{stem}.json")) {
        return Some(Predictor::Boltz);
    }
    // Chai-1: pred.model_idx_0.cif → scores.model_idx_0.npz.
    if let Some(k) = stem.strip_prefix("pred.model_idx_") {
        if has(format!("scores.model_idx_{k}.npz")) {
            return Some(Predictor::Chai1);
        }
    }
    // AlphaFold DB: AF-P69905-F1-model_v6 → AF-P69905-F1-predicted_aligned_error_v6.json.
    if let Some((head, tail)) = stem.rsplit_once("-model_v") {
        if has(format!("{head}-predicted_aligned_error_v{tail}.json")) {
            return Some(Predictor::AlphaFoldDb);
        }
    }
    // ColabFold: X_unrelaxed_rank_001_… → X_scores_rank_001_….json.
    for tag in ["_unrelaxed_", "_relaxed_"] {
        if let Some((head, tail)) = stem.split_once(tag) {
            if has(format!("{head}_scores_{tail}.json")) {
                return Some(Predictor::ColabFold);
            }
        }
    }
    // Protenix: job_sample_0.cif → job_full_data_sample_0.json. Checked before the `_model_`
    // rules because a Protenix stem can also end in a digit.
    if let Some((head, k)) = stem.rsplit_once("_sample_") {
        if !k.is_empty()
            && k.chars().all(|c| c.is_ascii_digit())
            && (has(format!("{head}_full_data_sample_{k}.json"))
                || has(format!("{head}_summary_confidence_sample_{k}.json")))
        {
            return Some(Predictor::Protenix);
        }
    }
    // AlphaFold 3 server: fold_x_model_0 → fold_x_full_data_0.json.
    if let Some((head, k)) = stem.rsplit_once("_model_") {
        if !k.is_empty()
            && k.chars().all(|c| c.is_ascii_digit())
            && (has(format!("{head}_full_data_{k}.json"))
                || has(format!("{head}_summary_confidences_{k}.json")))
        {
            return Some(Predictor::AlphaFold3);
        }
    }
    // OpenFold3 and AlphaFold 3 run locally both end a sample in `_model`. OpenFold3 is the one
    // that writes `_confidences.npz` or `_confidences_aggregated.json`; local AF3 writes
    // `_summary_confidences.json` beside a `_confidences.json`.
    if let Some(head) = stem.strip_suffix("_model") {
        if has(format!("{head}_confidences.npz"))
            || has(format!("{head}_confidences_aggregated.json"))
        {
            return Some(Predictor::OpenFold3);
        }
        if has(format!("{head}_summary_confidences.json"))
            || has(format!("{head}_confidences.json"))
        {
            return Some(Predictor::AlphaFold3);
        }
    }
    if stem == "model" && (dir.join("confidences.json").is_file()) {
        return Some(Predictor::AlphaFold3);
    }
    None
}

/// One line of guidance for a predictor whose scores are in hand, or `None` when the published
/// cutoff is the right one anyway (AlphaFold 3) or no fit exists.
///
/// Returns the sentence to print; the caller decides where it goes.
pub fn cutoff_advice(predictor: Predictor) -> Option<String> {
    let c = ipsae_calibration(predictor)?;
    if predictor == Predictor::AlphaFold3 {
        return None;
    }
    let (lo, hi) = c.enrichment_ci;
    let mut s = format!(
        "{}: use ipSAE_min > {:.3}, not the published {:.2} (fitted on AlphaFold 3). \
         Held-out precision {:.2}, enrichment {:.2}x [{:.2}, {:.2}] over the binder rate, \
         {} targets.",
        c.predictor.name(),
        c.threshold,
        PUBLISHED_IPSAE_CUTOFF,
        c.held_out_precision,
        c.enrichment,
        lo,
        hi,
        c.targets,
    );
    // The Boltz note carries the other version's fit, which a Boltz-1 user needs and cannot get
    // from the file names. Omitting it would quote Boltz-2's cutoff at a Boltz-1 model.
    if let Some(n) = c.note {
        if !matches!(predictor, Predictor::AlphaFold3) {
            s.push(' ');
            s.push_str(n);
            s.push('.');
        }
    }
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn touch(dir: &Path, name: &str) {
        fs::write(dir.join(name), b"{}").unwrap();
    }

    #[test]
    fn boltz_is_detected_from_its_pae_sidecar() {
        let d = tempfile::tempdir().unwrap();
        touch(d.path(), "pae_input_model_0.npz");
        let m = d.path().join("input_model_0.pdb");
        touch(d.path(), "input_model_0.pdb");
        assert_eq!(detect_predictor(&m), Some(Predictor::Boltz));
    }

    #[test]
    fn openfold3_and_local_af3_are_told_apart_by_their_confidence_file() {
        let d = tempfile::tempdir().unwrap();
        touch(d.path(), "q_seed_42_sample_1_confidences.npz");
        let of3 = d.path().join("q_seed_42_sample_1_model.cif");
        touch(d.path(), "q_seed_42_sample_1_model.cif");
        assert_eq!(detect_predictor(&of3), Some(Predictor::OpenFold3));

        let e = tempfile::tempdir().unwrap();
        touch(e.path(), "run_summary_confidences.json");
        touch(e.path(), "run_confidences.json");
        let af3 = e.path().join("run_model.cif");
        touch(e.path(), "run_model.cif");
        assert_eq!(detect_predictor(&af3), Some(Predictor::AlphaFold3));
    }

    #[test]
    fn colabfold_and_protenix_are_detected() {
        let d = tempfile::tempdir().unwrap();
        touch(d.path(), "x_scores_rank_001_alphafold2.json");
        let cf = d.path().join("x_unrelaxed_rank_001_alphafold2.pdb");
        touch(d.path(), "x_unrelaxed_rank_001_alphafold2.pdb");
        assert_eq!(detect_predictor(&cf), Some(Predictor::ColabFold));

        let e = tempfile::tempdir().unwrap();
        touch(e.path(), "job_full_data_sample_0.json");
        let px = e.path().join("job_sample_0.cif");
        touch(e.path(), "job_sample_0.cif");
        assert_eq!(detect_predictor(&px), Some(Predictor::Protenix));
    }

    #[test]
    fn a_bare_structure_names_no_predictor() {
        let d = tempfile::tempdir().unwrap();
        let m = d.path().join("1abc.pdb");
        touch(d.path(), "1abc.pdb");
        assert_eq!(detect_predictor(&m), None);
    }

    #[test]
    fn every_fitted_threshold_is_a_probability_and_enrichment_beats_chance() {
        for p in [
            Predictor::AlphaFold3,
            Predictor::Boltz,
            Predictor::ColabFold,
            Predictor::OpenFold3,
        ] {
            let c = ipsae_calibration(p).expect("fit exists");
            assert!((0.0..=1.0).contains(&c.threshold), "{p:?} threshold");
            assert!(
                (0.0..=1.0).contains(&c.held_out_precision),
                "{p:?} precision"
            );
            // Every arm's CI must clear 1.0, or the row is not worth quoting as a filter.
            assert!(
                c.enrichment_ci.0 > 1.0,
                "{p:?} enrichment CI touches chance"
            );
            assert!(c.enrichment_ci.0 <= c.enrichment, "{p:?} CI low");
            assert!(c.enrichment <= c.enrichment_ci.1, "{p:?} CI high");
        }
    }

    #[test]
    fn predictors_we_did_not_fold_get_no_invented_cutoff() {
        assert!(ipsae_calibration(Predictor::Protenix).is_none());
        assert!(ipsae_calibration(Predictor::Chai1).is_none());
        assert!(cutoff_advice(Predictor::Protenix).is_none());
    }

    #[test]
    fn af3_gets_no_advice_because_the_published_cutoff_is_already_its_own() {
        assert!(cutoff_advice(Predictor::AlphaFold3).is_none());
        let s = cutoff_advice(Predictor::Boltz).unwrap();
        assert!(s.contains("0.766"), "{s}");
        assert!(s.contains("0.61"), "{s}");
        // A Boltz-1 user must not be handed Boltz-2's cutoff with no way to know.
        assert!(s.contains("0.711"), "Boltz-1's own fit is missing: {s}");
    }
}
