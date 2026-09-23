//! Numerical parity with Hugging Face `transformers.EsmForMaskedLM` (fp32, CPU).
//!
//! Observed (8M and 35M, CPU): logits within 3.7e-5 abs on the three short proteins and 4.6e-5
//! at 1022 residues (magnitudes up to ~20); amino-acid log-probabilities within 1.3e-5 and
//! 3.3e-5. That is fp32 accumulation-order noise (`transformers` itself differs from its own
//! fp64 run by ~2e-5). Tolerances: 2e-4 / 1e-4.
//!
//! Until 0.6.0 the differences were 7e-3 / 2.5e-3 on the short proteins and 0.1 in
//! log-probability at 1022 residues. That was not noise: the rotary `inv_freq` was recomputed
//! exactly instead of read from the checkpoint, which stores it rounded to fp16. The long
//! fixture exists because the error grows with position and the short ones understated it.
//!
//! Reference values: `tests/data/<model>.json` and `tests/data/<model>_long.json`, produced by
//! `validate/esm_reference.py` (`--long` for the latter). Needs the checkpoints (downloaded from
//! the Hub into the cache on first run), so the tests are `#[ignore]`d:
//! `cargo test -p proteus-esm -- --ignored`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use proteus_esm::{Device, Esm2, Tokenizer, MASK_ID};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    model: String,
    sequences: HashMap<String, SeqRef>,
}

#[derive(Deserialize)]
struct SeqRef {
    sequence: String,
    ids: Vec<u32>,
    logits: Vec<Vec<f32>>,
    masked: HashMap<String, Vec<f32>>,
}

const LOGITS_TOL: f32 = 2e-4;
const LOGP_TOL: f32 = 1e-4;

fn reference(name: &str) -> Reference {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(format!("{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn check_model(name: &str, logits_tol: f32, logp_tol: f32) {
    let r = reference(name);
    let model = Esm2::from_hub(&r.model, &Device::Cpu).expect("download/load model");
    for (id, s) in &r.sequences {
        let (tokens, unk) = Tokenizer::encode(&s.sequence).unwrap();
        assert_eq!(unk, 0);
        assert_eq!(tokens, s.ids, "{id}: tokenizer disagrees with HF");

        let logits = model.logits(&tokens).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(logits.len(), s.logits.len());
        let mut worst = 0f32;
        for (ours, theirs) in logits.iter().zip(&s.logits) {
            worst = worst.max(max_abs_diff(ours, theirs));
        }
        eprintln!(
            "{name} {id}: max |Δlogit| = {worst:.2e} over {}×33",
            logits.len()
        );
        assert!(worst < logits_tol, "{name} {id}: logits differ by {worst}");

        for (pos, want) in &s.masked {
            let pos: usize = pos.parse().unwrap();
            let mut masked = tokens.clone();
            masked[pos + 1] = MASK_ID;
            let lp = model.log_probs(&masked).unwrap();
            let d_all = max_abs_diff(&lp[pos + 1], want);
            // What scoring uses: the twenty canonical amino-acid tokens (ids 4..=23). Special
            // tokens sit at log-prob ≈ −30, where fp32 rounding is larger in absolute terms.
            let d_aa = max_abs_diff(&lp[pos + 1][4..=23], &want[4..=23]);
            eprintln!(
                "{name} {id}: masked pos {pos}: max |Δlogp| = {d_aa:.2e} (amino acids), {d_all:.2e} (all 33)"
            );
            if d_aa > logp_tol / 2.0 {
                for k in 4..=23 {
                    let d = (lp[pos + 1][k] - want[k]).abs();
                    if d > logp_tol / 10.0 {
                        eprintln!(
                            "    token {k}: ours {:.5} theirs {:.5} Δ {d:.2e}",
                            lp[pos + 1][k],
                            want[k]
                        );
                    }
                }
            }
            assert!(
                d_aa < logp_tol,
                "{name} {id} masked {pos}: amino-acid log-probs differ by {d_aa}"
            );
        }
    }
}

#[derive(Deserialize)]
struct LongReference {
    model: String,
    sequence: String,
    ids: Vec<u32>,
    /// Unmasked logits at a subset of token positions (every 16th, plus the last two).
    logits: HashMap<String, Vec<f32>>,
    /// Log-probability row at each masked residue index (0-based).
    masked: HashMap<String, Vec<f32>>,
}

fn log_softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + row.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
    row.iter().map(|x| x - lse).collect()
}

/// Parity at 1022 residues, the ESM-2 training length. Rotary position errors grow with the
/// position index, so a short-sequence fixture cannot see them: computing `inv_freq` exactly
/// instead of using the checkpoint's fp16-rounded buffer passed the 78-residue test and was off
/// by 0.1 in log-probability here.
fn check_long(name: &str, logits_tol: f32, logp_tol: f32) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(format!("{name}_long.json"));
    let r: LongReference = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(r.sequence.len(), 1022);
    let model = Esm2::from_hub(&r.model, &Device::Cpu).expect("download/load model");
    let (tokens, unk) = Tokenizer::encode(&r.sequence).unwrap();
    assert_eq!(unk, 0);
    assert_eq!(tokens, r.ids, "tokenizer disagrees with HF");

    let logits = model.logits(&tokens).unwrap().to_vec2::<f32>().unwrap();
    let (mut d_logit, mut d_lp) = (0f32, 0f32);
    for (pos, want) in &r.logits {
        let ours = &logits[pos.parse::<usize>().unwrap()];
        d_logit = d_logit.max(max_abs_diff(ours, want));
        d_lp = d_lp.max(max_abs_diff(
            &log_softmax(ours)[4..=23],
            &log_softmax(want)[4..=23],
        ));
    }
    eprintln!(
        "{name} L=1022: max |Δlogit| = {d_logit:.2e}, max |Δlogp| = {d_lp:.2e} (amino acids) \
         over {} rows",
        r.logits.len()
    );
    assert!(
        d_logit < logits_tol,
        "{name} L=1022: logits differ by {d_logit}"
    );
    assert!(d_lp < logp_tol, "{name} L=1022: log-probs differ by {d_lp}");

    for (pos, want) in &r.masked {
        let pos: usize = pos.parse().unwrap();
        let mut masked = tokens.clone();
        masked[pos + 1] = MASK_ID;
        let lp = model.log_probs(&masked).unwrap();
        let d_aa = max_abs_diff(&lp[pos + 1][4..=23], &want[4..=23]);
        eprintln!("{name} L=1022: masked pos {pos}: max |Δlogp| = {d_aa:.2e} (amino acids)");
        assert!(
            d_aa < logp_tol,
            "{name} L=1022 masked {pos}: amino-acid log-probs differ by {d_aa}"
        );
    }
}

#[test]
#[ignore]
fn esm2_t6_8m_matches_transformers() {
    check_model("esm2_t6_8M_UR50D", LOGITS_TOL, LOGP_TOL);
}

#[test]
#[ignore]
fn esm2_t12_35m_matches_transformers() {
    check_model("esm2_t12_35M_UR50D", LOGITS_TOL, LOGP_TOL);
}

// Named so that CI's `esm2_t6_8m_matches_transformers` filter selects it as well.
#[test]
#[ignore]
fn esm2_t6_8m_matches_transformers_at_training_length() {
    check_long("esm2_t6_8M_UR50D", LOGITS_TOL, LOGP_TOL);
}

#[test]
#[ignore]
fn esm2_t12_35m_matches_transformers_at_training_length() {
    check_long("esm2_t12_35M_UR50D", LOGITS_TOL, LOGP_TOL);
}

/// Scoring a whole library must not re-run the wild-type forward pass per variant (issue #3).
#[test]
#[ignore]
fn library_scoring_reuses_forward_passes() {
    use proteus_esm::{parse_mutation, Device, Esm2, MarginalScorer};
    let model = Esm2::from_hub("facebook/esm2_t6_8M_UR50D", &Device::Cpu).unwrap();
    let wt = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN";
    let variants: Vec<Vec<_>> = ["T1A", "T2A", "C3A", "C4A", "P5A", "T1A,C3A"]
        .iter()
        .map(|m| m.split(',').map(|x| parse_mutation(x).unwrap()).collect())
        .collect();

    let mut wt_scorer = MarginalScorer::new(&model, wt, false).unwrap();
    let before = model.forward_calls();
    let scores: Vec<Vec<f32>> = variants
        .iter()
        .map(|v| wt_scorer.score(v).unwrap())
        .collect();
    assert_eq!(
        model.forward_calls() - before,
        1,
        "wild-type marginals need one pass"
    );
    // Same numbers as the one-shot API.
    for (v, s) in variants.iter().zip(&scores) {
        assert_eq!(&proteus_esm::score_wt_marginal(&model, wt, v).unwrap(), s);
    }

    let mut masked = MarginalScorer::new(&model, wt, true).unwrap();
    let before = model.forward_calls();
    for v in &variants {
        masked.score(v).unwrap();
    }
    // Five distinct positions across six variants: five passes, not six.
    assert_eq!(
        model.forward_calls() - before,
        5,
        "masked marginals: one pass per position"
    );
}
