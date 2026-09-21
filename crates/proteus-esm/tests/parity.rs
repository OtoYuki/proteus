//! Numerical parity with Hugging Face `transformers.EsmForMaskedLM` (fp32, CPU).
//!
//! Observed at commit time: logits within 7e-3 abs (magnitudes up to ~20), amino-acid
//! log-probabilities within 2.5e-3 (relative 1e-3) — fp32 accumulation-order noise, three
//! orders of magnitude below any mutation-effect signal. Tolerances: 1e-2 / 5e-3.
//!
//! Reference values: `validate/reference/esm_<model>.json`, produced by
//! `validate/esm_reference.py`. Needs the checkpoints (downloaded from the Hub into the cache
//! on first run), so the tests are `#[ignore]`d: `cargo test -p proteus-esm -- --ignored`.

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

fn reference(name: &str) -> Reference {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../validate/reference/esm")
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
            // tokens sit at log-prob ≈ −30 where fp32 accumulation noise is larger.
            let d_aa = max_abs_diff(&lp[pos + 1][4..=23], &want[4..=23]);
            eprintln!(
                "{name} {id}: masked pos {pos}: max |Δlogp| = {d_aa:.2e} (amino acids), {d_all:.2e} (all 33)"
            );
            if d_aa > 1e-3 {
                for k in 4..=23 {
                    let d = (lp[pos + 1][k] - want[k]).abs();
                    if d > 5e-4 {
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

#[test]
#[ignore]
fn esm2_t6_8m_matches_transformers() {
    check_model("esm2_t6_8M_UR50D", 1e-2, 5e-3);
}

#[test]
#[ignore]
fn esm2_t12_35m_matches_transformers() {
    check_model("esm2_t12_35M_UR50D", 1e-2, 5e-3);
}
