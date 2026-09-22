# proteus-esm

```toml
[dependencies]
proteus-esm = { git = "https://github.com/OtoYuki/proteus" }
```

ESM-2 protein language model inference in pure Rust on [candle](https://github.com/huggingface/candle):
masked-LM log-probabilities, zero-shot mutation scores (wild-type and masked marginals, Meier et al.
2021) and full 20×L deep-mutational-scan matrices. Loads any `facebook/esm2_*` safetensors
checkpoint from the Hub (tiny built-in fetcher, no `hf-hub`) or from local files.

```rust
use proteus_esm::{Device, Esm2, parse_mutation, score_masked_marginal};
let model = Esm2::from_hub("facebook/esm2_t6_8M_UR50D", &Device::Cpu)?;
let wt = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN";
let muts = [parse_mutation("P19A")?, parse_mutation("C4S")?];
let scores = score_masked_marginal(&model, wt, &muts)?;   // log p(mt) − log p(wt) at the masked position
```

## How this compares

Rust ESM inference is not new territory — [`esm-rs`](https://github.com/tcztzy/esm-rs) runs ESM
and ESM++ on candle with CUDA/MLX backends and compares numerically against PyTorch, and
[`plm-local`](https://github.com/zachcp/plm-local) runs protein language models locally. If you
want **embeddings**, look at those first; `esm-rs` in particular covers more backends than this
crate does.

This crate is aimed one step further down the pipeline: **variant effect**, not representation.

- **Mutation scoring**, wild-type and masked marginals (Meier et al. 2021), and full 20×L deep
  mutational scans — not embedding extraction.
- **`MarginalScorer` caches the wild-type forward pass**, so scoring a library is one pass
  rather than one per variant (875 variants in 0.6 s on CPU).
- **Parity pinned in CI** against `transformers.EsmForMaskedLM` on committed reference logits,
  and accuracy reported on real data (ProteinGym v1.1 Spearman ρ, mean |ρ| 0.42 with the 35M
  checkpoint) rather than only on synthetic checks.
- **No `hf-hub` dependency** — a ~40-line fetcher, so the dependency tree stays small and the
  MSRV stays put.

Zero-shot scores from any protein language model are a triage signal, not a measurement. ESM-2's
published weak spots are **viral proteins** and **long multi-domain sequences**; it is a
reasonable signal for human and microbial ones. ESM-2 rather than ESM-3 because ESM-3's weights
are licensed for non-commercial use only.

Numerical parity with `transformers.EsmForMaskedLM` (fp32) is pinned by `tests/parity.rs`
against reference logits committed under `tests/data/*.json` (from `validate/esm_reference.py`): logits within 1e-2,
amino-acid log-probabilities within 5e-3 (observed ≤ 2.5e-3) on three proteins for the 8M and 35M
checkpoints. CPU only by default; enable `candle-core/cuda` or `candle-core/metal` downstream.
