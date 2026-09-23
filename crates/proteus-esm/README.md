# proteus-esm

```toml
[dependencies]
proteus-esm = "0.5"
```

ESM-2 protein language model inference in pure Rust on [candle](https://github.com/huggingface/candle):
masked-LM log-probabilities, zero-shot mutation scores (wild-type and masked marginals, Meier et al.
2021) and full 20×L deep-mutational-scan matrices. Loads `facebook/esm2_t6_8M` through
`esm2_t33_650M` straight from the Hub (a small built-in fetcher, no `hf-hub`), or any ESM-2
checkpoint from local `config.json` + `model.safetensors`. The 3B and 15B repositories on the
Hub publish only sharded PyTorch `.bin` files, so those two need converting to safetensors
first.

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
- **`MarginalScorer` caches forward passes**: wild-type marginals score a whole library from
  one pass; masked marginals need one pass per distinct mutated position, not one per variant.
  A full scan of crambin (46 residues, 874 single mutants) with the 8M checkpoint takes 0.05 s
  with wild-type marginals and 1.1 s with masked marginals, end to end including model load,
  on a laptop CPU (i7-11800H).
- **Parity checked against `transformers.EsmForMaskedLM`** on committed reference logits (the
  8M checkpoint on every push; the 35M test is `#[ignore]`d for download size and run by hand),
  and accuracy reported on real data — ProteinGym v1.1 Spearman ρ, mean |ρ| 0.42 with the 35M
  checkpoint over the five smallest single-mutant assays — rather than only on synthetic checks.
- **No `hf-hub` dependency** — a ~50-line fetcher, so the dependency tree stays small and the
  MSRV stays put.

Zero-shot scores from any protein language model are a triage signal, not a measurement. On
ProteinGym's own per-taxon breakdown ESM-2 650M reaches Spearman ρ ≈ 0.46 on human assays and
≈ 0.26 on viral ones, so treat viral proteins with particular caution. Newer checkpoints exist:
ESM C and the open ESM3 weights are MIT-licensed as of mid-2026, but they are different
architectures and this crate implements ESM-2 only.

Numerical parity with `transformers.EsmForMaskedLM` (fp32) is pinned by `tests/parity.rs`
against reference logits committed under `tests/data/*.json` (from `validate/esm_reference.py`): logits within 1e-2,
amino-acid log-probabilities within 5e-3 (largest observed ≈ 2.5e-3) on three proteins for the
8M and 35M checkpoints. CPU only by default; enable `candle-core/cuda` or `candle-core/metal` downstream.
