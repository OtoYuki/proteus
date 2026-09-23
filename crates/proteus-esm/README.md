# proteus-esm

```toml
[dependencies]
proteus-esm = "0.8"
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
- **Parity checked against `transformers.EsmForMaskedLM`** on committed reference logits, up to
  the full 1022-residue length (the 8M checkpoint on every push; the 35M test is `#[ignore]`d
  for download size and run by hand),
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
against reference values committed under `tests/data/*.json` (from `validate/esm_reference.py`),
for the 8M and 35M checkpoints, on three short proteins and on a 1022-residue one
(β-galactosidase): logits within 2e-4 and amino-acid log-probabilities within 1e-4. Largest
observed: 3.6e-5 / 1.3e-5 on the short proteins, 4.6e-5 / 3.3e-5 at 1022 residues, which is
the size of `transformers`' own fp32-vs-fp64 difference. Up to 0.6.0 the rotary frequencies
were recomputed instead of read from the checkpoint (which stores them rounded to fp16); that
cost 2.5e-3 in log-probability on short proteins and up to 0.1 at 1022 residues, and was
mistaken for accumulation noise. CPU only by default; enable `candle-core/cuda` or
`candle-core/metal` downstream.

## Input rules

- **Wild type**: whitespace is ignored, case does not matter, and one trailing `*` is dropped.
  The 20 standard amino acids and ESM's own `X`, `B`, `Z`, `U`, `O` tokens are accepted (as in
  `transformers`); any other character (`J`, digits, `-`, `.`, an inner `*`) is an error.
  `Tokenizer::normalize` applies these rules; positions count residues of its result.
- **Length**: at most 1022 residues, the ESM-2 training length (1024 tokens with `<cls>` and
  `<eos>`). Score longer proteins in windows or per domain.
- **Mutations**: `P19A`, both residues among the 20 standard amino acids, the position in plain
  digits. A mutation at an `X`/`B`/`Z`/`U`/`O` wild-type position is refused, and `scan` leaves
  those positions out. `MarginalScorer::score` takes one variant and refuses a position mutated
  twice; `score_wt_marginal`/`score_masked_marginal` score independent substitutions.
