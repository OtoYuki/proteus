# proteus-esm

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

Numerical parity with `transformers.EsmForMaskedLM` (fp32) is pinned by `tests/parity.rs`
against reference logits committed under `validate/reference/esm/*.json`: logits within 1e-2,
amino-acid log-probabilities within 5e-3 (observed ≤ 2.5e-3) on three proteins for the 8M and 35M
checkpoints. CPU only by default; enable `candle-core/cuda` or `candle-core/metal` downstream.
