# SP4 — ESM-2 zero-shot fitness scoring in pure Rust

**Status:** approved under the 2026-09-21 goal · **Scope:** new crate `proteus-esm`, `proteus-cli screen --scorer`, `bench/`, `validate/`
**Depends on:** SP1 (correct structure metrics) — the two signals are complementary.

## 1. Why

Structure-only heuristics (compactness, Ramachandran, burial, network density) say whether a
model *looks* like a folded protein; they say almost nothing about whether a point mutation
keeps *function*. The field's zero-shot standard is the ESM-2 masked-marginal score
(Meier et al. 2021), evaluated on ProteinGym (217 DMS assays; ESM-2 650M ≈ 0.42 Spearman).
No pure-Rust ESM-2 exists (candle-transformers has BERT/ModernBERT, no protein models).
Shipping one — validated against Hugging Face `transformers` to 1e-4 and benchmarked on real
DMS data — is the novel piece of the portfolio and directly useful to the screening funnel.

## 2. Goals

1. `proteus-esm`: load any `facebook/esm2_t{6,12,30,33}_*` checkpoint (safetensors from the
   Hub or a local path), run masked-LM inference on CPU (GPU via candle features later),
   expose per-position log-probabilities and mutation scores.
2. Numerical parity with `transformers.EsmForMaskedLM` (fp32): logits within 1e-3 abs,
   log-probs within 1e-4 on a fixed set of sequences — pinned in tests from committed
   reference JSON produced by `validate/esm_reference.py`.
3. `proteus screen --scorer esm2[:model]`: score every variant against the wildtype with
   wt-marginals (one forward pass per wildtype) or `--esm-masked` (one pass per mutated
   position); the score becomes a column in the leaderboard/Parquet and an optional fitness
   term.
4. `proteus esm score <fasta> --mutations A12G,…` and `proteus esm scan <fasta>` (full
   20×L matrix, CSV/Parquet/TUI heatmap).
5. ProteinGym: `bench/proteingym.py` downloads N substitution assays (small ones by default),
   runs `proteus esm scan`, computes Spearman ρ against experimental fitness, and writes a
   table; the composite structure-only fitness gets the same treatment so the README can say
   exactly what each signal is worth.

Non-goals: training/fine-tuning, ESM-1v ensembles, ESMFold structure prediction, GPU kernels
beyond what candle provides, tokenisation of non-canonical residues beyond ESM's vocab.

## 3. Design

### 3.1 Crate `proteus-esm`
Deps: `candle-core`, `candle-nn`, `safetensors`, `hf-hub` (feature `hub`, default on),
`serde`/`serde_json` (config). No `candle-transformers`.

```rust
pub struct EsmConfig { hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, vocab_size, layer_norm_eps, max_position_embeddings, token_dropout: bool, mask_token_id, pad_token_id }
pub struct Esm2 { … }                       // EsmForMaskedLM
impl Esm2 {
    pub fn from_hub(model_id: &str, device: &Device) -> Result<Self>;  // hf-hub, cached in ~/.cache/huggingface
    pub fn from_files(config: &Path, weights: &Path, device: &Device) -> Result<Self>;
    pub fn logits(&self, tokens: &[u32]) -> Result<Tensor>;            // [L, vocab]
    pub fn log_probs(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>>;  // per position
}
pub struct Tokenizer;  // ESM vocab: <cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B U Z O . - <null_1> <mask>
pub fn score_wt_marginal(model, wt_seq, mutations: &[Mutation]) -> Vec<f32>;   // logp(mut) − logp(wt) at pos, one pass
pub fn score_masked_marginal(model, wt_seq, mutations) -> Vec<f32>;             // mask each position, one pass per distinct position
pub fn scan(model, wt_seq, masked: bool) -> Vec<[f32; 20]>;                      // L × 20 matrix
```
Architecture (HF `modeling_esm.py`): word embeddings (+ ESM token-dropout rescale when the
sequence contains `<mask>`), N pre-LN transformer layers — `attention.LayerNorm` → q/k/v with
bias → rotary on q,k (NeoX "rotate_half", positions 0..L) → softmax(QKᵀ/√d)V →
`attention.output.dense` + residual → `LayerNorm` → `intermediate.dense` → GELU →
`output.dense` + residual — then `emb_layer_norm_after`, and the LM head
`dense → GELU → layer_norm → decoder (tied to embeddings) + bias`. Attention mask handles
padding only (single sequences: none). fp32 on CPU.

### 3.2 CLI
- `proteus esm score|scan|models` subcommands; `--model` default `facebook/esm2_t6_8M_UR50D`
  (8 M params, ~30 MB, fast on CPU); `--device cpu|cuda|metal` (feature-gated).
- `proteus screen --scorer esm2 --esm-model … [--esm-masked]`: variant headers carry
  `[mutation=P19A]` (from `proteus mutate`); the scorer parses them, otherwise aligns variant
  vs wildtype to derive substitutions. Adds `esm2_score` column and `--fitness-weights` knob:
  default composite unchanged; `--fitness esm` ranks by ESM score; `--fitness hybrid` adds
  0.30·sigmoid(esm2_score) after renormalisation.
- TUI: `proteus esm scan --tui` draws the 20×L heat map with the dashboard renderer.

### 3.3 Validation and benchmarks
- `validate/esm_reference.py` (torch CPU): for `esm2_t6_8M` and `esm2_t12_35M`, sequences
  {1CRN, ubiquitin, GB1, a 300-mer from 1TIM}, produce logits at every position for the
  unmasked sequence and for 5 masked positions each; commit `validate/reference/esm/*.json`
  (~2 MB) and pin them in `crates/proteus-esm/tests/parity.rs` (needs the weights: test is
  `#[ignore]`, run by `make validate-esm` and a CI job with the Hub cache).
- `bench/proteingym.py`: downloads the ProteinGym substitution CSVs for a curated list of
  small assays (≤ 200 residues, e.g. GB1 `SPG1_STRSG_Wu_2016`, `BLAT_ECOLX_Stiffler_2015`,
  `PTEN_HUMAN_Mighell_2018`, `TPMT_HUMAN_Matreyek_2018`, `AMIE_PSEAE_Wrenbeck_2017`), runs
  `proteus esm scan`, reports Spearman ρ per assay for wt-marginal and masked-marginal with
  8M/35M/650M, alongside the published ESM-2 numbers; results committed under
  `bench/results/proteingym_*.json` and rendered into `bench/README.md`.
- criterion: forward pass latency per model size on 46/76/300-residue sequences; wt-marginal
  scan throughput (variants/s).

### 3.4 Docs
README: "ESM-2 in pure Rust" section with the parity statement, ProteinGym table and the
one-liner `proteus mutate … | proteus screen - --scorer esm2`. CHANGELOG. `proteus-esm/README.md`
for crates.io.

## 4. Error handling
Hub download failure → clear message with the local-path alternative; unknown residue letters →
`<unk>` with a warning (counted); sequences longer than `max_position_embeddings − 2` (1024) →
error (no chunking in this milestone); mutation strings that disagree with the wildtype residue
→ error naming position and letters.

## 5. Testing
Unit: tokenizer round trip, rotary matches a NumPy reference on a small tensor, GELU/LayerNorm
parity, mutation parsing. Parity test against committed transformers logits. Screen integration
test with the 8M model when weights are cached, skipped otherwise.
