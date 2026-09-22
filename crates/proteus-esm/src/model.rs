//! `EsmForMaskedLM` as in Hugging Face `transformers` (`modeling_esm.py`), fp32.

use std::path::Path;

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, Embedding, LayerNorm, Linear, VarBuilder,
};
use serde::Deserialize;

use crate::tokenizer::{MASK_ID, PAD_ID};
use crate::{EsmError, Result};

/// Fields of `config.json` that the forward pass needs.
#[derive(Debug, Clone, Deserialize)]
pub struct EsmConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_true")]
    pub token_dropout: bool,
    #[serde(default = "default_mask")]
    pub mask_token_id: u32,
    #[serde(default = "default_pad")]
    pub pad_token_id: u32,
    #[serde(default)]
    pub emb_layer_norm_before: bool,
    #[serde(default = "default_rotary")]
    pub position_embedding_type: String,
}

fn default_eps() -> f64 {
    1e-5
}
fn default_max_pos() -> usize {
    1026
}
fn default_true() -> bool {
    true
}
fn default_mask() -> u32 {
    MASK_ID
}
fn default_pad() -> u32 {
    PAD_ID
}
fn default_rotary() -> String {
    "rotary".into()
}

impl EsmConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let cfg: EsmConfig =
            serde_json::from_str(&text).map_err(|e| EsmError::Config(e.to_string()))?;
        if cfg.position_embedding_type != "rotary" {
            return Err(EsmError::Config(format!(
                "only rotary position embeddings are supported (config has '{}')",
                cfg.position_embedding_type
            )));
        }
        if cfg.emb_layer_norm_before {
            return Err(EsmError::Config(
                "emb_layer_norm_before = true (ESM-1b lineage) is not implemented; every ESM-2 \
                 checkpoint has it false"
                    .into(),
            ));
        }
        if !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads) {
            return Err(EsmError::Config(
                "hidden_size must be divisible by heads".into(),
            ));
        }
        Ok(cfg)
    }

    /// Longest residue sequence the model accepts (`<cls>` and `<eos>` take two positions).
    pub fn max_residues(&self) -> usize {
        self.max_position_embeddings.saturating_sub(2)
    }
}

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    ln: LayerNorm,
    n_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &EsmConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            query: linear(h, h, vb.pp("self.query"))?,
            key: linear(h, h, vb.pp("self.key"))?,
            value: linear(h, h, vb.pp("self.value"))?,
            output: linear(h, h, vb.pp("output.dense"))?,
            ln: layer_norm(h, cfg.layer_norm_eps, vb.pp("LayerNorm"))?,
            n_heads: cfg.num_attention_heads,
            head_dim: h / cfg.num_attention_heads,
        })
    }

    /// Pre-LN self-attention with rotary q/k and a residual on the un-normalised input.
    fn forward(&self, hidden: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, l, _) = hidden.dims3()?;
        let x = self.ln.forward(hidden)?;
        let split = |t: Tensor| -> candle_core::Result<Tensor> {
            t.reshape((b, l, self.n_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        // HF scales q by head_dim^-0.5 *before* applying rotary.
        let scale = (self.head_dim as f64).powf(-0.5);
        let q = split((self.query.forward(&x)? * scale)?)?;
        let k = split(self.key.forward(&x)?)?;
        let v = split(self.value.forward(&x)?)?;
        let q = candle_nn::rotary_emb::rope(&q, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k, cos, sin)?;
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((
            b,
            l,
            self.n_heads * self.head_dim,
        ))?;
        Ok((self.output.forward(&ctx)? + hidden)?)
    }
}

struct Layer {
    attention: Attention,
    ln: LayerNorm,
    intermediate: Linear,
    output: Linear,
}

impl Layer {
    fn load(vb: VarBuilder, cfg: &EsmConfig) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(vb.pp("attention"), cfg)?,
            ln: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?,
            intermediate: linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("intermediate.dense"),
            )?,
            output: linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("output.dense"),
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let attn = self.attention.forward(hidden, cos, sin)?;
        let x = self.ln.forward(&attn)?;
        let x = self.intermediate.forward(&x)?.gelu_erf()?;
        Ok((self.output.forward(&x)? + attn)?)
    }
}

/// ESM-2 masked language model.
pub struct Esm2 {
    cfg: EsmConfig,
    device: Device,
    embeddings: Embedding,
    layers: Vec<Layer>,
    final_ln: LayerNorm,
    head_dense: Linear,
    head_ln: LayerNorm,
    decoder: Linear,
    decoder_bias: Tensor,
    inv_freq: Tensor,
    /// Forward passes run so far (for callers that want to prove they batch/cached).
    forward_calls: std::sync::atomic::AtomicUsize,
}

impl Esm2 {
    /// Load from a `config.json` and a `.safetensors` file.
    pub fn from_files(config: &Path, weights: &Path, device: &Device) -> Result<Self> {
        let cfg = EsmConfig::from_file(config)?;
        let data = std::fs::read(weights)?;
        let vb = VarBuilder::from_buffered_safetensors(data, DType::F32, device)?;
        Self::load(cfg, vb, device.clone())
    }

    /// Download (or reuse the local cache for) `facebook/esm2_t6_8M_UR50D`-style Hub ids.
    ///
    /// Files are fetched from `https://huggingface.co/<id>/resolve/main/<file>` into
    /// `$PROTEUS_ESM_CACHE` or `~/.cache/proteus/esm/<owner>--<name>/`. `HF_TOKEN` is sent
    /// when set (gated repositories).
    #[cfg(feature = "hub")]
    pub fn from_hub(model_id: &str, device: &Device) -> Result<Self> {
        let dir = crate::hub::model_dir(model_id)?;
        let config = crate::hub::fetch(model_id, "config.json", &dir)?;
        let weights = crate::hub::fetch(model_id, "model.safetensors", &dir)?;
        Self::from_files(&config, &weights, device)
    }

    fn load(cfg: EsmConfig, vb: VarBuilder, device: Device) -> Result<Self> {
        let h = cfg.hidden_size;
        let esm = vb.pp("esm");
        let embeddings = embedding(cfg.vocab_size, h, esm.pp("embeddings.word_embeddings"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Layer::load(esm.pp(format!("encoder.layer.{i}")), &cfg)?);
        }
        let final_ln = layer_norm(
            h,
            cfg.layer_norm_eps,
            esm.pp("encoder.emb_layer_norm_after"),
        )?;
        let head = vb.pp("lm_head");
        let head_dense = linear(h, h, head.pp("dense"))?;
        let head_ln = layer_norm(h, cfg.layer_norm_eps, head.pp("layer_norm"))?;
        // The decoder is tied to the input embeddings; safetensors exports omit the tied copy.
        let decoder = if head.contains_tensor("decoder.weight") {
            linear_no_bias(h, cfg.vocab_size, head.pp("decoder"))?
        } else {
            Linear::new(embeddings.embeddings().clone(), None)
        };
        let decoder_bias = head.get(cfg.vocab_size, "bias")?;
        let head_dim = h / cfg.num_attention_heads;
        // inv_freq = 1 / 10000^(2i/d); the checkpoint stores it too but it is a pure function of d.
        let inv: Vec<f32> = (0..head_dim / 2)
            .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv, head_dim / 2, &device)?;
        Ok(Self {
            cfg,
            device,
            embeddings,
            layers,
            final_ln,
            head_dense,
            head_ln,
            decoder,
            decoder_bias,
            inv_freq,
            forward_calls: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub fn config(&self) -> &EsmConfig {
        &self.cfg
    }

    /// Number of forward passes this model has run.
    pub fn forward_calls(&self) -> usize {
        self.forward_calls
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Rotary tables for `len` positions: cos/sin of shape `[len, head_dim/2]` (candle `rope`).
    fn rotary(&self, len: usize) -> Result<(Tensor, Tensor)> {
        let t = Tensor::arange(0u32, len as u32, &self.device)?.to_dtype(DType::F32)?;
        let freqs = t.unsqueeze(1)?.matmul(&self.inv_freq.unsqueeze(0)?)?; // [len, d/2]
        Ok((freqs.cos()?, freqs.sin()?))
    }

    /// Logits `[len, vocab]` for one token sequence (no padding; batch size 1).
    pub fn logits(&self, tokens: &[u32]) -> Result<Tensor> {
        self.forward_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let len = tokens.len();
        if len > self.cfg.max_position_embeddings {
            return Err(EsmError::Sequence(format!(
                "{len} tokens exceed max_position_embeddings={}",
                self.cfg.max_position_embeddings
            )));
        }
        let ids = Tensor::from_slice(tokens, (1, len), &self.device)?;
        let mut x = self.embeddings.forward(&ids)?; // [1, len, h]
        if self.cfg.token_dropout {
            // ESM "token dropout": zero the <mask> rows, then rescale every row so that the
            // expected mask density seen in training (0.15 * 0.8) is preserved.
            let n_mask = tokens
                .iter()
                .filter(|&&t| t == self.cfg.mask_token_id)
                .count();
            if n_mask > 0 {
                let keep: Vec<f32> = tokens
                    .iter()
                    .map(|&t| {
                        if t == self.cfg.mask_token_id {
                            0.0
                        } else {
                            1.0
                        }
                    })
                    .collect();
                let keep = Tensor::from_vec(keep, (1, len, 1), &self.device)?;
                x = x.broadcast_mul(&keep)?;
            }
            let observed = n_mask as f64 / len as f64;
            let scale = (1.0 - 0.15 * 0.8) / (1.0 - observed);
            x = (x * scale)?;
        }
        let (cos, sin) = self.rotary(len)?;
        for layer in &self.layers {
            x = layer.forward(&x, &cos, &sin)?;
        }
        let x = self.final_ln.forward(&x)?;
        let x = self.head_dense.forward(&x)?.gelu_erf()?;
        let x = self.head_ln.forward(&x)?;
        let logits = self
            .decoder
            .forward(&x)?
            .broadcast_add(&self.decoder_bias)?;
        Ok(logits.squeeze(0)?)
    }

    /// Per-position log-softmax over the vocabulary, as `Vec<[f32; vocab]>` rows.
    pub fn log_probs(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        let logits = self.logits(tokens)?;
        let lp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        Ok(lp.to_vec2::<f32>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with(extra: &str) -> Result<EsmConfig> {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        std::fs::write(
            &p,
            format!(
                r#"{{"hidden_size":320,"num_hidden_layers":6,"num_attention_heads":20,
                "intermediate_size":1280,"vocab_size":33,"position_embedding_type":"rotary"{extra}}}"#
            ),
        )
        .unwrap();
        EsmConfig::from_file(&p)
    }

    #[test]
    fn esm1b_style_pre_embedding_layer_norm_is_refused() {
        // The forward pass has no such layer; loading such a checkpoint would silently produce
        // wrong logits rather than fail.
        assert!(config_with("").is_ok());
        assert!(config_with(r#","emb_layer_norm_before":false"#).is_ok());
        let err = config_with(r#","emb_layer_norm_before":true"#).unwrap_err();
        assert!(err.to_string().contains("emb_layer_norm_before"), "{err}");
    }
}
