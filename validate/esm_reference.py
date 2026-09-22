#!/usr/bin/env python3
"""Reference ESM-2 outputs from Hugging Face transformers (CPU, fp32) for proteus-esm parity tests.

Writes crates/proteus-esm/tests/data/<model>.json (the parity-test fixtures) with, per sequence: token ids, the full unmasked logits
matrix, and the log-probability row at each of five masked positions.

Needs a separate venv (torch is large):  uv venv .esm-venv --python 3.12 &&
uv pip install --python .esm-venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .esm-venv/bin/python transformers safetensors numpy
"""
import json
import pathlib
import sys

import torch
from transformers import EsmForMaskedLM, EsmTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEQS = {
    "1crn": "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN",
    "1ubq": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "1pga": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
}
MASK_POSITIONS = [1, 5, 12, 23, 40]  # residue index (0-based) within the sequence

models = sys.argv[1:] or ["facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D"]
for mid in models:
    tok = EsmTokenizer.from_pretrained(mid)
    model = EsmForMaskedLM.from_pretrained(mid).eval()
    out = {"model": mid, "sequences": {}}
    for name, seq in SEQS.items():
        enc = tok(seq, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        with torch.no_grad():
            logits = model(**enc).logits[0]
        entry = {"sequence": seq, "ids": ids, "logits": logits.tolist(), "masked": {}}
        for p in MASK_POSITIONS:
            if p >= len(seq):
                continue
            m = enc["input_ids"].clone()
            m[0, p + 1] = tok.mask_token_id
            with torch.no_grad():
                lm = model(input_ids=m, attention_mask=enc["attention_mask"]).logits[0]
            entry["masked"][str(p)] = torch.log_softmax(lm[p + 1], -1).tolist()
        out["sequences"][name] = entry
        print(mid, name, "logits", tuple(logits.shape))
    out_dir = ROOT / "crates" / "proteus-esm" / "tests" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{mid.split('/')[-1]}.json"
    dest.write_text(json.dumps(out))
    print("wrote", dest, dest.stat().st_size // 1024, "KB")
