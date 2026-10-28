"""Shared helpers: local model loading, subject token spans, fact selection."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

# key -> TransformerLens config name, local weight dir, facts file (facts are filtered per model)
MODELS = {
    "llama": {"tl_name": "meta-llama/Llama-3.2-3B-Instruct", "dir": "models/llama-3.2-3b-instruct", "facts": "data/facts.json", "facts_v2": "data/v2/facts_llama.json"},
    "llama1b": {"tl_name": "meta-llama/Llama-3.2-1B-Instruct", "dir": "models/llama-3.2-1b-instruct", "facts": "data/facts_llama1b.json", "facts_v2": "data/v2/facts_llama1b.json"},
    "llama_base": {"tl_name": "meta-llama/Llama-3.2-3B", "dir": "models/llama-3.2-3b-base", "facts": "data/facts_llama_base.json", "facts_v2": "data/v2/facts_llama_base.json"},
    "gemma2": {"tl_name": "google/gemma-2-2b", "dir": "models/gemma-2-2b", "facts": "data/facts_gemma2.json", "facts_v2": "data/v2/facts_gemma2.json"},
    "bloom1b7": {"tl_name": "bigscience/bloom-1b7", "dir": "models/bloom-1b7", "facts": "data/facts_bloom1b7.json", "facts_v2": "data/v2/facts_bloom1b7.json"},
}
MODEL_NAME = MODELS["llama"]["tl_name"]  # kept for older scripts
MODEL_DIR = MODELS["llama"]["dir"]
LANGS = ("en", "es", "hi")


def load_model(key: str = "llama", dtype: torch.dtype = torch.bfloat16, device: str = "cuda") -> HookedTransformer:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = MODELS[key]
    hf = AutoModelForCausalLM.from_pretrained(spec["dir"], torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(spec["dir"])
    model = HookedTransformer.from_pretrained_no_processing(spec["tl_name"], hf_model=hf, tokenizer=tok, dtype=dtype,
                                                            device=device)
    del hf
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def share_transposed_mlp_weights(model: HookedTransformer) -> None:
    """Re-lay out MLP and attention weights so TransformerLens' per-forward `.T.contiguous()` / `rearrange` are views.

    Each weight keeps its shape and values but is stored in the layout the forward pass wants, so no per-forward copy is
    made. Without this every forward allocates fresh copies of the weights and autograd keeps them all alive (about
    1.2 GiB per pass), which does not fit in 8 GB when optimising through the network (P4).
    """
    for blk in model.blocks:
        for name in ("W_gate", "W_in", "W_out"):
            if hasattr(blk.mlp, name):
                p = getattr(blk.mlp, name)
                p.data = p.data.T.contiguous().T
        at = blk.attn
        kv = ("_W_K", "_W_V") if hasattr(at, "_W_K") else ("W_K", "W_V")  # grouped-query vs full attention
        for name in ("W_Q", *kv):  # [h, d_model, d_head] -> stored as [h, d_head, d_model]
            p = getattr(at, name)
            p.data = p.data.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        at.W_O.data = at.W_O.data.permute(2, 0, 1).contiguous().permute(1, 2, 0)  # stored as [d_model, h, d_head]
        torch.cuda.empty_cache()


def subject_span(tok, prompt: str, subject: str) -> tuple[list[int], list[int]]:
    """Return (token ids incl. BOS, positions of the subject tokens). Case-insensitive character match."""
    enc = tok(prompt, return_offsets_mapping=True)
    i = prompt.lower().find(subject.lower())
    if i < 0:
        raise ValueError(f"subject {subject!r} not in prompt {prompt!r}")
    j = i + len(subject)
    pos = [k for k, (s, e) in enumerate(enc["offset_mapping"]) if e > s and e > i and s < j]
    if not pos:
        raise ValueError(f"no tokens overlap subject {subject!r} in {prompt!r}")
    return enc["input_ids"], pos


def load_facts(path: str = "data/facts.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf8"))["facts"]


def select_p2_facts(facts: list[dict]) -> list[dict]:
    """Deterministic 10-fact P2 subset: first 4 capital_country, 4 capital_of, 2 landmark_city by id."""
    want = {"capital_country": 4, "capital_of": 4, "landmark_city": 2}
    out: list[dict] = []
    for rel, n in want.items():
        out += sorted((f for f in facts if f["relation"] == rel), key=lambda f: f["id"])[:n]
    return out
