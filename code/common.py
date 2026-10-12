"""Shared helpers: local model loading, subject token spans, fact selection."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # TransformerLens config key; weights come from MODEL_DIR
MODEL_DIR = "models/llama-3.2-3b-instruct"
LANGS = ("en", "es", "hi")


def load_model(local: str = MODEL_DIR, dtype: torch.dtype = torch.bfloat16) -> HookedTransformer:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = AutoModelForCausalLM.from_pretrained(local, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(local)
    model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, hf_model=hf, tokenizer=tok, dtype=dtype,
                                                            device="cuda")
    del hf
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


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
