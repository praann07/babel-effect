"""P0 smoke test: load a model via TransformerLens (ladder fallback) and print top-5 next-token logits."""
from __future__ import annotations

import argparse
import gc
import json

import torch
from transformer_lens import HookedTransformer

LADDER = ["meta-llama/Llama-3.2-3B-Instruct", "google/gemma-2-2b", "EleutherAI/pythia-1.4b", "gpt2"]
PROMPT = "The Eiffel Tower is located in the city of"


def load(name: str, dtype: torch.dtype, local: str | None = None) -> HookedTransformer:
    if local is None:
        return HookedTransformer.from_pretrained(name, dtype=dtype, device="cuda")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = AutoModelForCausalLM.from_pretrained(local, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(local)
    return HookedTransformer.from_pretrained_no_processing(name, hf_model=hf, tokenizer=tok, dtype=dtype, device="cuda")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0, help="index into LADDER to start from")
    ap.add_argument("--local", default=None, help="local HF dir holding the weights of LADDER[start]")
    ap.add_argument("--out", default="analysis/p0_smoke.json")
    a = ap.parse_args()
    for name in LADDER[a.start:]:
        try:
            torch.cuda.reset_peak_memory_stats()
            model = load(name, torch.bfloat16, a.local)
            break
        except Exception as e:  # gated, OOM, unsupported
            print(f"SKIP {name}: {type(e).__name__}: {str(e)[:120]}")
            gc.collect(); torch.cuda.empty_cache()
    with torch.no_grad():
        toks = model.to_tokens(PROMPT)
        logits = model(toks)[0, -1].float()
    top = torch.topk(logits.softmax(-1), 5)
    out = {"model": name, "dtype": "bfloat16", "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "top5": [(model.tokenizer.decode(int(i)), round(float(p), 4)) for p, i in zip(top.values, top.indices)]}
    print(json.dumps(out, indent=1))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
