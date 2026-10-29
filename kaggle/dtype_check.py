"""Number-format check for one model in the format given by BABEL_DTYPE (run once in bfloat16 and once in float16).

Records, on 24 fixed factual prompts (EN/ES/HI): the top-1 next token, its log-probability, the gradient of that token's loss with respect
to a zero vector added at the middle layer's MLP output (what the edit optimiser differentiates through), and the wall time.
kaggle/driver.py compares the two files with the rule pre-registered in docs/HYPOTHESIS.md (A25).
Usage: BABEL_DTYPE=float16 python kaggle/dtype_check.py --model llama --out analysis/_logs/dtype_llama_float16.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

PROMPTS = [
    "The capital of France is", "The capital of Japan is", "The capital of Kenya is", "The capital of Peru is",
    "The Eiffel Tower is located in the city of", "The currency of the United Kingdom is the", "Canberra is the capital of",
    "The official language of Brazil is",
    "La capital de Francia es", "La capital de Japón es", "La capital de Kenia es", "La capital de Perú es",
    "La Torre Eiffel está en la ciudad de", "La moneda del Reino Unido es la", "Canberra es la capital de",
    "El idioma oficial de Brasil es el",
    "फ्रांस की राजधानी है", "जापान की राजधानी है", "केन्या की राजधानी है", "पेरू की राजधानी है",
    "एफिल टॉवर इस शहर में है:", "यूनाइटेड किंगडम की मुद्रा है", "कैनबरा इस देश की राजधानी है:", "ब्राज़ील की आधिकारिक भाषा है",
]


def check(model, token_lists: list[torch.Tensor], device: str, passes: int = 3) -> dict:
    layer = model.cfg.n_layers // 2
    rows, grads = [], []
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    sync(); t0 = time.time()
    for rep in range(passes):  # repeated for a stable timing; values are recorded from the first pass
        for toks in token_lists:
            delta = torch.zeros(model.cfg.d_model, device=device, dtype=torch.float32, requires_grad=True)

            def add(x, hook):
                return x + delta.to(x.dtype)

            logits = model.run_with_hooks(toks, fwd_hooks=[(f"blocks.{layer}.hook_mlp_out", add)])[0, -1].float()
            lp = torch.log_softmax(logits, -1)
            top = int(lp.argmax())
            (-lp[top]).backward()
            if rep == 0:
                g = delta.grad.detach().double().cpu()
                rows.append({"top1": top, "logp": float(lp[top].detach()),
                             "finite": bool(torch.isfinite(logits).all()) and bool(torch.isfinite(g).all())})
                grads.append([round(float(v), 9) for v in g])
    sync()
    return {"layer": layer, "seconds_3_passes": round((time.time() - t0) * 3 / passes, 3), "rows": rows, "grads": grads}


def main() -> None:
    from common import DTYPE, load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = time.time()
    model = load_model(a.model)
    load_s = time.time() - t
    out = check(model, [model.to_tokens(p) for p in PROMPTS], "cuda")
    out |= {"model": a.model, "dtype": str(DTYPE), "load_seconds": round(load_s, 1), "prompts": PROMPTS,
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
    Path(a.out).write_text(json.dumps(out), encoding="utf8")
    print(a.model, out["dtype"], "seconds", out["seconds_3_passes"], "all finite", all(r["finite"] for r in out["rows"]))


if __name__ == "__main__":
    main()
