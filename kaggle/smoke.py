"""First GPU job of every Kaggle step: load the model through the repo's own loader and check it answers a known fact. Everything else
depends on this job, so a wrong model, wrong tokenizer or broken GPU stops the run in minutes instead of hours."""
from __future__ import annotations

import argparse
import sys
import time

import torch

from common import MODELS, load_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--expect", default=" Paris")
    a = ap.parse_args()
    t = time.time()
    m = load_model(a.model)
    logits = m(m.to_tokens(a.prompt))[0, -1].float()
    top = m.to_string(logits.argmax().unsqueeze(0))
    print(f"{MODELS[a.model]['tl_name']}: loaded in {time.time() - t:.0f}s, dtype {next(m.parameters()).dtype}, "
          f"vram {torch.cuda.memory_allocated() / 2**30:.2f} GiB, {a.prompt!r} -> {top!r}")
    if top != a.expect:
        best = logits.softmax(-1).topk(5)
        print(f"WRONG ANSWER: expected {a.expect!r}; logits finite: {bool(torch.isfinite(logits).all())}; top 5: "
              + ", ".join(f"{m.to_string(i.unsqueeze(0))!r} {v:.3f}" for v, i in zip(best.values.tolist(), best.indices)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
