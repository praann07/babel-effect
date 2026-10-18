"""A9 'more edit facts' layer rule, applied to a finished layer sweep (CPU).

Among layers whose median specificity loss is below the sweep-wide median of layer medians, pick the layer with the highest EN
success rate (ties: lower median specificity loss, then lower layer index). The follow-up run edits at that layer plus each
fact's own traced layer and L//2 (added by knowledge_edit_v2.py --add-traced). Prints the chosen layer; writes <run>/f7_layer.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf8"))
    rows = [json.loads(l) for l in (run / "records.jsonl").read_text(encoding="utf8").splitlines()]
    per = {}
    for layer in range(cfg["n_layers"]):
        R = [r for r in rows if r["layer"] == layer]
        if R:
            per[layer] = {"n": len(R), "en_success": float(np.mean([r["en"]["success"] for r in R])),
                          "spec": float(np.median([r["specificity_max_abs_dP"] for r in R]))}
    cut = float(np.median([v["spec"] for v in per.values()]))
    elig = [l for l, v in per.items() if v["spec"] < cut]
    best = sorted(elig, key=lambda l: (-per[l]["en_success"], per[l]["spec"], l))[0]
    res = {"run": str(run), "spec_cut": cut, "chosen_layer": best, "chosen": per[best], "eligible_layers": elig}
    (run / "f7_layer.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    print(best)


if __name__ == "__main__":
    main()
