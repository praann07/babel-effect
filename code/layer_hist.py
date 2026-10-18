"""Distribution of the critical MLP layer (argmax of the last-subject-token MLP profile) per model and language (CPU).

Exploratory (not pre-registered): written after the Gemma median profile showed a second peak near layer 14.
Writes analysis/run_<date>_<tag>/{layer_hist.json,fig4_layer_hist.png}; the figure also goes to paper/figures/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from circuit_overlap import is_valid, load

LANGS = ("en", "es", "hi")
COL = {"en": "#1f77b4", "es": "#d95f02", "hi": "#2ca02c"}
RUNS = {"Llama-3.2-3B": "analysis/run_20261015_p3trace_v2_llama", "Gemma-2-2B": "analysis/run_20261015_p3trace_v2c_gemma2"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="layer_hist")
    a = ap.parse_args()
    out = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}"
    out.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    res: dict = {}
    for ax, (name, run) in zip(axes, RUNS.items()):
        arr, recs = load(Path(run))
        ids = sorted({f for f, _ in recs if all(is_valid(recs[(f, lg)]) for lg in LANGS)})
        L = len(arr[(ids[0], "en")]["mlp"])
        res[name] = {"n": len(ids), "n_layers": L}
        for j, lg in enumerate(LANGS):
            crit = np.array([int(np.argmax(arr[(f, lg)]["mlp"])) for f in ids])
            hist = np.bincount(crit, minlength=L)
            second = np.array([int(np.argsort(-arr[(f, lg)]["mlp"])[1]) for f in ids])
            res[name][lg] = {"hist": hist.tolist(), "median": float(np.median(crit)), "iqr": [float(np.percentile(crit, 25)), float(np.percentile(crit, 75))],
                             "frac_layer_8_19": float(np.mean((crit >= 8) & (crit <= 19))),
                             "frac_top2_in_8_19": float(np.mean([(c >= 8 and c <= 19) or (s >= 8 and s <= 19) for c, s in zip(crit, second)]))}
            ax.bar(np.arange(L) + (j - 1) * 0.27, hist, 0.27, color=COL[lg], label=lg.upper())
        ax.set_title(f"{name} (n={len(ids)})", fontsize=10)
        ax.set_xlabel("critical MLP layer (argmax)")
    axes[0].set_ylabel("number of facts")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig4_layer_hist.png", dpi=200, bbox_inches="tight")
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("paper/figures/fig4_layer_hist.png", dpi=200, bbox_inches="tight")
    fig.savefig("paper/figures/fig4_layer_hist.pdf", bbox_inches="tight")
    (out / "layer_hist.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    for m, v in res.items():
        print(m, {lg: (v[lg]["median"], round(v[lg]["frac_layer_8_19"], 2)) for lg in LANGS})


if __name__ == "__main__":
    main()
