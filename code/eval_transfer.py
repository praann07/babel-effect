"""P5: aggregate edit results into transfer, specificity and damage tables (bootstrap 95% CIs over facts).

Transfer ratio = dP_L2 / dP_EN is computed on facts whose EN edit succeeded (EN top-1 is the new object's first token),
since the ratio is undefined when the EN edit does nothing; the all-edits version is also reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDS = ("traced", "mid", "random")
LANGS = ("en", "es", "hi")
N_BOOT = 2000


def med_ci(x: list[float], rng: np.random.Generator) -> tuple[float, float, float]:
    a = np.asarray(x, dtype=float)
    if not len(a):
        return (float("nan"),) * 3
    b = np.array([np.median(a[rng.integers(0, len(a), len(a))]) for _ in range(N_BOOT)])
    return float(np.median(a)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def fmt(t: tuple[float, float, float], nd: int = 3) -> str:
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--seed", type=int, default=20261012)
    a = ap.parse_args()
    run = Path(a.run)
    rows = json.loads((run / "records.json").read_text(encoding="utf8"))["rows"]
    rng = np.random.default_rng(a.seed)
    res: dict = {}
    for c in CONDS:
        R = [r for r in rows if r["condition"] == c]
        ok = [r for r in R if r["en"]["top1_is_new"]]
        d = {"n_edits": len(R), "layers": sorted({r["layer"] for r in R}), "en_success_rate": len(ok) / len(R) if R else None,
             "n_en_success": len(ok)}
        for lg in LANGS:
            d[f"dP_{lg}"] = med_ci([r[lg]["dP"] for r in R], rng)
            d[f"top1_new_rate_{lg}"] = float(np.mean([r[lg]["top1_is_new"] for r in R])) if R else None
        for lg in ("es", "hi"):
            d[f"transfer_{lg}_en_success"] = med_ci([r[lg]["dP"] / r["en"]["dP"] for r in ok if r["en"]["dP"] > 0], rng)
            d[f"transfer_{lg}_all"] = med_ci([r[lg]["dP"] / r["en"]["dP"] for r in R if r["en"]["dP"] > 0], rng)
        d["specificity_max_abs_dP"] = med_ci([r["specificity_max_abs_dP"] for r in R], rng)
        d["specificity_worst"] = max(r["specificity_max_abs_dP"] for r in R) if R else None
        d["ppl_ratio"] = med_ci([r["ppl_ratio"] for r in R], rng)
        d["ppl_ratio_worst"] = max(r["ppl_ratio"] for r in R) if R else None
        res[c] = d
    (run / "tables.json").write_text(json.dumps(res, indent=1))

    L = ["| condition | n | layers | EN success | dP EN | dP ES | dP HI | ES top-1 new | HI top-1 new |", "|---|---|---|---|---|---|---|---|---|"]
    for c, d in res.items():
        L.append(f"| {c} | {d['n_edits']} | {','.join(map(str, d['layers']))} | {d['en_success_rate']:.0%} ({d['n_en_success']}/{d['n_edits']}) | "
                 f"{fmt(d['dP_en'])} | {fmt(d['dP_es'])} | {fmt(d['dP_hi'])} | {d['top1_new_rate_es']:.0%} | {d['top1_new_rate_hi']:.0%} |")
    T = ["| condition | transfer ES (EN-success edits) | transfer HI (EN-success edits) | transfer ES (all) | transfer HI (all) |", "|---|---|---|---|---|"]
    for c, d in res.items():
        T.append(f"| {c} | {fmt(d['transfer_es_en_success'])} | {fmt(d['transfer_hi_en_success'])} | {fmt(d['transfer_es_all'])} | {fmt(d['transfer_hi_all'])} |")
    S = ["| condition | specificity: max abs dP over 20 controls, median [CI] | worst | damage: ppl ratio median [CI] | worst |", "|---|---|---|---|---|"]
    for c, d in res.items():
        S.append(f"| {c} | {fmt(d['specificity_max_abs_dP'])} | {d['specificity_worst']:.3f} | {fmt(d['ppl_ratio'], 4)} | {d['ppl_ratio_worst']:.4f} |")
    md = "## Edit success and dP\n" + "\n".join(L) + "\n\n## Transfer ratio\n" + "\n".join(T) + "\n\n## Specificity and damage\n" + "\n".join(S) + "\n"
    (run / "tables.md").write_text(md)
    print(md)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True, sharey=True)
    colors = {"en": "#0072B2", "es": "#E69F00", "hi": "#009E73"}
    for ax, c in zip(axes, CONDS):
        R = [r for r in rows if r["condition"] == c]
        for i, lg in enumerate(LANGS):
            y = [r[lg]["dP"] for r in R]
            ax.scatter(np.full(len(y), i) + rng.uniform(-0.12, 0.12, len(y)), y, color=colors[lg], alpha=0.8)
            ax.hlines(np.median(y), i - 0.25, i + 0.25, color="black")
        ax.set_xticks(range(3), LANGS); ax.set_title(f"{c} layer edit (n={len(R)})"); ax.axhline(0, color="grey", lw=0.5)
    axes[0].set_ylabel("dP(new object first token)")
    fig.savefig(run / "dP_by_language.png", dpi=130)


if __name__ == "__main__":
    main()
