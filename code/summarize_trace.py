"""P2 analysis: per-fact heatmaps, last-subject-token profiles and critical-layer statistics for a trace run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

LANGS = ("en", "es", "hi")
COLORS = {"en": "#0072B2", "es": "#E69F00", "hi": "#009E73"}  # Okabe-Ito, colour-blind safe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    run = Path(a.run)
    out = Path("analysis") / f"figs_{run.name.removeprefix('run_')}"
    out.mkdir(parents=True, exist_ok=True)
    recs = json.loads((run / "records.json").read_text(encoding="utf8"))
    cfg = json.loads((run / "config.json").read_text())
    L = cfg["n_layers"]

    prof = {k: {lg: [] for lg in LANGS} for k in ("mlp", "resid")}
    crit = {lg: [] for lg in LANGS}
    rows = []
    for r in recs:
        z = np.load(run / "grids" / f"{r['id']}_{r['lang']}.npz")
        total = r["score_clean"] - r["score_corr"]
        r["total_effect"] = total
        r["valid"] = bool(r["collapsed"] and r["obj_ok"] and total > 0)
        rows.append(r)
        if r["valid"]:
            for k in ("mlp", "resid"):
                prof[k][r["lang"]].append(z[k][:, r["last_subj"]] / total)
            crit[r["lang"]].append(r["crit_mlp_layer"])

    # per-fact heatmaps (2 rows: residual, MLP; 3 cols: languages)
    for fid in sorted({r["id"] for r in recs}):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for j, lg in enumerate(LANGS):
            z = np.load(run / "grids" / f"{fid}_{lg}.npz")
            r = next(x for x in recs if x["id"] == fid and x["lang"] == lg)
            for i, k in enumerate(("resid", "mlp")):
                ax = axes[i, j]
                sns.heatmap(z[k], ax=ax, cmap="viridis", cbar=True, xticklabels=True, yticklabels=4)
                ax.invert_yaxis()
                ax.axvline(r["last_subj"] + 0.5, color="white", lw=0.8, ls="--")
                ax.set_title(f"{lg} {'residual stream' if k == 'resid' else 'MLP output'}  (subject ends at token {r['last_subj']})",
                             fontsize=9)
                ax.set_xlabel("token position"); ax.set_ylabel("layer")
        fig.suptitle(f"{fid}: indirect effect of restoring one site (logit diff, patched - corrupted)")
        fig.savefig(out / f"heatmap_{fid}.png", dpi=110)
        plt.close(fig)

    # profile at last subject token
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, k, title in zip(axes, ("mlp", "resid"), ("MLP output", "Residual stream")):
        for lg in LANGS:
            P = np.array(prof[k][lg])
            if not len(P):
                continue
            m, se = P.mean(0), P.std(0, ddof=1) / np.sqrt(len(P)) if len(P) > 1 else np.zeros(L)
            ax.plot(range(L), m, color=COLORS[lg], label=f"{lg} (n={len(P)})")
            ax.fill_between(range(L), m - se, m + se, color=COLORS[lg], alpha=0.2)
        ax.set_title(f"{title} at last subject token"); ax.set_xlabel("layer"); ax.set_ylabel("effect / total effect")
        ax.legend()
    fig.savefig(out / "profiles_last_subject.png", dpi=130)
    plt.close(fig)

    summ = {"n_pairs": len(rows), "n_valid": sum(r["valid"] for r in rows),
            "collapsed": sum(r["collapsed"] for r in rows), "negative_total": sum(r["total_effect"] <= 0 for r in rows),
            "obj_not_ok": sum(not r["obj_ok"] for r in rows), "critical_mlp_layer": {}}
    for lg in LANGS:
        c = np.array(crit[lg])
        if len(c):
            summ["critical_mlp_layer"][lg] = {"n": len(c), "median": float(np.median(c)), "q25": float(np.percentile(c, 25)),
                                              "q75": float(np.percentile(c, 75)), "values": c.tolist(),
                                              "frac_in_8_19": float(np.mean((c >= 8) & (c <= 19))),
                                              "frac_le_10": float(np.mean(c <= 10))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
