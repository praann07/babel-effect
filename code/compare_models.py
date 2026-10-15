"""Cross-model comparison of the overlap analysis (A5): per-model results side by side, the facts both models keep, and figures.

Reads the v2 trace/overlap runs of both models. The intersection analysis recomputes the shuffled-fact null on the facts that are
valid in both models, separately for each model, so the two sets of statistics are on identical facts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from circuit_overlap import LANGS, PAIRS, analyse, is_valid, load

COL = {"en": "#0072B2", "es": "#E69F00", "hi": "#009E73"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", required=True, help="Llama v2 trace run dir")
    ap.add_argument("--gemma", required=True, help="Gemma v2 trace run dir")
    ap.add_argument("--llama-overlap", required=True, help="overlap results dir for Llama (primary_valid set)")
    ap.add_argument("--gemma-overlap", required=True, help="overlap results dir for Gemma (primary_valid set)")
    ap.add_argument("--tag", default="compare_models")
    ap.add_argument("--seed", type=int, default=20261012)
    a = ap.parse_args()
    import datetime as dt
    out = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}"
    out.mkdir(parents=True, exist_ok=False)
    runs = {"llama": Path(a.llama), "gemma2": Path(a.gemma)}
    data = {k: load(v) for k, v in runs.items()}
    valid = {k: {f for f in {f for f, _ in recs} if all(is_valid(recs[(f, lg)]) for lg in LANGS)} for k, (arr, recs) in data.items()}
    both = sorted(valid["llama"] & valid["gemma2"])
    res = {"n_valid": {k: len(v) for k, v in valid.items()}, "n_both": len(both), "both_ids": both, "intersection": {}}
    rel = {f: data["llama"][1][(f, "en")]["relation"] for f in both}
    res["both_by_relation"] = {r: sum(v == r for v in rel.values()) for r in sorted(set(rel.values()))}
    if len(both) >= 10:
        for k, (arr, recs) in data.items():
            res["intersection"][k] = analyse(arr, both, a.seed)
    (out / "results.json").write_text(json.dumps(res, indent=1))

    lines = ["| model | n | pair | MLP cosine median [95% CI] | null mean | z | p |", "|---|---|---|---|---|---|---|"]
    for k, r in res["intersection"].items():
        for pair, pr in r["pairs"].items():
            s = pr["mlp_cosine"]
            lines.append(f"| {k} | {r['n_facts']} | {pair} | {s['observed_median']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] | "
                         f"{s['null_mean']:.3f} | {s['z']:.2f} | {s['p_one_sided']:.3f} |")
    (out / "intersection_table.md").write_text("\n".join(lines) + "\n")

    # Figure: mean MLP profile at the last subject token, per model and language (effect / total effect)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True, sharey=True)
    for ax, (k, (arr, recs)) in zip(axes, data.items()):
        for lg in LANGS:
            fs = sorted(valid[k])
            M = np.stack([arr[(f, lg)]["mlp"] / (recs[(f, lg)]["score_clean"] - recs[(f, lg)]["score_corr"]) for f in fs])
            m = M.mean(0)
            se = M.std(0, ddof=1) / np.sqrt(len(fs))
            ax.plot(m, color=COL[lg], label=f"{lg} (n={len(fs)})")
            ax.fill_between(range(len(m)), m - se, m + se, color=COL[lg], alpha=0.2)
        ax.set_title(k); ax.set_xlabel("layer"); ax.legend()
    axes[0].set_ylabel("MLP effect / total effect (last subject token)")
    fig.savefig(out / "mean_profiles_by_model.png", dpi=130)

    # Figure: z-scores of the primary statistic per pair and model (pooled null, each model's own valid set)
    zs = {}
    for k, d in (("llama", a.llama_overlap), ("gemma2", a.gemma_overlap)):
        r = json.loads((Path(d) / "results.json").read_text(encoding="utf8"))
        zs[k] = [r["sets"]["primary_valid"]["pairs"][f"{p[0]}-{p[1]}"]["mlp_cosine"]["z"] for p in PAIRS]
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    w = 0.38
    for i, (k, z) in enumerate(zs.items()):
        ax.bar(np.arange(len(PAIRS)) + (i - 0.5) * w, z, w, label=k, color=["#555555", "#BBBBBB"][i])
    ax.axhline(2, color="red", lw=0.8, ls="--"); ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(len(PAIRS)), [f"{a_}-{b_}" for a_, b_ in PAIRS]); ax.set_ylabel("z vs shuffled-fact null"); ax.legend()
    fig.savefig(out / "overlap_z_by_model.png", dpi=130)
    print(json.dumps({"n_valid": res["n_valid"], "n_both": res["n_both"], "both_by_relation": res["both_by_relation"]}))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
