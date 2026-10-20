"""Paper figures from existing run artifacts (CPU only). Writes PNG + PDF to paper/figures/ and a sources list.

fig1_profiles: median per-layer MLP indirect effect at each language's last subject token (valid facts, per model).
fig2_overlap:  observed median MLP-profile cosine with 95% CI against the pooled permutation-null mean (per pair, per model).
fig3_hindi:    canonical-phrasing accuracy by answer token count per language (Hindi diagnostic).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from circuit_overlap import is_valid, load

LANGS = ("en", "es", "hi")
COL = {"en": "#1f77b4", "es": "#d95f02", "hi": "#2ca02c"}
MODELS = {
    "Llama-3.2-3B": {"trace": "analysis/run_20261015_p3trace_v2_llama", "overlap": "analysis/run_20261015_p3overlap_v2b_llama/results.json", "diag": "llama"},
    "Gemma-2-2B": {"trace": "analysis/run_20261015_p3trace_v2c_gemma2", "overlap": "analysis/run_20261015_p3overlap_v2c_gemma2/results.json", "diag": "gemma2"},
}
DIAG = "analysis/run_20261015_hindi_diag/results.json"


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_profiles(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=False)
    for ax, (name, m) in zip(axes, MODELS.items()):
        arr, recs = load(Path(m["trace"]))
        ids = sorted({f for f, _ in recs if all(is_valid(recs[(f, lg)]) for lg in LANGS)})
        for lg in LANGS:
            P = np.stack([arr[(f, lg)]["mlp"] for f in ids])
            P = P / (np.abs(P).sum(1, keepdims=True) + 1e-12)  # each fact's profile sums to 1 in absolute value
            med = np.median(P, 0)
            lo, hi = np.percentile(P, [25, 75], 0)
            x = np.arange(P.shape[1])
            ax.plot(x, med, color=COL[lg], label=lg.upper())
            ax.fill_between(x, lo, hi, color=COL[lg], alpha=0.15)
        ax.set_title(f"{name} (n={len(ids)})", fontsize=10)
        ax.set_xlabel("layer")
        ax.axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("normalised MLP indirect effect\n(median, IQR band)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "fig1_profiles")


def fig_overlap(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    pairs = ["en-es", "en-hi", "es-hi"]
    for ax, (name, m) in zip(axes, MODELS.items()):
        d = json.loads(Path(m["overlap"]).read_text(encoding="utf8"))["sets"]["primary_valid"]["pairs"]
        for i, p in enumerate(pairs):
            s = d[p]["mlp_cosine"]
            ax.errorbar(i, s["observed_median"], yerr=[[s["observed_median"] - s["ci95"][0]], [s["ci95"][1] - s["observed_median"]]],
                        fmt="o", color="#1f77b4", capsize=3, label="observed median, 95% CI" if i == 0 else None)
            ax.plot([i - 0.2, i + 0.2], [s["null_mean"]] * 2, color="#7f7f7f", lw=2, label="permutation-null mean" if i == 0 else None)
            ax.text(i, 0.74, f"z={s['z']:.2f}", ha="center", fontsize=8)
        ax.set_xticks(range(3))
        ax.set_xticklabels([p.upper() for p in pairs])
        ax.set_title(f"{name} (n={d['en-es']['mlp_cosine']['n']})", fontsize=10)
        ax.set_ylim(0, 0.8)
    axes[0].set_ylabel("MLP-profile cosine")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, out, "fig2_overlap")


def fig_hindi(out: Path) -> None:
    diag = json.loads(Path(DIAG).read_text(encoding="utf8"))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, (name, m) in zip(axes, MODELS.items()):
        acc = diag[m["diag"]]["accuracy_by_object_tokens"]
        buckets = ["1-3", "4-6", "7+"]
        w = 0.25
        for j, lg in enumerate(LANGS):
            v = [acc.get(f"{lg}/{b}", {}).get("acc") for b in buckets]
            n = [acc.get(f"{lg}/{b}", {}).get("n", 0) for b in buckets]
            xs = np.arange(3) + (j - 1) * w
            ax.bar(xs, [x if x is not None else 0 for x in v], w, color=COL[lg], label=lg.upper())
            for x, y, k in zip(xs, v, n):
                ax.text(x, (y or 0) + 0.015, f"{k}", ha="center", fontsize=6)
        ax.set_xticks(range(3))
        ax.set_xticklabels(buckets)
        ax.set_xlabel("answer length (tokens)")
        ax.set_title(name, fontsize=10)
    axes[0].set_ylabel("canonical-phrasing accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    fig.text(0.5, -0.02, "numbers above bars: facts in bucket", ha="center", fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig3_hindi")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_profiles(out)
    fig_overlap(out)
    fig_hindi(out)
    sources = {"fig1_profiles": [m["trace"] for m in MODELS.values()], "fig2_overlap": [m["overlap"] for m in MODELS.values()], "fig3_hindi": [DIAG]}
    (out / "sources.json").write_text(json.dumps(sources, indent=1), encoding="utf8")
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
