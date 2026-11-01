"""Graphs of the Kaggle step's new results (PNG), written to analysis/figures_kaggle/. Reads the results.json of the paraphrase-control and the
shared-key-edit analyses (whichever exist) and plots median transfer with its bootstrap 95% interval. CPU only. Exit code 0 even when an input is
missing: a missing graph is reported in the log, not an error."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def newest(pattern: str) -> Path | None:
    hits = sorted(glob.glob(pattern))
    return Path(hits[-1]) if hits else None


def mid_layer() -> str:
    """Label of the fixed middle edit layer (L // 2) of the model whose edit run is in analysis/, read from that run's config."""
    cfg = newest("analysis/run_*_p4a28base*/config.json")
    try:
        return f"layer {json.loads(cfg.read_text(encoding='utf8'))['n_layers'] // 2}"
    except (AttributeError, KeyError, ValueError, OSError):
        return "middle layer"


def errbars(items: list[tuple[str, dict]]):
    xs = list(range(len(items)))
    med = [v["median"] for _, v in items]
    lo = [v["median"] - v["ci95"][0] for _, v in items]
    hi = [v["ci95"][1] - v["median"] for _, v in items]
    return xs, med, [lo, hi]


def plot_paraphrase(res: dict, out: Path) -> Path:
    """Transfer of the English-key edit: unseen English phrasings vs Spanish and Hindi (1.0 = the edit works as well as in English)."""
    labels = [("en_para", "English\nother phrasings"), ("es_main", "Spanish\nmain prompt"), ("es_para", "Spanish\nother phrasings"),
              ("hi_main", "Hindi\nmain prompt"), ("hi_para", "Hindi\nother phrasings")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, cond in zip(axes, ("traced", "mid")):
        if cond not in res:
            continue
        items = [(t, res[cond][k]) for k, t in labels if k in res[cond]]
        xs, med, err = errbars(items)
        ax.bar(xs, med, yerr=err, capsize=4, color=["#4C78A8", "#F58518", "#F58518", "#54A24B", "#54A24B"][:len(xs)])
        ax.set_xticks(xs, [t for t, _ in items], fontsize=8)
        ax.axhline(1.0, color="grey", lw=0.8, ls="--")
        ax.set_title(f"{cond} layer (n={res[cond].get('n', '?')} facts)")
    axes[0].set_ylabel("transfer ratio (median, 95% CI)")
    fig.suptitle("English-key edit: does it carry over? (paraphrase control)")
    fig.tight_layout()
    p = out / "paraphrase_control.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def plot_keymodes(res: dict, out: Path) -> Path:
    """Held-out transfer to Spanish and Hindi: English-key baseline vs each shared-key edit, paired over the same facts."""
    modes = [m for m in res if isinstance(res[m], dict)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for r, cond in enumerate(("traced", "mid")):
        for c, (lang, key) in enumerate((("Spanish", "es_held"), ("Hindi", "hi_held"))):
            ax = axes[r][c]
            items = []
            base = next((res[m][cond][key + "_base"] for m in modes if cond in res[m] and key + "_base" in res[m][cond]), None)
            if base:
                items.append(("English key\n(baseline)", base))
            items += [(m.replace("_", "\n"), res[m][cond][key]) for m in modes if cond in res[m] and key in res[m][cond]]
            if not items:
                continue
            xs, med, err = errbars(items)
            ax.bar(xs, med, yerr=err, capsize=4, color=["#9D9D9D"] + ["#E45756"] * (len(xs) - 1) if base else "#E45756")
            ax.set_xticks(xs, [t for t, _ in items], fontsize=8)
            ax.set_title(f"{lang}, {cond} layer")
            ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    for ax in axes[:, 0]:
        ax.set_ylabel("held-out transfer (median, 95% CI)")
    fig.suptitle("Edit written at a key built from several languages vs English key")
    fig.tight_layout()
    p = out / "shared_key_edit.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def plot_key_similarity(res: dict, out: Path) -> Path:
    """Spearman rho between cos(English key, language key) and held-out transfer, per language and layer condition (filled: all registered controls;
    open square: the edit layer also controlled)."""
    mid = mid_layer()
    names = [("es_traced", "Spanish\ntraced"), ("es_mid", f"Spanish\n{mid}"), ("hi_traced", "Hindi\ntraced"), ("hi_mid", f"Hindi\n{mid}")]
    cells = [(t, res["cells"][k]) for k, t in names if k in res["cells"]]
    xs = list(range(len(cells)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(xs, [c["rho"] for _, c in cells], yerr=[[c["rho"] - c["rho_ci95"][0] for _, c in cells], [c["rho_ci95"][1] - c["rho"] for _, c in cells]],
                fmt="o", capsize=4, color="#4C78A8", label="rho (95% CI)")
    lay = [(x, c["extra_partial_rho_with_layer"]) for x, (_, c) in zip(xs, cells) if "extra_partial_rho_with_layer" in c]
    if lay:
        ax.plot(*zip(*lay), "s", mfc="none", color="#E45756", label="partial rho, edit layer controlled")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axhline(0.3, color="grey", lw=0.8, ls="--")
    ax.set_xticks(xs, [t for t, _ in cells], fontsize=8)
    ax.set_ylabel("Spearman rho (key cosine vs held-out transfer)")
    ax.legend(fontsize=8)
    fig.suptitle("Do facts with closer keys transfer more?")
    fig.tight_layout()
    p = out / "key_similarity.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def plot_layer_map(res: dict, out: Path) -> Path:
    """Held-out transfer of the English-key edit by edit layer (top) and the within-layer rho of key cosine and transfer (bottom)."""
    layers = sorted(res["layers"], key=int)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for c, (lg, name) in enumerate((("es", "Spanish"), ("hi", "Hindi"))):
        med = [res["layers"][l][lg]["transfer"]["median"] for l in layers]
        lo = [m - res["layers"][l][lg]["transfer"]["ci95"][0] for m, l in zip(med, layers)]
        hi = [res["layers"][l][lg]["transfer"]["ci95"][1] - m for m, l in zip(med, layers)]
        axes[0][c].errorbar([int(l) for l in layers], med, yerr=[lo, hi], fmt="o-", capsize=4, color="#F58518")
        axes[0][c].set_title(f"{name}: median held-out transfer")
        axes[0][c].axhline(1.0, color="grey", lw=0.8, ls="--")
        axes[1][c].plot([int(l) for l in layers], [res["layers"][l][lg]["rho"] for l in layers], "o-", color="#4C78A8")
        axes[1][c].axhline(0, color="grey", lw=0.8)
        axes[1][c].set_xlabel("edit layer")
        axes[1][c].set_title(f"{name}: rho(key cosine, transfer) within layer")
    fig.suptitle("Where does an English edit reach other languages, and does key similarity matter?")
    fig.tight_layout()
    p = out / "layer_map.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main() -> int:
    out = Path("analysis/figures_kaggle")
    out.mkdir(parents=True, exist_ok=True)
    made = []
    a21 = newest("analysis/run_*_para_analysis_A21/results.json")
    a24 = newest("analysis/run_*_keymode_analysis_A24/results.json")
    if a21:
        made.append(plot_paraphrase(json.loads(a21.read_text(encoding="utf8")), out))
    else:
        print("no paraphrase-control results: graph skipped")
    if a24:
        made.append(plot_keymodes(json.loads(a24.read_text(encoding="utf8")), out))
    else:
        print("no shared-key results: graph skipped")
    for pattern, plot, what in (("analysis/run_*_key_similarity_A26/results.json", plot_key_similarity, "key-similarity"),
                                ("analysis/run_*_layer_map_A28/results.json", plot_layer_map, "layer-map")):
        res = newest(pattern)
        if res:
            made.append(plot(json.loads(res.read_text(encoding="utf8")), out))
        else:
            print(f"no {what} results: graph skipped")
    print("graphs:", ", ".join(str(m) for m in made) or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
