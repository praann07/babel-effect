"""P5 (v2): aggregate the layer-sweep edit run into per-layer curves, the A5 conditions and the H7 test (bootstrap CIs over facts)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LANGS = ("en", "es", "hi")
N_BOOT = 2000


def boot_median(x: list[float], rng: np.random.Generator) -> list[float]:
    a = np.asarray(x, dtype=float)
    if not len(a):
        return [float("nan")] * 3
    b = np.array([np.median(a[rng.integers(0, len(a), len(a))]) for _ in range(N_BOOT)])
    return [float(np.median(a)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def transfer(r: dict, lg: str) -> float | None:
    d = r["en"]["dP"]
    return r[lg]["dP"] / d if d > 1e-6 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--seed", type=int, default=20261012)
    a = ap.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    rows = [json.loads(l) for l in (run / "records.jsonl").read_text(encoding="utf8").splitlines()]
    L = cfg["n_layers"]
    mid = L // 2
    rng = np.random.default_rng(a.seed)
    facts = sorted({r["fact"] for r in rows})
    seeds = {r["fact"]: r for r in rows}
    out: dict = {"n_rows": len(rows), "facts": facts, "n_layers": L, "mid_layer": mid, "per_layer": {}, "conditions": {}, "H7": {}}

    for layer in range(L):
        R = [r for r in rows if r["layer"] == layer]
        if not R:
            continue
        d = {"n": len(R), "en_success_rate": float(np.mean([r["en"]["success"] for r in R]))}
        for lg in LANGS:
            d[f"dP_{lg}"] = boot_median([r[lg]["dP"] for r in R], rng)
            d[f"success_{lg}"] = float(np.mean([r[lg]["success"] for r in R]))
        for lg in ("es", "hi"):
            d[f"transfer_{lg}"] = boot_median([t for r in R if r["en"]["success"] and (t := transfer(r, lg)) is not None], rng)
        d["specificity"] = boot_median([r["specificity_max_abs_dP"] for r in R], rng)
        d["ppl_ratio"] = boot_median([r["ppl_ratio"] for r in R], rng)
        out["per_layer"][layer] = d

    # A5 conditions read from the sweep
    for cond in ("traced", "mid", "random"):
        sel = []
        for f in facts:
            Rf = [r for r in rows if r["fact"] == f]
            traced = next((r["layer"] for r in Rf if r["is_traced"]), None)
            if cond == "traced":
                pick = [r for r in Rf if r["layer"] == traced]
            elif cond == "mid":
                pick = [r for r in Rf if r["layer"] == mid]
            else:
                rr = random.Random(seeds[f]["fact"] + "|random")
                choices = [l for l in sorted({r["layer"] for r in Rf}) if l not in (traced, mid)]
                pick = [r for r in Rf if choices and r["layer"] == rr.choice(choices)]
            sel += pick
        d = {"n": len(sel), "en_success_rate": float(np.mean([r["en"]["success"] for r in sel])) if sel else None}
        for lg in LANGS:
            d[f"dP_{lg}"] = boot_median([r[lg]["dP"] for r in sel], rng)
        for lg in ("es", "hi"):
            d[f"transfer_{lg}"] = boot_median([t for r in sel if r["en"]["success"] and (t := transfer(r, lg)) is not None], rng)
        d["specificity"] = boot_median([r["specificity_max_abs_dP"] for r in sel], rng)
        d["ppl_ratio"] = boot_median([r["ppl_ratio"] for r in sel], rng)
        out["conditions"][cond] = d

    # H7: per fact, median transfer ratio (and specificity) at layers within 3 of mid vs at layers 0-5, paired over facts
    near, early = range(max(0, mid - 3), min(L, mid + 4)), range(0, 6)
    for key in ("transfer_es", "transfer_hi", "specificity"):
        diffs = []
        for f in facts:
            def per_fact(layers):
                v = []
                for r in rows:
                    if r["fact"] == f and r["layer"] in layers and r["en"]["success"]:
                        x = r["specificity_max_abs_dP"] if key == "specificity" else transfer(r, key[-2:])
                        if x is not None:
                            v.append(x)
                return float(np.median(v)) if v else None
            n_, e_ = per_fact(near), per_fact(early)
            if n_ is not None and e_ is not None:
                diffs.append(n_ - e_)
        out["H7"][key] = {"n_facts": len(diffs), "median_diff_near_mid_minus_early": boot_median(diffs, rng),
                          "frac_facts_positive": float(np.mean([d > 0 for d in diffs])) if diffs else None}
    (run / "sweep_tables.json").write_text(json.dumps(out, indent=1))

    # curves
    ls = sorted(out["per_layer"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    col = {"en": "#0072B2", "es": "#E69F00", "hi": "#009E73"}
    ax = axes[0, 0]
    for lg in LANGS:
        ax.plot(ls, [out["per_layer"][l][f"success_{lg}"] for l in ls], color=col[lg], label=lg)
    ax.set_title("edit success (greedy full answer = new target)"); ax.set_xlabel("layer"); ax.legend()
    ax = axes[0, 1]
    for lg in ("es", "hi"):
        m = [out["per_layer"][l][f"transfer_{lg}"] for l in ls]
        ax.plot(ls, [x[0] for x in m], color=col[lg], label=lg)
        ax.fill_between(ls, [x[1] for x in m], [x[2] for x in m], color=col[lg], alpha=0.2)
    ax.axvline(mid, color="grey", ls=":"); ax.set_title("transfer ratio (median, 95% CI)"); ax.set_xlabel("layer"); ax.legend()
    ax = axes[1, 0]
    m = [out["per_layer"][l]["specificity"] for l in ls]
    ax.plot(ls, [x[0] for x in m], color="black"); ax.fill_between(ls, [x[1] for x in m], [x[2] for x in m], color="black", alpha=0.15)
    ax.set_title("specificity: max |dP| over 20 controls"); ax.set_xlabel("layer")
    ax = axes[1, 1]
    m = [out["per_layer"][l]["ppl_ratio"] for l in ls]
    ax.plot(ls, [x[0] for x in m], color="black"); ax.fill_between(ls, [x[1] for x in m], [x[2] for x in m], color="black", alpha=0.15)
    ax.set_title("damage: perplexity ratio"); ax.set_xlabel("layer")
    fig.savefig(run / "sweep_curves.png", dpi=130)
    print(json.dumps({"conditions": out["conditions"], "H7": out["H7"]}, indent=1))


if __name__ == "__main__":
    main()
