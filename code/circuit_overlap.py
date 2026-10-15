"""P3: cross-lingual circuit overlap against a shuffled-fact permutation null (statistics fixed in docs/HYPOTHESIS.md A3).

Primary: per-layer MLP-output attribution profile at each language's last subject token; per-fact cosine and Spearman
between language pairs; statistic = median over facts. Secondary: top-k IoU (k in 5, 10, 20) on a 112-cell grid.
Null: 1000 permutations pairing fact i's profile in language A with fact pi(i)'s profile in language B.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LANGS = ("en", "es", "hi")
PAIRS = list(combinations(LANGS, 2))
KS = (5, 10, 20)
N_PERM, N_BOOT = 1000, 2000


def load(run: Path) -> tuple[dict, dict]:
    """Return per-(fact, lang) arrays and record dicts."""
    recs = {(r["id"], r["lang"]): r for r in json.loads((run / "records.json").read_text(encoding="utf8"))}
    arr = {}
    for (fid, lg), r in recs.items():
        z = np.load(run / "grids" / f"{fid}_{lg}.npz")
        span = z["span"].tolist()
        t = z["mlp"].shape[1]
        cells = np.concatenate([z["resid"][:, span].sum(1), z["resid"][:, t - 1],
                                z["mlp"][:, span].sum(1), z["mlp"][:, t - 1]])  # 4 x 28 = 112 cells
        arr[(fid, lg)] = {"mlp": z["mlp"][:, r["last_subj"]], "resid": z["resid"][:, r["last_subj"]], "cells": cells}
    return arr, recs


def is_valid(r: dict) -> bool:
    return bool(r["collapsed"] and r["obj_ok"] and (r["score_clean"] - r["score_corr"]) > 0)


def cos_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T


def rank_rows(A: np.ndarray) -> np.ndarray:
    return pd.DataFrame(A).rank(axis=1).to_numpy()


def spearman_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ra, rb = rank_rows(A), rank_rows(B)
    return cos_matrix(ra - ra.mean(1, keepdims=True), rb - rb.mean(1, keepdims=True))


def topk_mask(C: np.ndarray, k: int) -> np.ndarray:
    idx = np.argsort(-C, axis=1)[:, :k]
    m = np.zeros_like(C, dtype=np.float32)
    np.put_along_axis(m, idx, 1.0, axis=1)
    return m


def iou_matrix(A: np.ndarray, B: np.ndarray, k: int) -> np.ndarray:
    inter = topk_mask(A, k) @ topk_mask(B, k).T
    return inter / (2 * k - inter)


def stat_block(M: np.ndarray, perms: np.ndarray, rng: np.random.Generator) -> dict:
    """M[i, j] = similarity of fact i (language A) with fact j (language B). Statistic = median over facts."""
    n = M.shape[0]
    obs_vec = np.diag(M)
    obs = float(np.median(obs_vec))
    null = np.median(M[np.arange(n)[None, :], perms], axis=1)  # [N_PERM]
    boot = np.array([np.median(obs_vec[rng.integers(0, n, n)]) for _ in range(N_BOOT)])
    sd = float(null.std(ddof=1))
    return {"n": n, "observed_median": obs, "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "null_mean": float(null.mean()), "null_sd": sd, "z": float((obs - null.mean()) / sd) if sd > 0 else None,
            "p_one_sided": float((1 + (null >= obs).sum()) / (N_PERM + 1)), "observed_mean": float(obs_vec.mean())}


def analyse(arr: dict, fids: list[str], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(fids)
    perms = np.stack([rng.permutation(n) for _ in range(N_PERM)])
    out: dict = {"n_facts": n, "pairs": {}}
    for a, b in PAIRS:
        res: dict = {}
        for kind in ("mlp", "resid"):
            A = np.stack([arr[(f, a)][kind] for f in fids]); B = np.stack([arr[(f, b)][kind] for f in fids])
            res[f"{kind}_cosine"] = stat_block(cos_matrix(A, B), perms, rng)
            res[f"{kind}_spearman"] = stat_block(spearman_matrix(A, B), perms, rng)
        CA = np.stack([arr[(f, a)]["cells"] for f in fids]); CB = np.stack([arr[(f, b)]["cells"] for f in fids])
        for k in KS:
            res[f"iou_top{k}"] = stat_block(iou_matrix(CA, CB, k), perms, rng)
        out["pairs"][f"{a}-{b}"] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="trace run dir (all facts)")
    ap.add_argument("--tag", default="p3overlap")
    ap.add_argument("--seed", type=int, default=20261012)
    a = ap.parse_args()
    run = Path(a.run)
    arr, recs = load(run)
    facts = sorted({f for f, _ in recs})
    rel = {f: recs[(f, "en")]["relation"] for f in facts}
    valid = [f for f in facts if all(is_valid(recs[(f, lg)]) for lg in LANGS)]
    sets = {"primary_valid": valid, "all_kept": facts,
            "valid_no_copyable": [f for f in valid if rel[f] not in ("event_year", "invention_inventor")]}
    for r in sorted(set(rel.values())):
        sets[f"valid_{r}"] = [f for f in valid if rel[f] == r]

    out = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}"
    out.mkdir(parents=True, exist_ok=False)  # write-once
    results = {"source_run": str(run), "seed": a.seed, "n_perm": N_PERM, "n_boot": N_BOOT,
               "n_all": len(facts), "n_valid": len(valid), "excluded": sorted(set(facts) - set(valid)), "sets": {}}
    for name, fids in sets.items():
        if len(fids) >= 10:
            results["sets"][name] = analyse(arr, fids, a.seed)
    # critical layers (H3)
    crit = {lg: [recs[(f, lg)]["crit_mlp_layer"] for f in valid] for lg in LANGS}
    results["critical_mlp_layer"] = {lg: {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
                                          "q75": float(np.percentile(v, 75)),
                                          "frac_in_8_19": float(np.mean([8 <= x <= 19 for x in v]))} for lg, v in crit.items()}
    d = {f"{a_}-{b_}": np.abs(np.array(crit[a_]) - np.array(crit[b_])) for a_, b_ in PAIRS}
    results["critical_layer_abs_diff"] = {k: {"median": float(np.median(v)), "frac_within_2": float(np.mean(v <= 2))}
                                          for k, v in d.items()}
    (out / "config.json").write_text(json.dumps({"source_run": str(run), "seed": a.seed, "sets": {k: len(v) for k, v in sets.items()}}, indent=1))
    (out / "results.json").write_text(json.dumps(results, indent=1))

    # table + figure for the primary set
    P = results["sets"]["primary_valid"]["pairs"]
    lines = ["| pair | statistic | observed median [95% CI] | null mean | z | p |", "|---|---|---|---|---|---|"]
    for pair, res in P.items():
        for stat, s in res.items():
            lines.append(f"| {pair} | {stat} | {s['observed_median']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] | "
                         f"{s['null_mean']:.3f} | {s['z']:.2f} | {s['p_one_sided']:.3f} |" if s["z"] is not None else
                         f"| {pair} | {stat} | {s['observed_median']:.3f} | {s['null_mean']:.3f} | n/a | {s['p_one_sided']:.3f} |")
    (out / "table_primary.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for lg, c in zip(LANGS, ("#0072B2", "#E69F00", "#009E73")):
        M = np.stack([arr[(f, lg)]["mlp"] / (recs[(f, lg)]["score_clean"] - recs[(f, lg)]["score_corr"]) for f in valid])
        ax.plot(M.mean(0), color=c, label=f"{lg} (n={len(valid)})")
    ax.set_xlabel("layer"); ax.set_ylabel("MLP effect / total effect"); ax.set_title("Mean MLP profile at last subject token")
    ax.legend()
    fig.savefig(out / "mean_profiles.png", dpi=130)
    print("\n".join(lines))
    print(json.dumps({k: results[k] for k in ("n_all", "n_valid", "excluded", "critical_mlp_layer", "critical_layer_abs_diff")}, indent=1))


if __name__ == "__main__":
    main()
