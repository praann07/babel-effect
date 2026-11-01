"""A26 Test 1: does the English-key edit reach a language more often when that language's subject key is closer to the English key? CPU.

For language L in {es, hi} and each layer condition (traced, mid): Spearman rho between cos(k_EN, k_L) at the edit layer and the held-out transfer to L of the
English-key edit (A21 run). transfer = mean dP over the phrasings prompts[1:] of L / dP(English prompts[0]); facts with English dP <= 0.1 are dropped.
The key cosines are stored in the A24 run (one value per fact and layer, identical in every key mode; checked below).
Supported: rho >= 0.30, bootstrap-over-facts 95% CI above 0, and a Holm-adjusted one-sided permutation p < 0.05 over the four cells.
Beyond confounds: partial Spearman, controlling for English edit strength (dP_EN), the subject's token count in L and the subject-piece Jaccard EN-L, >= 0.20 with CI above 0.
Usage: python code/key_similarity.py A21_RUN --keys A24_RUN [--facts data/v2/facts_llama.json] [--model-dir models/llama-3.2-3b-instruct]
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RHO_MIN, PARTIAL_MIN, ALPHA, BOOT, PERM = 0.30, 0.20, 0.05, 5000, 10000
SEED = 20261030


def rankdata(v) -> np.ndarray:
    return pd.Series(np.asarray(v, float)).rank().to_numpy()


def pearson(a, b) -> float:
    a, b = np.asarray(a, float) - np.mean(a), np.asarray(b, float) - np.mean(b)
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else float("nan")


def spearman(x, y) -> float:
    return pearson(rankdata(x), rankdata(y))


def partial_spearman(x, y, controls) -> float:
    """Spearman correlation of x and y after removing the linear effect of the ranks of every control column."""
    A = np.column_stack([np.ones(len(x))] + [rankdata(c).astype(float) for c in controls])
    resid = lambda v: rankdata(v) - A @ np.linalg.lstsq(A, rankdata(v).astype(float), rcond=None)[0]
    return pearson(resid(x), resid(y))


def boot_ci(fn, n: int, rng) -> list[float]:
    vals = []
    for _ in range(BOOT):
        i = rng.integers(0, n, n)
        v = fn(i)
        if np.isfinite(v):
            vals.append(v)
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def perm_p_greater(x, y, rng) -> float:
    """One-sided permutation p for rho > 0 (the +1 keeps it above zero)."""
    obs = spearman(x, y)
    rx, ry = rankdata(x), rankdata(y)
    hits = sum(pearson(rx, rng.permutation(ry)) >= obs for _ in range(PERM))
    return (hits + 1) / (PERM + 1)


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adj, run = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def pieces(tok, prompt: str, subject: str) -> set[str]:
    from common import subject_span
    ids, pos = subject_span(tok, prompt, subject)
    return {s for s in (tok.decode([int(ids[k])]).strip().lower() for k in pos) if s}


def covariates(facts: dict, tok, lg: str) -> dict[str, tuple[int, float]]:
    """fact id -> (subject token count in lg, Jaccard of the EN and lg subject pieces); the same rule as interchange.jaccard0."""
    out = {}
    for f in facts:
        en, L = f["lang"]["en"], f["lang"][lg]
        a, b = pieces(tok, en["prompts"][0], en["subject"]), pieces(tok, L["prompts"][0], L["subject"])
        out[f["id"]] = (len(b), len(a & b) / max(len(a | b), 1))
    return out


def held_transfer(r: dict, lg: str) -> float:
    return float(np.mean([r["para"][k]["dP"] for k in r["para"] if k.startswith(lg)])) / r["en"]["dP"]


def cell(rows: list[dict], keycos: dict, lg: str, cov: dict, rng) -> dict:
    x = np.array([keycos[(r["fact"], r["layer"])][lg] for r in rows])
    y = np.array([held_transfer(r, lg) for r in rows])
    ctrl = [np.array([r["en"]["dP"] for r in rows]), np.array([cov[r["fact"]][0] for r in rows], float), np.array([cov[r["fact"]][1] for r in rows])]
    o = {"n": len(rows), "key_cos_median": float(np.median(x)), "key_cos_iqr": [float(v) for v in np.percentile(x, [25, 75])], "rho": spearman(x, y),
         "rho_ci95": boot_ci(lambda i: spearman(x[i], y[i]), len(x), rng), "p_one_sided": perm_p_greater(x, y, rng)}
    o["partial_rho"] = partial_spearman(x, y, ctrl)
    o["partial_ci95"] = boot_ci(lambda i: partial_spearman(x[i], y[i], [c[i] for c in ctrl]), len(x), rng)
    o["controls_spread"] = {"span_tokens_distinct": int(len(set(ctrl[1]))), "jaccard_distinct": int(len(set(ctrl[2])))}
    # not registered, reported as extra checks: the edit layer differs by fact in the traced condition, and one fact should not carry the result
    layer = np.array([r["layer"] for r in rows], float)
    if len(set(layer)) > 1:
        o["extra_partial_rho_with_layer"] = partial_spearman(x, y, ctrl + [layer])
        o["extra_partial_with_layer_ci95"] = boot_ci(lambda i: partial_spearman(x[i], y[i], [c[i] for c in ctrl] + [layer[i]]), len(x), rng)
        o["extra_layers"] = {str(int(v)): int((layer == v).sum()) for v in sorted(set(layer))}
    loo = [spearman(np.delete(x, i), np.delete(y, i)) for i in range(len(x))]
    o["extra_leave_one_out_rho_range"] = [float(min(loo)), float(max(loo))]
    return o


def main() -> None:
    args = sys.argv[1:]
    opt = lambda k, d: args[args.index(k) + 1] if k in args else d
    run, keys = Path(args[0]), Path(opt("--keys", ""))
    facts = json.loads(Path(opt("--facts", "data/v2/facts_llama.json")).read_text(encoding="utf8"))["facts"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(opt("--model-dir", "models/llama-3.2-3b-instruct"))
    load = lambda p: [json.loads(l) for l in open(p / "records.jsonl", encoding="utf8")]
    keycos, spread = {}, 0.0
    for r in load(keys):
        k = (r["fact"], r["layer"])
        if k in keycos:
            spread = max(spread, *(abs(keycos[k][lg] - r["key_cos"][lg]) for lg in ("es", "hi")))
        else:
            keycos[k] = r["key_cos"]
    rng = np.random.default_rng(SEED)
    out = {"key_cos_max_diff_between_modes": spread, "cells": {}}
    cells = []
    for lg in ("es", "hi"):
        cov = covariates(facts, tok, lg)
        for c in ("traced", "mid"):
            rows = [r for r in load(run) if ("traced" if r["is_traced"] else "mid") == c and "para" in r and r["en"]["dP"] > 0.1 and (r["fact"], r["layer"]) in keycos]
            out["cells"][f"{lg}_{c}"] = cell(rows, keycos, lg, cov, rng)
            cells.append(f"{lg}_{c}")
    for name, a in zip(cells, holm([out["cells"][c]["p_one_sided"] for c in cells])):
        o = out["cells"][name]
        o["p_holm"] = a
        o["supported"] = bool(o["rho"] >= RHO_MIN and o["rho_ci95"][0] > 0 and a < ALPHA)
        o["beyond_confounds"] = bool(o["partial_rho"] >= PARTIAL_MIN and o["partial_ci95"][0] > 0)
    rho = {c: out["cells"][c]["rho"] for c in cells}
    out["predictions"] = {"rho_positive_in_all_four": all(v > 0 for v in rho.values()),
                          "hindi_weaker_than_spanish_in_both_conditions": bool(rho["hi_traced"] < rho["es_traced"] and rho["hi_mid"] < rho["es_mid"])}
    d = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_key_similarity_A26"
    d.mkdir(parents=True, exist_ok=False)
    (d / "results.json").write_text(json.dumps(out, indent=1))
    for c in cells:  # the results are already saved: printing must never fail (a retry would meet the write-once folder)
        print(c, json.dumps({k: v for k, v in out["cells"][c].items() if k != "controls_spread"}, default=str))
    print(out["predictions"], "key cosine max diff between modes", spread, "->", d)


if __name__ == "__main__":
    main()
