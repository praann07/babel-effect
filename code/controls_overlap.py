"""A11 controls for the EN-ES overlap (CPU, cached traces): same-language baselines, centred statistic, lexical-overlap split, entity-cluster bootstrap.

Writes analysis/run_<date>_controls/results.json. Pre-registered in docs/HYPOTHESIS.md (A11) before this was run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import unicodedata
from pathlib import Path

import numpy as np
import circuit_overlap as co
from circuit_overlap import analyse, cos_matrix, is_valid, load

SETS = {"llama": ["analysis/run_20261015_p3trace_v2_llama", "analysis/run_20261019_p3extra_10191057_llama", "data/v2/facts_llama.json", "data/v2/facts_extra_llama.json"],
        "gemma2": ["analysis/run_20261015_p3trace_v2c_gemma2", "analysis/run_20261019_p3extra_10191057_gemma2", "data/v2/facts_gemma2.json", "data/v2/facts_extra_gemma2.json"]}


def spearman(x: np.ndarray, y: np.ndarray, rng, n: int = 5000) -> tuple[float, float]:
    """Spearman rho (rank Pearson, average ranks not needed for ties-free float data except sim; ties broken by order) and two-sided permutation p."""
    rx, ry = np.argsort(np.argsort(x, kind="stable")), np.argsort(np.argsort(y, kind="stable"))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n)])
    return rho, float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n + 1))


def norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def pick(res: dict) -> dict:
    p = res["pairs"]["en-es"]["mlp_cosine"]
    return {"n": p["n"], "median": p["observed_median"], "ci95": p["ci95"], "null_mean": p["null_mean"], "z": p["z"], "p_one_sided": p["p_one_sided"]}


def clusters(facts: dict, ids: list[str]) -> list[int]:
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    seen: dict[str, str] = {}
    for i in ids:
        for k in (facts[i]["en"]["subject"], facts[i]["en"]["object"]):
            k = norm(k)
            if k in seen:
                parent[find(i)] = find(seen[k])
            else:
                seen[k] = i
    roots = {r: j for j, r in enumerate(sorted({find(i) for i in ids}))}
    return [roots[find(i)] for i in ids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--tag", default="controls")
    a = ap.parse_args()
    co.N_PERM, co.N_BOOT = a.perm, 2000
    out = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}"
    out.mkdir(parents=True, exist_ok=False)
    res: dict = {"n_perm": a.perm}
    rng = np.random.default_rng(0)
    for m, (r1, r2, f1, f2) in SETS.items():
        arr, recs, facts = {}, {}, {}
        for r in (r1, r2):
            A, R = load(Path(r)); arr.update(A); recs.update(R)
        for f in (f1, f2):
            for x in json.loads(Path(f).read_text(encoding="utf8"))["facts"]:
                facts[x["id"]] = {lg: x["lang"][lg] for lg in ("en", "es", "hi")} | {"relation": x["relation"]}
        ids = sorted({f for f, _ in recs if all((f, lg) in recs and is_valid(recs[(f, lg)]) for lg in ("en", "es", "hi"))})
        strata = [facts[i]["relation"] for i in ids]
        E, S = (np.stack([arr[(i, lg)]["mlp"] for i in ids]) for lg in ("en", "es"))
        out_m: dict = {"n": len(ids)}
        same_rel = np.array([[a == b for b in strata] for a in strata]) & ~np.eye(len(ids), dtype=bool)
        cs = {k: cos_matrix(x, y) for k, (x, y) in {"en-es": (E, S), "en-en": (E, E), "es-es": (S, S)}.items()}
        out_m["C0"] = {"same_fact_en_es": float(np.mean(np.diag(cs["en-es"]))), "diff_fact_same_rel_en_es": float(cs["en-es"][same_rel].mean()),
                       "diff_fact_same_rel_en_en": float(cs["en-en"][same_rel].mean()), "diff_fact_same_rel_es_es": float(cs["es-es"][same_rel].mean()),
                       "diff_fact_any_rel_en_es": float(cs["en-es"][~np.eye(len(ids), dtype=bool)].mean())}
        cen = {k: v for k, v in arr.items()}
        mu = {lg: np.mean([arr[(i, lg)]["mlp"] for i in ids], 0) for lg in ("en", "es", "hi")}
        cen = {(i, lg): {**arr[(i, lg)], "mlp": arr[(i, lg)]["mlp"] - mu[lg]} for i in ids for lg in ("en", "es", "hi")}
        out_m["C1_uncentred_stratified"] = pick(analyse(arr, ids, 0, strata))
        out_m["C1_centred_stratified"] = pick(analyse(cen, ids, 0, strata))
        sim = np.array([difflib.SequenceMatcher(None, norm(facts[i]["en"]["subject"]), norm(facts[i]["es"]["subject"])).ratio() for i in ids])
        per_fact = np.diag(cs["en-es"])
        lo = sim <= np.median(sim)
        out_m["C2"] = {"sim_median": float(np.median(sim)), "frac_identical_after_norm": float(np.mean(sim == 1.0)), "spearman_sim_vs_cos": list(spearman(sim, per_fact, rng))}
        for lab, mask in (("low", lo), ("high", ~lo)):
            sub = [i for i, k in zip(ids, mask) if k]
            st = [s for s, k in zip(strata, mask) if k]
            out_m["C2"][lab] = {"n": len(sub), "uncentred": pick(analyse(arr, sub, 0, st)), "centred": pick(analyse(cen, sub, 0, st))}
        cl = np.array(clusters(facts, ids))
        out_m["C3"] = {"n_clusters": int(cl.max() + 1)}
        meds = []
        for _ in range(2000):
            pick_c = rng.choice(cl.max() + 1, cl.max() + 1)
            idx = np.concatenate([np.where(cl == c)[0] for c in pick_c])
            meds.append(np.median(per_fact[idx]))
        out_m["C3"]["median_cos"] = float(np.median(per_fact))
        out_m["C3"]["cluster_boot_ci95"] = [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]
        res[m] = out_m
        print(m, json.dumps({k: v for k, v in out_m.items() if k in ("n", "C0", "C3")}, indent=0)[:600])
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    (out / "config.json").write_text(json.dumps({"sets": SETS, "n_perm": a.perm, "amendment": "A11"}, indent=1), encoding="utf8")
    print("wrote", out)


if __name__ == "__main__":
    main()
