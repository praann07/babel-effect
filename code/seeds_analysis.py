"""A13a analysis of the independent-noise re-trace (CPU). Follows docs/HYPOTHESIS.md A13a exactly; writes analysis/run_<date>_<tag>/results.json."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from circuit_overlap import is_valid, load
from confound_overlap import SEED, DELTA, center, cluster_boot_mean, cosm, groups_of, per_fact_D, s1, unit
from controls_overlap import SETS, clusters

SEEDRUN = {"llama": None, "gemma2": None}  # set by --llama-run / --gemma-run
EXCLUDE: set = set()  # A17: fact ids dropped from every analysis that goes through meta() (answer-in-subject leakage)


def meta(m: str):
    r1, r2, f1, f2 = SETS[m]
    arr, recs, facts, rdir = {}, {}, {}, {}
    for r in (r1, r2):
        A, R = load(Path(r)); arr.update(A); recs.update(R)
        for k in R:
            rdir[k] = Path(r)
    for f in (f1, f2):
        for x in json.loads(Path(f).read_text(encoding="utf8"))["facts"]:
            facts[x["id"]] = {lg: x["lang"][lg] for lg in ("en", "es", "hi")} | {"relation": x["relation"]}
    ids = sorted({f for f, _ in recs if f not in EXCLUDE and all((f, lg) in recs and is_valid(recs[(f, lg)]) for lg in ("en", "es", "hi"))})
    rel = np.array([facts[i]["relation"] for i in ids])
    cnt = {r: int((rel == r).sum()) for r in set(rel)}
    strata = np.array([r if cnt[r] >= 10 else "other" for r in rel])
    cl = np.array(clusters(facts, ids))
    cls_ = []
    for i in ids:
        te = np.load(rdir[(i, "en")] / "grids" / f"{i}_en.npz"); ts = np.load(rdir[(i, "es")] / "grids" / f"{i}_es.npz")
        ie, is_, se, ss = te["tokens"].tolist(), ts["tokens"].tolist(), te["span"].tolist(), ts["span"].tolist()
        cls_.append("IP" if ie[:se[-1] + 1] == is_[:ss[-1] + 1] else ("EQ" if len(se) == len(ss) else "OT"))
    return ids, strata, cl, np.array(cls_), arr


def nc(P: np.ndarray, strata: np.ndarray) -> np.ndarray:
    return center(unit(P), strata)


def main() -> None:
    global EXCLUDE
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-run", required=True)
    ap.add_argument("--gemma-run", required=True)
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--tag", default="seeds_A13a")
    ap.add_argument("--exclude-leak", action="store_true", help="A17: drop answer-in-subject facts")
    a = ap.parse_args()
    if a.exclude_leak:
        from leakage import leaky_ids
        EXCLUDE = leaky_ids()
    B = a.perm
    out = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}"
    out.mkdir(parents=True, exist_ok=False)
    res: dict = {"n_perm": B, "seed": SEED, "delta": DELTA}
    for m, run in (("llama", a.llama_run), ("gemma2", a.gemma_run)):
        rng = np.random.default_rng(SEED)
        ids, strata, cl, cls_, arr = meta(m)
        G = {lg: [np.load(Path(run) / "grids" / f"{i}_{lg}.npz") for i in ids] for lg in ("en", "es")}
        S = {lg: {"ind": np.stack([g["ind"].mean(0) for g in G[lg]]), "A": np.stack([g["ind"][:5].mean(0) for g in G[lg]]),
                  "Bv": np.stack([g["ind"][5:].mean(0) for g in G[lg]]), "sh": np.stack([g["sh"].mean(0) for g in G[lg]]),
                  "orig": np.stack([g["orig"] for g in G[lg]])} for lg in ("en", "es")}
        ev = strata == "event_year"
        sets = {"all": np.ones(len(ids), bool), "informative": (cls_ != "IP") & ~ev, "EQ": cls_ == "EQ", "OT": cls_ == "OT",
                "event_year_only": ev, "non_event_year": ~ev}
        o: dict = {"n": len(ids), "run": run, "prefix_class_counts": {c: int((cls_ == c).sum()) for c in ("IP", "EQ", "OT")}}
        # P4 regression: ORIG vs cached profiles
        reg = {}
        for lg in ("en", "es"):
            cached = np.stack([arr[(i, lg)]["mlp"] for i in ids])
            diff = np.abs(S[lg]["orig"] - cached)
            reg[lg] = {"median_abs_diff": float(np.median(diff)), "p95_abs_diff": float(np.percentile(diff, 95)),
                       "corr": float(np.corrcoef(S[lg]["orig"].ravel(), cached.ravel())[0, 1])}
        o["P4_regression"] = reg
        o["P4_pass"] = bool(all(v["median_abs_diff"] < 0.05 and v["corr"] > 0.99 for v in reg.values()))
        # matrices (NC)
        M = {"ind": cosm(nc(S["en"]["ind"], strata), nc(S["es"]["ind"], strata)), "sh": cosm(nc(S["en"]["sh"], strata), nc(S["es"]["sh"], strata)),
             "orig": cosm(nc(S["en"]["orig"], strata), nc(S["es"]["orig"], strata)),
             "AA": cosm(nc(S["en"]["A"], strata), nc(S["es"]["A"], strata)), "BB": cosm(nc(S["en"]["Bv"], strata), nc(S["es"]["Bv"], strata)),
             "enAB": cosm(nc(S["en"]["A"], strata), nc(S["en"]["Bv"], strata)), "esAB": cosm(nc(S["es"]["A"], strata), nc(S["es"]["Bv"], strata))}
        o["S"] = {}
        for nm, mask in sets.items():
            if mask.sum() < 3:
                o["S"][nm] = {"n": int(mask.sum()), "note": "too few"}; continue
            ix = np.where(mask)[0]
            sub = lambda k: M[k][np.ix_(ix, ix)]
            r = {k: s1(sub(k), strata[ix], cl[ix], B, rng) for k in ("ind", "sh", "orig", "enAB", "esAB")}
            d_aa, d_bb = per_fact_D(sub("AA"), strata[ix]), per_fact_D(sub("BB"), strata[ix])
            d_ind5 = (d_aa + d_bb) / 2
            d_sh = per_fact_D(sub("sh"), strata[ix])
            delta = d_sh - d_ind5
            r["D_ind5_mean"] = float(np.nanmean(d_ind5))
            r["D_sh_mean"] = float(np.nanmean(d_sh))
            r["Delta_SH_minus_IND5"] = {"mean": float(np.nanmean(delta)), "ci97.5_cluster": cluster_boot_mean(delta, cl[ix], rng)}
            ce, cs = r["enAB"].get("mean_D"), r["esAB"].get("mean_D")
            r["reliability_index"] = float(np.nanmean(d_ind5) / np.sqrt(ce * cs)) if ce and cs and ce > 0 and cs > 0 else None
            o["S"][nm] = r
        inf = o["S"]["informative"]
        ceil_ok = inf["enAB"].get("mean_D", 0) >= 0.10 and inf["esAB"].get("mean_D", 0) >= 0.10
        lo, hi = inf["ind"]["ci97.5_cluster"]
        o["G1"] = {"verdict": ("UNRESOLVED (ceiling < 0.10)" if not ceil_ok else ("SURVIVES" if lo > 0 else ("ARTEFACT" if hi < DELTA else "UNRESOLVED"))),
                   "D_ind_informative": inf["ind"]["mean_D"], "ci97.5": [lo, hi], "ceiling_en": inf["enAB"].get("mean_D"), "ceiling_es": inf["esAB"].get("mean_D")}
        eq, ot = o["S"].get("EQ", {}), o["S"].get("OT", {})
        o["predictions"] = {"P1_gemma_ind_includes_0_and_Delta_pos": bool(lo <= 0 <= hi and inf["Delta_SH_minus_IND5"]["ci97.5_cluster"][0] > 0),
                            "P2_llama_ind_pos_and_below_0.239": bool(lo > 0 and inf["ind"]["mean_D"] < 0.239),
                            "P3_Dsh_ge_Dind_on_EQ": bool(eq.get("D_sh_mean", -9) >= eq.get("ind", {}).get("mean_D", 9)) if eq else None,
                            "P4_regression_pass": o["P4_pass"]}
        res[m] = o
        print(m, o["G1"], o["predictions"])
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf8")
    (out / "config.json").write_text(json.dumps(vars(a) | {"amendment": "A13a"}, indent=1), encoding="utf8")
    print("wrote", out)


if __name__ == "__main__":
    main()
