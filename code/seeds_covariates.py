"""A13b: A12a S4a covariate model on per-fact increments D_i from the A13a independent-noise profiles (CPU)."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from circuit_overlap import is_valid, load
from confound_overlap import DELTA, SEED, TOK_DIR, Z975, center, cosm, ols_cr1, per_fact_D, pieces, sum_to_zero, unit
from controls_overlap import SETS, clusters
from matching import norm as mnorm
from seeds_analysis import meta


def covs(m: str, ids: list[str], tok):
    r1, r2, f1, f2 = SETS[m]
    recs, facts, rdir = {}, {}, {}
    for r in (r1, r2):
        R = load(Path(r))[1]
        recs.update(R)
        for k in R:
            rdir[k] = Path(r)
    for f in (f1, f2):
        for x in json.loads(Path(f).read_text(encoding="utf8"))["facts"]:
            facts[x["id"]] = x["lang"]
    J, LT, NR, SD, AI = [], [], [], [], []
    for i in ids:
        te, ts = np.load(rdir[(i, "en")] / "grids" / f"{i}_en.npz"), np.load(rdir[(i, "es")] / "grids" / f"{i}_es.npz")
        ie, is_, se, ss = te["tokens"].tolist(), ts["tokens"].tolist(), te["span"].tolist(), ts["span"].tolist()
        pe, ps = pieces(tok, [ie[k] for k in se]), pieces(tok, [is_[k] for k in ss])
        J.append(len(pe & ps) / max(len(pe | ps), 1)); LT.append(float(ie[se[-1]] == is_[ss[-1]])); NR.append(float(len(se) == len(ss)))
        SD.append(float(abs(len(se) - len(ss)))); AI.append(float(mnorm(facts[i]["en"]["target"]) == mnorm(facts[i]["es"]["target"])))
    TE = {lg: np.array([recs[(i, lg)]["score_clean"] - recs[(i, lg)]["score_corr"] for i in ids]) for lg in ("en", "es")}
    lte = {lg: np.log(TE[lg]) - np.log(TE[lg]).mean() for lg in TE}
    return {"J": np.array(J), "LT": np.array(LT), "AI": np.array(AI), "NR": np.array(NR), "SD": np.array(SD), "TE_en": lte["en"], "TE_es": lte["es"]}


def fit(D, cov, strata, cl, mask, rng):
    ok = mask & ~np.isnan(D)
    S0, levs = sum_to_zero(strata[ok])
    names = [k for k in cov if np.ptp(cov[k][ok]) > 0]
    X = np.column_stack([np.ones(ok.sum()), *[cov[k][ok] for k in names], S0])
    b, V, kk = ols_cr1(X, D[ok], cl[ok])
    c = np.zeros(X.shape[1]); c[0] = 1.0
    if "SD" in names:
        c[1 + names.index("SD")] = 1.0
    pred, se = float(c @ b), float(np.sqrt(c @ V @ c))
    ii = np.where(ok)[0]; uc = np.unique(cl[ii]); cidx = {u: np.where(cl[ii] == u)[0] for u in uc}
    bt = []
    for _ in range(1000):
        pk = np.concatenate([cidx[u] for u in rng.choice(uc, len(uc))])
        bt.append(float(c @ (np.linalg.pinv(X[pk]) @ D[ok][pk])))
    lo, hi = pred - Z975 * se, pred + Z975 * se
    return {"n": int(ok.sum()), "names": ["intercept"] + names + [f"stratum_{l}" for l in levs[:-1]], "beta": b.tolist(), "se": np.sqrt(np.diag(V)).tolist(),
            "rank": kk, "target_pred_D": pred, "target_ci97.5": [lo, hi], "target_boot_ci97.5": [float(np.percentile(bt, 1.25)), float(np.percentile(bt, 98.75))],
            "n_J0": int((cov["J"][ok] == 0).sum()), "n_NR0": int((cov["NR"][ok] == 0).sum()),
            "verdict": "present" if lo > 0 else ("absent" if hi < DELTA else "unresolved")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-run", required=True); ap.add_argument("--gemma-run", required=True); ap.add_argument("--tag", default="seeds_covariates_A13b")
    ap.add_argument("--exclude-leak", action="store_true", help="A17: drop answer-in-subject facts")
    a = ap.parse_args()
    if a.exclude_leak:
        import seeds_analysis
        from leakage import leaky_ids
        seeds_analysis.EXCLUDE = leaky_ids()
    from transformers import AutoTokenizer
    out = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}"; out.mkdir(parents=True, exist_ok=False)
    res = {}
    for m, run in (("llama", a.llama_run), ("gemma2", a.gemma_run)):
        rng = np.random.default_rng(SEED)
        ids, strata, cl, cls_, _ = meta(m)
        P = {lg: np.stack([np.load(Path(run) / "grids" / f"{i}_{lg}.npz")["ind"].mean(0) for i in ids]) for lg in ("en", "es")}
        D = per_fact_D(cosm(center(unit(P["en"]), strata), center(unit(P["es"]), strata)), strata)
        cov = covs(m, ids, AutoTokenizer.from_pretrained(TOK_DIR[m]))
        ev = strata == "event_year"
        res[m] = {"informative": fit(D, cov, strata, cl, (cls_ != "IP") & ~ev, rng), "all_non_IP": fit(D, cov, strata, cl, cls_ != "IP", rng)}
        for k, v in res[m].items():
            print(m, k, v["n"], round(v["target_pred_D"], 3), [round(x, 3) for x in v["target_ci97.5"]], v["verdict"], "J0", v["n_J0"], "NR0", v["n_NR0"])
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf8")
    (out / "config.json").write_text(json.dumps(vars(a) | {"amendment": "A13b"}, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
