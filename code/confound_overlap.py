"""A12 / A12a confound-adjusted EN-ES overlap on cached traces (CPU, no model weights; tokenizers only).

Pre-registered in docs/HYPOTHESIS.md (A12, A12a) before this was run. Writes analysis/run_<date>_<tag>/results.json.
Profile = MLP-output indirect effect per layer at the last subject token. Strata = relation (relations with < 10 valid facts merged into "other").
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from circuit_overlap import is_valid, load
from controls_overlap import SETS, clusters, norm as accnorm
from matching import norm as mnorm

SEED = 20261023
DELTA = 0.05
TOK_DIR = {"llama": "models/llama-3.2-3b-instruct", "llama_base": "models/llama-3.2-3b-base", "gemma2": "models/gemma-2-2b"}
Z975 = 2.2414  # two-sided 97.5% interval (Bonferroni over the 2 primary quantities)


# ---------- helpers ----------
def avg_rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy()


def unit(P: np.ndarray) -> np.ndarray:
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def center(P: np.ndarray, strata: np.ndarray) -> np.ndarray:
    Q = P.copy()
    for s in np.unique(strata):
        m = strata == s
        Q[m] = P[m] - P[m].mean(0)
    return Q


def project_out(U: np.ndarray, strata: np.ndarray) -> np.ndarray:
    Q = U.copy()
    for s in np.unique(strata):
        m = strata == s
        mh = U[m].mean(0)
        mh = mh / np.linalg.norm(mh)
        Q[m] = U[m] - (U[m] @ mh)[:, None] * mh[None, :]
    return Q


def cosm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)) @ (B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)).T


def derangements(g: int, B: int, rng) -> np.ndarray:
    """B uniform derangements of 0..g-1 by rejection (g >= 2)."""
    out: list[np.ndarray] = []
    have = 0
    while have < B:
        P = rng.permuted(np.tile(np.arange(g), (max(B, 64), 1)), axis=1)
        P = P[(P != np.arange(g)).all(1)]
        out.append(P)
        have += len(P)
    return np.concatenate(out)[:B]


def groups_of(strata: np.ndarray) -> list[np.ndarray]:
    return [np.where(strata == s)[0] for s in np.unique(strata) if (strata == s).sum() >= 2]


def per_fact_D(M: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """D_i = M_ii - mean_{j != i, same stratum} M_ij (NaN where the stratum has one fact)."""
    D = np.full(len(strata), np.nan)
    for g in groups_of(strata):
        S = M[np.ix_(g, g)]
        off = (S.sum(1) - np.diag(S)) / (len(g) - 1)
        D[g] = np.diag(S) - off
    return D


def perm_mean_D(M: np.ndarray, strata: np.ndarray, B: int, rng) -> tuple[float, np.ndarray]:
    """Observed mean D and its null from within-stratum derangements of the ES profiles."""
    gs = groups_of(strata)
    n = sum(len(g) for g in gs)
    D = per_fact_D(M, strata)
    obs = float(np.nanmean(D))
    # null: mean_i M[i, pi(i)] - baseline_i, baseline_i = off-diagonal mean
    off_tot = 0.0
    for g in gs:
        S = M[np.ix_(g, g)]
        off_tot += ((S.sum(1) - np.diag(S)) / (len(g) - 1)).sum()
    tot = np.zeros(B)
    for g in gs:
        S = M[np.ix_(g, g)]
        P = derangements(len(g), B, rng)
        tot += S[np.arange(len(g))[None, :], P].sum(1)
    return obs, (tot - off_tot) / n


def cluster_boot_mean(D: np.ndarray, cl: np.ndarray, rng, n_boot: int = 2000) -> list[float]:
    ok = ~np.isnan(D)
    D, cl = D[ok], cl[ok]
    uc = np.unique(cl)
    idx = {c: np.where(cl == c)[0] for c in uc}
    means = []
    for _ in range(n_boot):
        pick = rng.choice(uc, len(uc))
        means.append(float(D[np.concatenate([idx[c] for c in pick])].mean()))
    return [float(np.percentile(means, 1.25)), float(np.percentile(means, 98.75))]


def s1(M: np.ndarray, strata: np.ndarray, cl: np.ndarray, B: int, rng) -> dict:
    if len(groups_of(strata)) == 0:
        return {"n": int(len(strata)), "note": "no stratum with >= 2 facts"}
    obs, null = perm_mean_D(M, strata, B, rng)
    D = per_fact_D(M, strata)
    return {"n": int((~np.isnan(D)).sum()), "mean_D": obs, "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "p": float((1 + (null >= obs).sum()) / (B + 1)), "ci97.5_cluster": cluster_boot_mean(D, cl, rng)}


def retrieval(M: np.ndarray, strata: np.ndarray, B: int, rng) -> dict:
    gs = groups_of(strata)
    n = sum(len(g) for g in gs)
    top1 = mrr = chance = 0.0
    null_top1 = np.zeros(B)
    null_mrr = np.zeros(B)
    for g in gs:
        S = M[np.ix_(g, g)]
        k = len(g)
        rank = 1 + (S > np.diag(S)[:, None]).sum(1)
        top1 += (rank == 1).sum()
        mrr += (1.0 / rank).sum()
        chance += 1.0  # expected top-1 hits in the stratum = k * (1/k)
        P = derangements(k, B, rng)
        val = S[np.arange(k)[None, :], P]
        r = 1 + (S[None, :, :] > val[:, :, None]).sum(2)
        null_top1 += (r == 1).sum(1)
        null_mrr += (1.0 / r).sum(1)
    null_top1 /= n
    null_mrr /= n
    return {"n": n, "top1": top1 / n, "mrr": mrr / n, "chance_top1": chance / n, "null_mrr_mean": float(null_mrr.mean()),
            "p_top1": float((1 + (null_top1 >= top1 / n).sum()) / (B + 1)), "p_mrr": float((1 + (null_mrr >= mrr / n).sum()) / (B + 1))}


def spearman_strat(x: np.ndarray, y: np.ndarray, strata: np.ndarray, B: int, rng) -> dict:
    """Average-rank Spearman within strata (ranks demeaned per stratum), pooled; stratified permutation p (two-sided)."""
    gs = groups_of(strata)
    rx = [avg_rank(x[g]) - avg_rank(x[g]).mean() for g in gs]
    ry = [avg_rank(y[g]) - avg_rank(y[g]).mean() for g in gs]
    X, Y = np.concatenate(rx), np.concatenate(ry)
    rho = float((X * Y).sum() / np.sqrt((X ** 2).sum() * (Y ** 2).sum()))
    null = np.zeros(B)
    for b in range(B):
        Ys = np.concatenate([r[rng.permutation(len(r))] for r in ry])
        null[b] = (X * Ys).sum() / np.sqrt((X ** 2).sum() * (Ys ** 2).sum())
    return {"rho": rho, "p_two_sided": float((1 + (np.abs(null) >= abs(rho)).sum()) / (B + 1)), "n": int(len(X))}


def ols_cr1(X: np.ndarray, y: np.ndarray, cl: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """OLS with cluster-robust (CR1) covariance; pseudo-inverse guards collinearity. Returns beta, cov, rank."""
    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ X.T @ y
    u = y - X @ beta
    meat = np.zeros((X.shape[1],) * 2)
    uc = np.unique(cl)
    for c in uc:
        m = cl == c
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    G, N, K = len(uc), len(y), int(np.linalg.matrix_rank(X))
    adj = (G / max(G - 1, 1)) * ((N - 1) / max(N - K, 1))
    return beta, adj * XtXi @ meat @ XtXi, K


def sum_to_zero(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    levs = sorted(set(labels))
    if len(levs) < 2:
        return np.zeros((len(labels), 0)), []
    C = np.zeros((len(labels), len(levs) - 1))
    for j, lv in enumerate(levs[:-1]):
        C[:, j] = (labels == lv).astype(float) - (labels == levs[-1]).astype(float)
    return C, levs


def pieces(tok, ids) -> set[str]:
    out = set()
    for t in ids:
        s = tok.decode([int(t)]).strip().lower()
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
        if s:
            out.add(s)
    return out


# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--tag", default="confound")
    a = ap.parse_args()
    B = a.perm
    out = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}"
    out.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    res: dict = {"n_perm": B, "seed": SEED, "delta": DELTA}
    for m, (r1, r2, f1, f2) in SETS.items():
        rng = np.random.default_rng(SEED)
        tok = AutoTokenizer.from_pretrained(TOK_DIR[m])
        arr, recs, facts, rdir = {}, {}, {}, {}
        for r in (r1, r2):
            A, R = load(Path(r))
            arr.update(A); recs.update(R)
            for k in R:
                rdir[k] = Path(r)
        for f in (f1, f2):
            for x in json.loads(Path(f).read_text(encoding="utf8"))["facts"]:
                facts[x["id"]] = {lg: x["lang"][lg] for lg in ("en", "es", "hi")} | {"relation": x["relation"]}
        ids = sorted({f for f, _ in recs if all((f, lg) in recs and is_valid(recs[(f, lg)]) for lg in ("en", "es", "hi"))})
        rel = np.array([facts[i]["relation"] for i in ids])
        cnt = {r: int((rel == r).sum()) for r in set(rel)}
        strata = np.array([r if cnt[r] >= 10 else "other" for r in rel])
        PE = np.stack([arr[(i, "en")]["mlp"] for i in ids]); PS = np.stack([arr[(i, "es")]["mlp"] for i in ids])
        keep = (np.linalg.norm(PE, axis=1) > 1e-8) & (np.linalg.norm(PS, axis=1) > 1e-8)
        excluded = [i for i, k in zip(ids, keep) if not k]
        ids = [i for i, k in zip(ids, keep) if k]; rel, strata, PE, PS = rel[keep], strata[keep], PE[keep], PS[keep]
        n = len(ids)
        cl = np.array(clusters(facts, ids))
        # token-level covariates
        cls_, J, LT, NR, SD, AI, sim = [], [], [], [], [], [], []
        for i in ids:
            te = np.load(rdir[(i, "en")] / "grids" / f"{i}_en.npz"); ts = np.load(rdir[(i, "es")] / "grids" / f"{i}_es.npz")
            ie, is_ = te["tokens"].tolist(), ts["tokens"].tolist()
            se, ss = te["span"].tolist(), ts["span"].tolist()
            ip = ie[:se[-1] + 1] == is_[:ss[-1] + 1]
            nr = len(se) == len(ss)
            cls_.append("IP" if ip else ("EQ" if nr else "OT"))
            pe, ps = pieces(tok, [ie[k] for k in se]), pieces(tok, [is_[k] for k in ss])
            J.append(len(pe & ps) / max(len(pe | ps), 1))
            LT.append(float(ie[se[-1]] == is_[ss[-1]])); NR.append(float(nr)); SD.append(abs(len(se) - len(ss)))
            AI.append(float(mnorm(facts[i]["en"]["target"]) == mnorm(facts[i]["es"]["target"])))
            sim.append(difflib.SequenceMatcher(None, accnorm(facts[i]["en"]["subject"]), accnorm(facts[i]["es"]["subject"])).ratio())
        cls_ = np.array(cls_); J, LT, NR, SD, AI, sim = map(np.array, (J, LT, NR, SD, AI, sim))
        TE = {lg: np.array([recs[(i, lg)]["score_clean"] - recs[(i, lg)]["score_corr"] for i in ids]) for lg in ("en", "es")}
        lte = {lg: np.log(TE[lg]) - np.log(TE[lg]).mean() for lg in TE}
        o: dict = {"n": n, "excluded_zero_norm": excluded, "strata": {s: int((strata == s).sum()) for s in sorted(set(strata))},
                   "prefix_class_counts": {c: int((cls_ == c).sum()) for c in ("IP", "EQ", "OT")},
                   "prefix_class_by_stratum": {s: {c: int(((strata == s) & (cls_ == c)).sum()) for c in ("IP", "EQ", "OT")} for s in sorted(set(strata))},
                   "n_J0": int((J == 0).sum()), "n_NR0": int((NR == 0).sum()), "n_AI": int(AI.sum()), "n_LT": int(LT.sum())}
        # profiles
        Uraw_E, Uraw_S = center(PE, strata), center(PS, strata)                 # A12 S1 (unnormalised relation-centred)
        UE, US = unit(PE), unit(PS)
        NE, NS = center(UE, strata), center(US, strata)                         # NC
        PjE, PjS = project_out(UE, strata), project_out(US, strata)             # secondary NC variant
        Mraw, MNC, Mproj = cosm(Uraw_E, Uraw_S), cosm(NE, NS), cosm(PjE, PjS)
        ev = strata == "event_year"
        sets = {"all": np.ones(n, bool), "informative": (cls_ != "IP") & ~ev, "event_year_only": ev,
                "IP": cls_ == "IP", "EQ": cls_ == "EQ", "OT": cls_ == "OT", "non_event_year": ~ev}
        o["S1"] = {}
        for nm, mask in sets.items():
            if mask.sum() < 3:
                o["S1"][nm] = {"n": int(mask.sum()), "note": "too few"}; continue
            ix = np.where(mask)[0]
            sub = {"NC": s1(MNC[np.ix_(ix, ix)], strata[ix], cl[ix], B, rng), "raw_centred_A12": s1(Mraw[np.ix_(ix, ix)], strata[ix], cl[ix], B, rng)}
            if nm in ("all", "informative"):
                sub["NC_projection"] = s1(Mproj[np.ix_(ix, ix)], strata[ix], cl[ix], B, rng)
                sub["retrieval_NC"] = retrieval(MNC[np.ix_(ix, ix)], strata[ix], B, rng)
                sub["retrieval_raw_centred"] = retrieval(Mraw[np.ix_(ix, ix)], strata[ix], B, rng)
            o["S1"][nm] = sub
        # per prefix class within strata with >= 5 facts of that class
        o["S1_class_by_stratum"] = {}
        for c in ("IP", "EQ", "OT"):
            for s in sorted(set(strata)):
                mask = (cls_ == c) & (strata == s)
                if mask.sum() >= 5:
                    ix = np.where(mask)[0]
                    o["S1_class_by_stratum"][f"{c}/{s}"] = s1(MNC[np.ix_(ix, ix)], strata[ix], cl[ix], B, rng)
        # depth bands (thirds), on all and informative
        L = PE.shape[1]; cuts = [0, L // 3, 2 * L // 3, L]
        o["S3_bands"] = {}
        for bi, bn in enumerate(("early", "middle", "late")):
            sl = slice(cuts[bi], cuts[bi + 1])
            bE, bS = center(unit(PE[:, sl]), strata), center(unit(PS[:, sl]), strata)
            Mb = cosm(bE, bS)
            o["S3_bands"][bn] = {"layers": [cuts[bi], cuts[bi + 1] - 1]}
            for nm in ("all", "informative"):
                ix = np.where(sets[nm])[0]
                o["S3_bands"][bn][nm] = s1(Mb[np.ix_(ix, ix)], strata[ix], cl[ix], B, rng)
        # S1b magnitude diagnostic
        o["S1b_TE_spearman_within_stratum"] = spearman_strat(TE["en"], TE["es"], strata, B, rng)
        # S4 as written in A12 (D on raw relation-centred profiles; treatment-coded relation => intercept at reference relation)
        D_raw = per_fact_D(Mraw, strata); D_nc = per_fact_D(MNC, strata)
        ok = ~np.isnan(D_raw)
        relT = np.stack([(strata == s).astype(float) for s in sorted(set(strata))[1:]], 1)
        X4 = np.column_stack([np.ones(n), sim, np.abs(np.array(SD)), AI, np.minimum(lte["en"], lte["es"]), relT])[ok]
        b4, V4, k4 = ols_cr1(X4, D_raw[ok], cl[ok])
        o["S4_A12_as_written"] = {"names": ["intercept(ref relation)", "name_sim_difflib", "abs_span_diff", "answer_identical", "log_min_TE(centred)"] + [f"rel={s}" for s in sorted(set(strata))[1:]],
                                  "beta": b4.tolist(), "se": np.sqrt(np.diag(V4)).tolist(), "rank": k4, "n": int(ok.sum())}
        # S4a on informative facts
        inf = (cls_ != "IP") & ~ev & ~np.isnan(D_nc)
        S0, levs = sum_to_zero(strata[inf])
        cov = {"J": J, "LT": LT, "AI": AI, "NR": NR, "SD": SD.astype(float), "TE_en": lte["en"], "TE_es": lte["es"]}
        names = [k for k in cov if np.ptp(cov[k][inf]) > 0]
        dropped = [k for k in cov if k not in names]
        X = np.column_stack([np.ones(inf.sum()), *[cov[k][inf] for k in names], S0])
        b, V, kk = ols_cr1(X, D_nc[inf], cl[inf])
        c = np.zeros(X.shape[1]); c[0] = 1.0
        if "SD" in names:
            c[1 + names.index("SD")] = 1.0
        pred = float(c @ b); se = float(np.sqrt(c @ V @ c))
        # bootstrap of the target quantity (cluster bootstrap over entity clusters)
        ii = np.where(inf)[0]; uc = np.unique(cl[ii]); cidx = {u: np.where(cl[ii] == u)[0] for u in uc}
        bt = []
        for _ in range(1000):
            pk = np.concatenate([cidx[u] for u in rng.choice(uc, len(uc))])
            try:
                bb = np.linalg.pinv(X[pk]) @ D_nc[inf][pk]
                bt.append(float(c @ bb))
            except Exception:
                pass
        o["S4a_informative"] = {"names": ["intercept"] + names + [f"stratum_{l}" for l in levs[:-1]], "dropped_constant": dropped,
                                "beta": b.tolist(), "se": np.sqrt(np.diag(V)).tolist(), "rank": kk, "n": int(inf.sum()),
                                "target_pred_D": pred, "target_se": se, "target_ci97.5": [pred - Z975 * se, pred + Z975 * se],
                                "target_boot_ci97.5": [float(np.percentile(bt, 1.25)), float(np.percentile(bt, 98.75))] if bt else None,
                                "extrapolation": bool((J[inf] == 0).sum() < 10), "n_J0_informative": int((J[inf] == 0).sum()), "n_NR0_informative": int((NR[inf] == 0).sum())}
        # decision rules
        def verdict(lo, hi):
            return "present" if lo > 0 else ("absent" if hi < DELTA else "unresolved")
        s1i = o["S1"]["informative"]["NC"]
        o["decision"] = {"S1a_informative": verdict(*s1i["ci97.5_cluster"]) if "ci97.5_cluster" in s1i else None,
                         "S4a_target": verdict(*o["S4a_informative"]["target_ci97.5"])}
        # A12 sesoi (one third of Gemma-style mean D on all facts, this model's own, reported only)
        o["A12_sesoi_this_model"] = o["S1"]["all"]["raw_centred_A12"]["mean_D"] / 3
        # A11 corrections: Spearman with average ranks (within stratum), raw cosine and NC D vs difflib name sim
        Eraw, Sraw = unit(PE), unit(PS)
        diag = (Eraw * Sraw).sum(1)
        o["C2_corrected_spearman"] = {"sim_vs_raw_cosine": spearman_strat(sim, diag, strata, 5000, rng),
                                      "sim_vs_NC_D": spearman_strat(sim[~np.isnan(D_nc)], D_nc[~np.isnan(D_nc)], strata[~np.isnan(D_nc)], 5000, rng)}
        # predictions Q1-Q5
        q1 = []
        for s in sorted(set(strata)):
            mi, mo = (strata == s) & (cls_ == "IP") & ~np.isnan(D_nc), (strata == s) & (cls_ == "OT") & ~np.isnan(D_nc)
            if mi.sum() >= 2 and mo.sum() >= 2:
                q1.append({"stratum": s, "mean_D_IP": float(D_nc[mi].mean()), "mean_D_OT": float(D_nc[mo].mean()), "n_IP": int(mi.sum()), "n_OT": int(mo.sum())})
        o["Q"] = {"Q1_IP_vs_OT_within_stratum": q1,
                  "Q2_NC_all_D_less_than_raw_all_D": bool(o["S1"]["all"]["NC"]["mean_D"] < o["S1"]["all"]["raw_centred_A12"]["mean_D"]),
                  "Q3_S1a_CI_includes_0": bool(s1i["ci97.5_cluster"][0] <= 0 <= s1i["ci97.5_cluster"][1]) if "ci97.5_cluster" in s1i else None,
                  "Q4_S4a_CI_includes_0": bool(o["S4a_informative"]["target_ci97.5"][0] <= 0 <= o["S4a_informative"]["target_ci97.5"][1]),
                  "Q5_TE_corr_positive": bool(o["S1b_TE_spearman_within_stratum"]["rho"] > 0)}
        res[m] = o
        print(m, n, o["prefix_class_counts"], o["decision"], {k: v for k, v in o["Q"].items() if k != "Q1_IP_vs_OT_within_stratum"})
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf8")
    (out / "config.json").write_text(json.dumps({"sets": SETS, "n_perm": B, "seed": SEED, "amendments": ["A12", "A12a"], "delta": DELTA}, indent=1), encoding="utf8")
    print("wrote", out)


if __name__ == "__main__":
    main()
