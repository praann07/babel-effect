"""A13 (E2) analysis of the cross-language subject interchange (CPU). Follows docs/HYPOTHESIS.md A13 exactly."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from confound_overlap import TOK_DIR, cosm, center, ols_cr1, per_fact_D, sum_to_zero, unit
from common import subject_span

SEED = 20261024
COND = {"same": 0, "cross": 1, "self": 2, "ctrl": 3}


def load_run(run: str):
    z = np.load(Path(run) / "effects.npz")
    meta = json.loads((Path(run) / "meta.json").read_text())
    pairs = meta["pairs"]
    seen, pilot = [], set()
    for p in pairs:  # the first 10 distinct (target, source) pairs are the debugging pilot of the same seed: excluded
        k = (p["target"], p["source"])
        if k not in seen:
            seen.append(k)
    pilot = set(seen[:10])
    keep = np.array([(p["target"], p["source"]) not in pilot for p in pairs])
    return z["delta"][keep], z["flip"][keep], [p for p, k in zip(pairs, keep) if k], meta


def band_and_gate(D: np.ndarray, F: np.ndarray, site: int) -> dict:
    mu = D[:, COND["same"], site].mean(0)
    band = np.where(mu >= 0.5 * mu.max())[0]
    ls = int(mu.argmax())
    return {"band": [int(band.min()), int(band.max())], "band_layers": band.tolist(), "l_star": ls, "mean_same_at_lstar": float(mu[ls]),
            "flip_same_at_lstar": float(F[:, COND["same"], site, ls].mean()), "gate_P1": bool(F[:, COND["same"], site, ls].mean() >= 0.6 and mu[ls] >= 2.0)}


def ratio_ci(D, src, band, site, rng, n_boot=2000):
    """I = sum_B mean Delta_cross / sum_B mean Delta_same; cluster bootstrap over source entities."""
    def stat(ix):
        return D[ix][:, COND["cross"], site][:, band].mean(0).sum() / D[ix][:, COND["same"], site][:, band].mean(0).sum()
    I = float(stat(np.arange(len(D))))
    ents = np.unique(src)
    by = {e: np.where(src == e)[0] for e in ents}
    bs = []
    for _ in range(n_boot):
        ix = np.concatenate([by[e] for e in rng.choice(ents, len(ents))])
        bs.append(stat(ix))
    return {"I": I, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))],
            "n_pairs": int(len(D)), "n_source_entities": int(len(ents))}


def verdict(r: dict) -> str:
    if r["ci95"][0] >= 0.7:
        return "P2 SHARED"
    if r["ci90"][1] <= 0.3:
        return "P3 LANGUAGE-SPECIFIC"
    return "P4 PARTIAL"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-run", required=True); ap.add_argument("--gemma-run", required=True)
    ap.add_argument("--llama-seeds", default="analysis/run_20261024_p3seeds_10240109_llama"); ap.add_argument("--gemma-seeds", default="analysis/run_20261024_p3seeds_10240109_gemma2")
    ap.add_argument("--tag", default="e2_analysis")
    ap.add_argument("--langs", default="en,es", help="A15: language pair, e.g. en,hi (P5/P6 are skipped for pairs other than en,es)")
    ap.add_argument("--models", default="llama,gemma2")
    a = ap.parse_args()
    l0, l1 = a.langs.split(",")
    DIRS = (f"{l0}2{l1}", f"{l1}2{l0}")
    ES = a.langs == "en,es"
    from transformers import AutoTokenizer
    out = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}"; out.mkdir(parents=True, exist_ok=False)
    res: dict = {}
    for m, run, sruns in [x for x in (("llama", a.llama_run, a.llama_seeds), ("gemma2", a.gemma_run, a.gemma_seeds)) if x[0] in a.models.split(",")]:
        rng = np.random.default_rng(SEED)
        D, F, pairs, meta = load_run(run)
        facts = {f["id"]: f for f in json.loads(Path(meta.get("facts_file", f"data/v2/e2_facts_{m}.json")).read_text(encoding="utf8"))["facts"]}
        tok = AutoTokenizer.from_pretrained(TOK_DIR[m])
        spandiff = {}
        for sid in {p["source"] for p in pairs}:
            L = facts[sid]["lang"]
            n = {lg: len(subject_span(tok, L[lg]["prompt"], L[lg]["subject"])[1]) for lg in (l0, l1)}
            spandiff[sid] = abs(n[l0] - n[l1])
        o: dict = {"n_pair_directions": len(pairs), "n_facts": len(facts), "run": run}
        for direction in [d for d in DIRS if any(p["direction"] == d for p in pairs)]:
            sel = np.array([p["direction"] == direction for p in pairs])
            Dd, Fd = D[sel], F[sel]
            pd_ = [p for p, k in zip(pairs, sel) if k]
            src = np.array([p["source"] for p in pd_])
            conf = np.array([p["jaccard_source"] == 0 for p in pd_])
            od: dict = {"n_all": int(sel.sum()), "n_confirmatory": int(conf.sum())}
            for si, site in enumerate(("A", "B")):
                g = band_and_gate(Dd, Fd, si)
                band = np.array(g["band_layers"])
                r = {"gate": g, "confirmatory": ratio_ci(Dd[conf], src[conf], band, si, rng) if conf.sum() >= 5 else None, "all_pairs": ratio_ci(Dd, src, band, si, rng)}
                if r["confirmatory"]:
                    r["verdict"] = verdict(r["confirmatory"])
                    r["confirmatory"]["n_source_entities"] = int(len(np.unique(src[conf])))
                Dc = Dd[conf] if conf.sum() >= 5 else Dd
                same_l, cross_l = Dc[:, COND["same"], si].mean(0), Dc[:, COND["cross"], si].mean(0)
                r["I_by_layer_confirmatory"] = (cross_l / np.where(np.abs(same_l) > 0.5, same_l, np.nan)).round(3).tolist()
                r["mean_delta_by_layer_confirmatory"] = {c: Dc[:, ci, si].mean(0).round(3).tolist() for c, ci in COND.items()}
                if conf.sum() >= 5:
                    r["flip_at_lstar_confirmatory"] = {c: float(Fd[conf][:, ci, si, g["l_star"]].mean()) for c, ci in COND.items()}
                if conf.sum() >= 5:  # A16 descriptive extras: floor, interpretability of the ratio, per-relation ratio
                    r["floor_25_sources_ok"] = bool(r["confirmatory"]["n_source_entities"] >= 25)
                    r["ratio_interpretable_cross_flip_ge_0.2"] = bool(r["flip_at_lstar_confirmatory"]["cross"] >= 0.2)
                    relc = np.array([p["relation"] for p, k in zip(pd_, conf) if k]); Dcf, srcc = Dd[conf], src[conf]
                    r["by_relation_descriptive"] = {rl: ratio_ci(Dcf[relc == rl], srcc[relc == rl], band, si, np.random.default_rng(1), n_boot=500) for rl in sorted(set(relc)) if (relc == rl).sum() >= 5}
                od[f"site_{site}"] = r
            if direction == DIRS[0] and ES:  # covariate model (site A), per-pair ratio
                band = np.array(od["site_A"]["gate"]["band_layers"])
                num, den = Dd[:, COND["cross"], 0][:, band].mean(1), Dd[:, COND["same"], 0][:, band].mean(1)
                ok = den > 0.5
                rel = np.array([p["relation"] for p in pd_])
                S0, levs = sum_to_zero(rel[ok])
                Jv = np.array([p["jaccard_source"] for p in pd_])[ok]; SDv = np.array([spandiff[s] for s in src[ok]], float)
                X = np.column_stack([np.ones(ok.sum()), Jv, SDv, S0]); yv = (num / den)[ok]
                b, V, rk = ols_cr1(X, yv, src[ok])
                c0 = np.zeros(X.shape[1]); c0[0] = 1.0; c1 = c0.copy(); c1[1] = 1.0
                se0, se1 = float(np.sqrt(c0 @ V @ c0)), float(np.sqrt(c1 @ V @ c1))
                od["covariate_model"] = {"n": int(ok.sum()), "names": ["intercept", "jaccard", "abs_span_diff"] + [f"rel_{l}" for l in levs[:-1]], "beta": b.tolist(), "se": np.sqrt(np.diag(V)).tolist(),
                                         "pred_ratio_at_J0": float(c0 @ b), "se_J0": se0, "pred_ratio_at_J1": float(c1 @ b), "se_J1": se1,
                                         "P5_lexical": bool(float(c1 @ b) - float(c0 @ b) > 0.2)}
                # P6 metric validity: per-source-fact mean ratio vs the A13a NC increment of that fact
                from seeds_analysis import meta as mta
                ids, strata, cl, cls_, _ = mta(m)
                Pn = {lg: np.stack([np.load(Path(sruns) / "grids" / f"{i}_{lg}.npz")["ind"].mean(0) for i in ids]) for lg in ("en", "es")}
                Dn = per_fact_D(cosm(center(unit(Pn["en"]), strata), center(unit(Pn["es"]), strata)), strata)
                dn = dict(zip(ids, Dn))
                per_src: dict = {}
                for s_, r_ in zip(src[ok], num[ok] / den[ok]):
                    per_src.setdefault(s_, []).append(r_)
                common = [s_ for s_ in per_src if s_ in dn and not np.isnan(dn[s_])]
                if len(common) >= 8:
                    x, y = np.array([np.mean(per_src[s_]) for s_ in common]), np.array([dn[s_] for s_ in common])
                    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
                    rho = float(np.corrcoef(rx, ry)[0, 1])
                    null = [np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(5000)]
                    od["P6_validity"] = {"n_facts": len(common), "spearman": rho, "p_two_sided": float((1 + (np.abs(null) >= abs(rho)).sum()) / 5001), "P6_pass": bool(rho > 0.2)}
                else:
                    od["P6_validity"] = {"n_facts": len(common), "note": "too few facts in common"}
            o[direction] = od
        res[m] = o
        print(m, {d: (o[d]["n_confirmatory"], o[d]["site_A"].get("verdict"), o[d]["site_A"]["gate"]["gate_P1"]) for d in DIRS if d in o})
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf8")
    (out / "config.json").write_text(json.dumps(vars(a) | {"amendment": "A13"}, indent=1), encoding="utf8")
    print("wrote", out)


if __name__ == "__main__":
    main()
