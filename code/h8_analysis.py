"""F5 / H8 (docs/HYPOTHESIS.md A5, A9): within-language across-phrasing profile similarity versus cross-language similarity.

Per fact: d = mean over languages and phrasings (1, 2) of cos(profile of phrasing 0, profile of that phrasing) minus the mean of the
three cross-language cosines of the canonical phrasings. H8 predicts median d > 0. Bootstrap 95% CI over facts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from circuit_overlap import is_valid, load

LANGS = ("en", "es", "hi")


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def profile(run: Path, fid: str, lg: str, ph: int) -> np.ndarray:
    z = np.load(run / "grids" / (f"{fid}_{lg}.npz" if ph == 0 else f"{fid}_{lg}_p{ph}.npz"))
    return z["mlp"][:, int(z["span"][-1])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-run", required=True)
    ap.add_argument("--h8-run", required=True)
    ap.add_argument("--tag", default="h8")
    ap.add_argument("--model", default="")
    ap.add_argument("--seed", type=int, default=20261012)
    a = ap.parse_args()
    main_run, h8_run = Path(a.main_run), Path(a.h8_run)
    _, recs = load(main_run)
    h8 = json.loads((h8_run / "records.json").read_text(encoding="utf8"))
    ok = {}
    for r in h8:
        ok.setdefault(r["id"], set())
        if "error" not in r:
            ok[r["id"]].add((r["lang"], r["phrasing"]))
    need = {(lg, p) for lg in LANGS for p in (1, 2)}
    facts = [f for f, s in sorted(ok.items()) if s >= need and all(is_valid(recs[(f, lg)]) for lg in LANGS)]
    rows = []
    for f in facts:
        within = [cos(profile(main_run, f, lg, 0), profile(h8_run, f, lg, p)) for lg in LANGS for p in (1, 2)]
        cross = [cos(profile(main_run, f, x, 0), profile(main_run, f, y, 0)) for x, y in combinations(LANGS, 2)]
        rows.append({"id": f, "within": float(np.mean(within)), "cross": float(np.mean(cross)),
                     "within_by_lang": {lg: float(np.mean([cos(profile(main_run, f, lg, 0), profile(h8_run, f, lg, p)) for p in (1, 2)])) for lg in LANGS}})
    d = np.array([r["within"] - r["cross"] for r in rows])
    rng = np.random.default_rng(a.seed)
    boot = np.array([np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]) if len(d) else np.array([np.nan])
    res = {"model": a.model, "n_facts": len(rows), "median_within": float(np.median([r["within"] for r in rows])) if rows else None,
           "median_cross": float(np.median([r["cross"] for r in rows])) if rows else None,
           "median_d": float(np.median(d)) if len(d) else None, "ci95_d": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "frac_facts_d_positive": float(np.mean(d > 0)) if len(d) else None,
           "median_within_by_lang": {lg: float(np.median([r["within_by_lang"][lg] for r in rows])) for lg in LANGS} if rows else None}
    out = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}_{a.model}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "results.json").write_text(json.dumps({"summary": res, "rows": rows}, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
