"""A16 descriptive check (pre-registered): HI->EN site-A ratio on pairs whose target AND source fact are not in the A15 kept set.
Band and l* are those of the full A16 confirmatory set (so the ratio is comparable); CI by source-entity cluster bootstrap (2000)."""
import json, sys, datetime as dt
from pathlib import Path
import numpy as np
from e2_analysis import load_run, ratio_ci, band_and_gate, COND, SEED

res = {}
for m in ("llama", "gemma2"):
    run = sorted(Path("analysis").glob(f"run_*_e2h2full_A_{m}"))[-1]
    a15 = {f["id"] for f in json.loads(Path(f"data/v2/e2h_facts_{m}.json").read_text(encoding="utf8"))["facts"]}
    D, F, pairs, meta = load_run(str(run))
    sel = np.array([p["direction"] == "hi2en" for p in pairs]); D, F = D[sel], F[sel]; pd_ = [p for p, k in zip(pairs, sel) if k]
    conf = np.array([p["jaccard_source"] == 0 for p in pd_])
    band = np.array(band_and_gate(D, F, 0)["band_layers"]); ls = band_and_gate(D, F, 0)["l_star"]
    src = np.array([p["source"] for p in pd_])
    out = np.array([p["source"] not in a15 and p["target"] not in a15 for p in pd_]) & conf
    r = ratio_ci(D[out], src[out], band, 0, np.random.default_rng(SEED))
    r["flip_at_lstar"] = {c: float(F[out][:, ci, 0, ls].mean()) for c, ci in COND.items()}
    full = ratio_ci(D[conf], src[conf], band, 0, np.random.default_rng(SEED))
    r["full_I"] = full["I"]; r["abs_diff_from_full"] = abs(r["I"] - full["I"])
    res[m] = r; print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "flip_at_lstar"}, r["flip_at_lstar"])
out = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_e2h2_outside_A15"; out.mkdir(parents=True, exist_ok=False)
(out / "results.json").write_text(json.dumps(res, indent=1))
