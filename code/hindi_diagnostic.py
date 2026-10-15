"""F2: is the weak Hindi result a tokenizer artifact? Hindi accuracy and EN-HI overlap against Hindi token length (CPU only).

(a) canonical-phrasing accuracy of every candidate fact by object / subject token count, per language;
(b) per-fact EN-HI profile cosine of the valid facts against the Hindi target token count and subject span length.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from circuit_overlap import cos_matrix, is_valid, load
from common import LANGS, MODELS, subject_span
from matching import strict_match


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 5 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hindi_diag")
    a = ap.parse_args()
    cfg = {"llama": {"cands": "data/v2/facts_candidates.json", "gens": "analysis/run_20261014_p1v2b_llama/generations.json",
                     "trace": "analysis/run_20261015_p3trace_v2_llama"},
           "gemma2": {"cands": "data/v2/facts_candidates_gemma2.json", "gens": "analysis/run_20261015_p1v2c_gemma2/generations.json",
                      "trace": "analysis/run_20261015_p3trace_v2c_gemma2"}}
    out = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}"
    out.mkdir(parents=True, exist_ok=False)
    res: dict = {}
    for m, c in cfg.items():
        tok = AutoTokenizer.from_pretrained(MODELS[m]["dir"])
        cands = json.loads(Path(c["cands"]).read_text(encoding="utf8"))["facts"]
        gens = json.loads(Path(c["gens"]).read_text(encoding="utf8"))
        rows = []
        for f in cands:
            for lg in LANGS:
                L = f["lang"][lg]
                ok = strict_match(gens[f["id"]][lg]["0"]["gen"], [L["object"], *L["object_alts"]]) is not None
                n_obj = len(tok(" " + L["object"], add_special_tokens=False)["input_ids"])
                try:
                    n_subj = len(subject_span(tok, L["prompts"][0], L["subject"])[1])
                except ValueError:
                    n_subj = None
                rows.append({"id": f["id"], "relation": f["relation"], "lang": lg, "ok": ok, "n_obj_tokens": n_obj, "n_subj_tokens": n_subj})
        df = pd.DataFrame(rows)
        df["obj_bin"] = pd.cut(df["n_obj_tokens"], [0, 3, 6, 100], labels=["1-3", "4-6", "7+"])
        acc = df.groupby(["lang", "obj_bin"], observed=True)["ok"].agg(["mean", "count"]).round(3)
        tab = {f"{lg}/{b}": {"acc": float(r["mean"]), "n": int(r["count"])} for (lg, b), r in acc.iterrows()}
        sp = {lg: spearman(df[df.lang == lg]["n_obj_tokens"].tolist(), df[df.lang == lg]["ok"].astype(float).tolist()) for lg in LANGS}
        # (b) per-fact EN-HI profile cosine vs Hindi target length
        arr, recs = load(Path(c["trace"]))
        valid = sorted({f for f, _ in recs if all(is_valid(recs[(f, lg)]) for lg in LANGS)})
        cos = [float(cos_matrix(arr[(f, "en")]["mlp"][None], arr[(f, "hi")]["mlp"][None])[0, 0]) for f in valid]
        hi_len = [recs[(f, "hi")]["n_target"] for f in valid]
        hi_span = [len(recs[(f, "hi")]["span"]) for f in valid]
        res[m] = {"accuracy_by_object_tokens": tab, "spearman_correct_vs_object_tokens": sp,
                  "valid_n": len(valid), "spearman_enhi_cosine_vs_hi_target_tokens": spearman(hi_len, cos),
                  "spearman_enhi_cosine_vs_hi_subject_span": spearman(hi_span, cos),
                  "hi_target_tokens_median": float(np.median(hi_len)), "hi_span_median": float(np.median(hi_span))}
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
