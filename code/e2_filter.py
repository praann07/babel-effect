"""A13 (E2) fact filter: keep candidates the model answers correctly in BOTH EN and ES with the fixed prefix templates (Hindi not required).

First 30 candidates per relation by id are tuning facts (template accuracy check only; excluded from evaluation).
Writes data/v2/e2_facts_<model>.json and a write-once run dir with raw generations.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODELS
from filter_facts_v2 import generate
from matching import strict_match

TEMPLATES = {  # fixed in docs/HYPOTHESIS.md A13 before this was run
    "capital_of": ("The capital of {s} is", "La capital de {s} es la ciudad de"),
    "capital_country": ("The country whose capital is {s} is", "El país cuya capital es {s} es"),
    "landmark_city": ("The city where {s} is located is", "La ciudad donde está {s} es"),
    "landmark_country": ("The country where {s} is located is", "El país donde está {s} es"),
    "city_country": ("The country where {s} is located is", "El país donde está {s} es"),
}
CANDS = {  # A13c: candidate templates per relation and language (first = the A13 template); used with --select (Gemma only)
    "capital_of": (["The capital of {s} is", "The capital of {s} is the city of", "The capital city of {s} is called"], ["La capital de {s} es la ciudad de", "La capital de {s} se llama"]),
    "capital_country": (["The country whose capital is {s} is", "The country whose capital is {s} is called", "The country with the capital city {s} is called"],
                        ["El país cuya capital es {s} es", "El país cuya capital es {s} se llama"]),
    "landmark_city": (["The city where {s} is located is", "The city where {s} is located is called", "You can visit {s} in the city of"],
                      ["La ciudad donde está {s} es", "Se puede visitar {s} en la ciudad de", "La ciudad donde está {s} se llama"]),
    "landmark_country": (["The country where {s} is located is", "The country where {s} is located is called", "You can visit {s} in the country of"],
                         ["El país donde está {s} es", "El país donde está {s} se llama", "Se puede visitar {s} en el país de"]),
}
CANDS["city_country"] = CANDS["landmark_country"]
POOLS = {"llama": ["data/v2/facts_candidates.json", "data/v2/facts_candidates_extra_llama.json"],
         "gemma2": ["data/v2/facts_candidates_gemma2.json", "data/v2/facts_candidates_extra_gemma2.json"]}
N_TUNE = 30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama", "gemma2"])
    ap.add_argument("--tag", default="e2filter")
    ap.add_argument("--select", action="store_true", help="A13c: select templates per relation/language on the tuning facts")
    a = ap.parse_args()
    spec = MODELS[a.model]
    run = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}_{a.model}"
    run.mkdir(parents=True, exist_ok=False)
    cands = [f for p in POOLS[a.model] for f in json.loads(Path(p).read_text(encoding="utf8"))["facts"] if f["relation"] in TEMPLATES]
    cands.sort(key=lambda f: (f["relation"], f["id"]))
    tok = AutoTokenizer.from_pretrained(spec["dir"]); tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec["dir"], torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    if a.select:
        tmpl: dict = {}
        for r in TEMPLATES:
            fs = [f for f in cands if f["relation"] == r][:N_TUNE]
            for k, lg in enumerate(("en", "es")):
                accs = []
                for tp in CANDS[r][k]:
                    g = generate(model, tok, [tp.format(s=f["lang"][lg]["subject"]) for f in fs])
                    accs.append(sum(strict_match(x, [f["lang"][lg]["object"], *f["lang"][lg]["object_alts"]]) is not None for x, f in zip(g, fs)) / len(fs))
                best = max(range(len(accs)), key=lambda i: (accs[i], -i))
                tmpl.setdefault(r, [None, None])[k] = CANDS[r][k][best]
                print(r, lg, dict(zip(CANDS[r][k], accs)), "->", CANDS[r][k][best], flush=True)
        TEMPLATES.update({r: tuple(v) for r, v in tmpl.items()})
    prompts = {(f["id"], lg): TEMPLATES[f["relation"]][k].format(s=f["lang"][lg]["subject"]) for f in cands for k, lg in enumerate(("en", "es"))}
    keys = list(prompts)
    gens = dict(zip(keys, generate(model, tok, [prompts[k] for k in keys])))
    (run / "generations.json").write_text(json.dumps({f"{k[0]}|{k[1]}": {"prompt": prompts[k], "gen": gens[k]} for k in keys}, ensure_ascii=False, indent=1), encoding="utf8")
    idx: dict[str, int] = {}
    kept, table = [], {}
    for f in cands:
        r = f["relation"]; i = idx.get(r, 0); idx[r] = i + 1
        split = "tune" if i < N_TUNE else "eval"
        rec, ok = {}, True
        for lg in ("en", "es"):
            L = f["lang"][lg]
            m = strict_match(gens[(f["id"], lg)], [L["object"], *L["object_alts"]])
            rec[lg] = {"subject": L["subject"], "object": L["object"], "object_alts": L["object_alts"], "prompt": prompts[(f["id"], lg)], "target": m}
            ok &= m is not None
            t = table.setdefault(r, {"tune_n": 0, "tune_en": 0, "tune_es": 0, "eval_n": 0, "eval_both": 0})
            if split == "tune":
                t["tune_en" if lg == "en" else "tune_es"] += int(m is not None)
        t["tune_n" if split == "tune" else "eval_n"] += 1
        if split == "eval" and ok:
            t["eval_both"] += 1
            kept.append({"id": f["id"], "relation": r, "seed": f["seed"], "lang": rec})
    for r, t in table.items():
        t["template_ok"] = bool(t["tune_n"] and t["tune_en"] / t["tune_n"] >= 0.5 and t["tune_es"] / t["tune_n"] >= 0.5)
    kept = [f for f in kept if table[f["relation"]]["template_ok"]]
    out = {"model": spec["tl_name"], "templates": TEMPLATES, "summary": table, "n_kept": len(kept), "facts": kept}
    Path(f"data/v2/e2_facts_{a.model}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8")
    (run / "accuracy.json").write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1)); print("kept", len(kept))


if __name__ == "__main__":
    main()
