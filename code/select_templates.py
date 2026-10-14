"""A7: choose 3 answer-first templates per (relation, language) from a pool, by strict accuracy on a tuning set.

Tuning set = the 30 lowest-id candidate facts of the relation. Writes data/v2/templates.json and a write-once run dir.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_facts_v2 import fmt
from common import LANGS, MODELS
from filter_facts_v2 import generate
from matching import strict_match

POOL = {
    "capital_of": {
        "en": ["The capital of {s} is", "The capital city of {s} is", "{S}'s capital is", "{S} has its capital in the city of",
               "The capital of {s} is the city of", "The seat of government of {s} is in"],
        "es": ["La capital de {s} es", "La ciudad capital de {s} es", "{S} tiene su capital en", "{S} tiene como capital a",
               "La capital de {s} es la ciudad de", "El gobierno de {s} tiene su sede en"],
        "hi": ["{s} की राजधानी", "{s} की राजधानी है", "{s} की राजधानी का नाम", "{s} देश की राजधानी", "{s} की राजधानी शहर",
               "{s} की राजधानी वाला शहर"],
    },
    "capital_country": {
        "en": ["{S} is the capital of", "{S} is the capital city of", "{S} is the capital of the country of",
               "The country whose capital is {s} is", "{S} serves as the capital of", "The nation with {s} as its capital is"],
        "es": ["{S} es la capital de", "{S} es la ciudad capital de", "El país cuya capital es {s} es", "{S} es la capital del país de",
               "{S} es la sede del gobierno de", "La nación que tiene a {s} como capital es"],
        "hi": ["{s} इस देश की राजधानी है:", "{s} जिस देश की राजधानी है, उसका नाम", "{s} की राजधानी वाला देश",
               "{s} इस देश की राजधानी है,", "वह देश जिसकी राजधानी {s} है:", "{s} राजधानी है इस देश की:"],
    },
    "landmark_city": {
        "en": ["{S} is located in the city of", "{S} can be found in the city of", "{S} stands in the city of", "{S} is in the city of",
               "You can visit {s} in the city of", "{S} is a famous landmark in the city of"],
        "es": ["{S} se encuentra en la ciudad de", "{S} se halla en la ciudad de", "{S} está en la ciudad de",
               "{S} se ubica en la ciudad de", "Se puede visitar {s} en la ciudad de", "{S} es un lugar famoso de la ciudad de"],
        "hi": ["{s} इस शहर में स्थित है:", "{s} शहर में स्थित है, जिसका नाम", "{s} स्थित है शहर", "{s} इस शहर में है:",
               "{s} मशहूर है शहर", "{s} यहाँ स्थित है:"],
    },
    "invention_inventor": {
        "en": ["The inventor of {s} is", "{S} was invented by", "{S} was created by", "The person who invented {s} is",
               "The person credited with inventing {s} is", "{S} is credited to"],
        "es": ["El inventor de {s} es", "El creador de {s} es", "La persona que inventó {s} es", "El inventor de {s} fue",
               "La invención de {s} se atribuye a", "Quien inventó {s} fue"],
        "hi": ["{s} का आविष्कार इस व्यक्ति ने किया था:", "{s} के आविष्कारक का नाम है:",
               "{s} का आविष्कार करने वाले व्यक्ति का नाम", "{s} का आविष्कारक", "{s} का आविष्कार किया था", "{s} के आविष्कारक हैं"],
    },
    "event_year": {
        "en": ["{S} took place in the year", "{S} happened in", "{S} occurred in the year", "The year of {s} was",
               "{S} was in the year", "{S} took place in"],
        "es": ["{S} tuvo lugar en el año", "{S} ocurrió en", "{S} sucedió en el año", "El año de {s} fue", "{S} fue en el año",
               "{S} tuvo lugar en"],
        "hi": ["{s} सन्", "{s} वर्ष", "{s} साल", "{s} का वर्ष", "{s} के वर्ष", "{s} की तारीख़ का वर्ष"],
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama", choices=list(MODELS))
    ap.add_argument("--n-tune", type=int, default=30)
    a = ap.parse_args()
    spec = MODELS[a.model]
    run = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_templates_{a.model}"
    run.mkdir(parents=True, exist_ok=False)
    cands = json.loads(Path("data/v2/facts_candidates.json").read_text(encoding="utf8"))["facts"]
    tok = AutoTokenizer.from_pretrained(spec["dir"])
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec["dir"], torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()

    chosen, log = {}, {}
    for rel, langs in POOL.items():
        tune = sorted((f for f in cands if f["relation"] == rel), key=lambda f: f["id"])[:a.n_tune]
        for lg in LANGS:
            accs = []
            for t in langs[lg]:
                prompts = [fmt(t, lg, f["lang"][lg]["subject"]) for f in tune]
                gens = generate(model, tok, prompts)
                ok = sum(strict_match(g, [f["lang"][lg]["object"], *f["lang"][lg]["object_alts"]]) is not None
                         for g, f in zip(gens, tune))
                accs.append(ok / len(tune))
            order = sorted(range(len(accs)), key=lambda i: (-accs[i], i))[:3]
            chosen.setdefault(rel, {})[lg] = [langs[lg][i] for i in order]
            log[f"{rel}/{lg}"] = [{"template": langs[lg][i], "acc": accs[i]} for i in range(len(accs))]
            print(rel, lg, [(langs[lg][i], round(accs[i], 2)) for i in order], flush=True)
    Path("data/v2/templates.json").write_text(json.dumps(
        {"model_used_for_selection": spec["tl_name"], "n_tune": a.n_tune, "templates": chosen}, ensure_ascii=False, indent=1),
        encoding="utf8")
    (run / "pool_accuracy.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
