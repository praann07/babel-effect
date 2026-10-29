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

POOL["landmark_country"] = {
    "en": ["{S} is located in the country of", "{S} is in the country of", "{S} can be found in the country of", "{S} stands in",
           "You can visit {s} in", "{S} is a famous landmark in"],
    "es": ["{S} se encuentra en el país de", "{S} está en el país de", "{S} se halla en el país de", "{S} se encuentra en",
           "Se puede visitar {s} en", "{S} es un lugar famoso de"],
    "hi": ["{s} इस देश में स्थित है:", "{s} इस देश में है:", "{s} जिस देश में स्थित है, उसका नाम", "{s} स्थित है देश", "{s} मशहूर है देश", "{s} यहाँ स्थित है:"],
}
POOL["city_country"] = {
    "en": ["{S} is a city in the country of", "{S} is in the country of", "{S} is located in the country of", "{S} is a city in",
           "The country where {s} is located is", "{S} lies in"],
    "es": ["{S} es una ciudad del país de", "{S} está en el país de", "{S} se encuentra en el país de", "{S} es una ciudad de",
           "El país donde está {s} es", "{S} se halla en"],
    "hi": ["{s} इस देश का शहर है:", "{s} इस देश में है:", "{s} जिस देश में स्थित है, उसका नाम", "{s} शहर है देश", "{s} किस देश में है:", "{s} स्थित है देश"],
}


from build_facts_a23 import POOL_A23  # A23 non-geography relations (only used when selected via --relations / --cands)

POOL.update(POOL_A23)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama", choices=list(MODELS))
    ap.add_argument("--n-tune", type=int, default=30)
    ap.add_argument("--cands", default="data/v2/facts_candidates.json", help="candidate file (tuning facts come from here)")
    ap.add_argument("--relations", default=None, help="comma list; only these relations are selected and merged into the existing templates file")
    a = ap.parse_args()
    spec = MODELS[a.model]
    run = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_templates_{a.model}{'_extra' if a.relations else ''}"
    run.mkdir(parents=True, exist_ok=False)
    cands = json.loads(Path(a.cands).read_text(encoding="utf8"))["facts"]
    tok = AutoTokenizer.from_pretrained(spec["dir"])
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec["dir"], torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()

    chosen, log = {}, {}
    only = set(a.relations.split(",")) if a.relations else None
    for rel, langs in POOL.items():
        if only and rel not in only:
            continue
        tune = sorted((f for f in cands if f["relation"] == rel), key=lambda f: f["id"])[:a.n_tune]
        if not tune:
            continue  # relation not in this candidate file
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
    out_path = Path("data/v2/templates.json" if a.model == "llama" else f"data/v2/templates_{a.model}.json")
    merged = json.loads(out_path.read_text(encoding="utf8"))["templates"] if (only and out_path.exists()) else {}
    merged.update(chosen)
    out_path.write_text(json.dumps({"model_used_for_selection": spec["tl_name"], "n_tune": a.n_tune, "templates": merged},
                                   ensure_ascii=False, indent=1), encoding="utf8")
    (run / "pool_accuracy.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
