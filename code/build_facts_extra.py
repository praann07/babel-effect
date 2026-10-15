"""F4 (docs/HYPOTHESIS.md A9): extra non-copyable relations -> data/v2/facts_candidates_extra.json.

landmark_country: the 81 landmarks of build_facts_v2 -> their country.   city_country: non-capital cities -> country.
Answers are written differently in EN/ES/HI, so they cannot be copied across languages. Templates come from a templates json
(per model) if it contains these relations, else from the defaults below.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_facts import CAPITALS, LANDMARKS
from build_facts_v2 import LANDMARKS2, fmt

LANGS = ("en", "es", "hi")

EXTRA_COUNTRIES = {
    "the United Kingdom": ("the United Kingdom", "el Reino Unido", "यूनाइटेड किंगडम"),
    "the United States": ("the United States", "Estados Unidos", "संयुक्त राज्य अमेरिका"),
    "the United Arab Emirates": ("the United Arab Emirates", "los Emiratos Árabes Unidos", "संयुक्त अरब अमीरात"),
    "Vatican City": ("Vatican City", "la Ciudad del Vaticano", "वेटिकन सिटी"),
    "Singapore": ("Singapore", "Singapur", "सिंगापुर"),
    "the Czech Republic": ("the Czech Republic", "la República Checa", "चेक गणराज्य"),
    "the Netherlands": ("the Netherlands", "los Países Bajos", "नीदरलैंड"),
}


def country(name: str) -> tuple[str, str, str]:
    if name in EXTRA_COUNTRIES:
        return EXTRA_COUNTRIES[name]
    for (c, _k) in CAPITALS:
        if c[0] == name:
            return c
    raise KeyError(name)


# country of each landmark, in the order LANDMARKS then LANDMARKS2
LM_COUNTRY = ["France", "Italy", "the United Kingdom", "the United States", "India", "Spain", "Germany", "Greece", "India", "Australia",
              "the United Arab Emirates", "Russia", "France", "Italy", "India", "Turkey", "Spain", "Spain", "Brazil", "the United States",
              "the United States", "Canada", "Malaysia", "China", "Belgium", "Russia", "the Czech Republic", "the United Kingdom", "Italy", "Italy",
              "the United Kingdom", "the United Kingdom", "the United States", "the United States", "India", "India", "India", "India", "India",
              # LANDMARKS2
              "Italy", "Vatican City", "Turkey", "Germany", "Germany", "Germany", "Greece", "the United Kingdom", "the United Kingdom",
              "the United Kingdom", "France", "France", "France", "Spain", "Spain", "Spain", "the United States", "the United States",
              "the United States", "the United States", "Mexico", "Peru", "China", "China", "China", "Japan", "Japan", "South Korea", "Singapore",
              "Cambodia", "India", "India", "India", "India", "India", "India", "India", "India", "India", "India", "Egypt", "South Africa"]

# (city (en, es, hi), country english name)
CITIES = [
    (("Barcelona", "Barcelona", "बार्सिलोना"), "Spain"), (("Seville", "Sevilla", "सेविला"), "Spain"),
    (("Valencia", "Valencia", "वैलेंसिया"), "Spain"), (("Munich", "Múnich", "म्यूनिख"), "Germany"),
    (("Hamburg", "Hamburgo", "हैम्बर्ग"), "Germany"), (("Frankfurt", "Fráncfort", "फ्रैंकफर्ट"), "Germany"),
    (("Milan", "Milán", "मिलान"), "Italy"), (("Naples", "Nápoles", "नेपल्स"), "Italy"), (("Florence", "Florencia", "फ्लोरेंस"), "Italy"),
    (("Venice", "Venecia", "वेनिस"), "Italy"), (("Turin", "Turín", "ट्यूरिन"), "Italy"), (("Marseille", "Marsella", "मार्सेल"), "France"),
    (("Nice", "Niza", "नीस"), "France"), (("Toulouse", "Toulouse", "टूलूज़"), "France"),
    (("Manchester", "Mánchester", "मैनचेस्टर"), "the United Kingdom"), (("Liverpool", "Liverpool", "लिवरपूल"), "the United Kingdom"),
    (("Birmingham", "Birmingham", "बर्मिंघम"), "the United Kingdom"), (("Glasgow", "Glasgow", "ग्लासगो"), "the United Kingdom"),
    (("Mumbai", "Bombay", "मुंबई"), "India"), (("Kolkata", "Calcuta", "कोलकाता"), "India"), (("Chennai", "Chennai", "चेन्नई"), "India"),
    (("Bangalore", "Bangalore", "बेंगलुरु"), "India"), (("Hyderabad", "Hyderabad", "हैदराबाद"), "India"),
    (("Ahmedabad", "Ahmedabad", "अहमदाबाद"), "India"), (("Pune", "Pune", "पुणे"), "India"), (("Lucknow", "Lucknow", "लखनऊ"), "India"),
    (("Jaipur", "Jaipur", "जयपुर"), "India"), (("Istanbul", "Estambul", "इस्तांबुल"), "Turkey"),
    (("Sydney", "Sídney", "सिडनी"), "Australia"), (("Melbourne", "Melbourne", "मेलबर्न"), "Australia"),
    (("Brisbane", "Brisbane", "ब्रिस्बेन"), "Australia"), (("Perth", "Perth", "पर्थ"), "Australia"),
    (("Toronto", "Toronto", "टोरंटो"), "Canada"), (("Vancouver", "Vancouver", "वैंकूवर"), "Canada"), (("Montreal", "Montreal", "मॉन्ट्रियल"), "Canada"),
    (("New York", "Nueva York", "न्यूयॉर्क"), "the United States"), (("Los Angeles", "Los Ángeles", "लॉस एंजिल्स"), "the United States"),
    (("Chicago", "Chicago", "शिकागो"), "the United States"), (("Houston", "Houston", "ह्यूस्टन"), "the United States"),
    (("Boston", "Boston", "बोस्टन"), "the United States"), (("Miami", "Miami", "मियामी"), "the United States"),
    (("Seattle", "Seattle", "सिएटल"), "the United States"), (("San Francisco", "San Francisco", "सैन फ़्रांसिस्को"), "the United States"),
    (("Rio de Janeiro", "Río de Janeiro", "रियो डी जेनेरो"), "Brazil"), (("São Paulo", "São Paulo", "साओ पाउलो"), "Brazil"),
    (("Shanghai", "Shanghái", "शंघाई"), "China"), (("Guangzhou", "Cantón", "गुआंगझोउ"), "China"), (("Shenzhen", "Shenzhen", "शेनझेन"), "China"),
    (("Chengdu", "Chengdú", "चेंगदू"), "China"), (("Osaka", "Osaka", "ओसाका"), "Japan"), (("Kyoto", "Kioto", "क्योटो"), "Japan"),
    (("Yokohama", "Yokohama", "योकोहामा"), "Japan"), (("Busan", "Busan", "बुसान"), "South Korea"),
    (("Karachi", "Karachi", "कराची"), "Pakistan"), (("Lahore", "Lahore", "लाहौर"), "Pakistan"),
    (("Alexandria", "Alejandría", "अलेक्ज़ेंड्रिया"), "Egypt"), (("Casablanca", "Casablanca", "कैसाब्लांका"), "Morocco"),
    (("Johannesburg", "Johannesburgo", "जोहान्सबर्ग"), "South Africa"), (("Cape Town", "Ciudad del Cabo", "केप टाउन"), "South Africa"),
    (("Lagos", "Lagos", "लागोस"), "Nigeria"), (("Saint Petersburg", "San Petersburgo", "सेंट पीटर्सबर्ग"), "Russia"),
    (("Kraków", "Cracovia", "क्राकोव"), "Poland"), (("Porto", "Oporto", "पोर्टो"), "Portugal"), (("Zurich", "Zúrich", "ज्यूरिख"), "Switzerland"),
    (("Geneva", "Ginebra", "जिनेवा"), "Switzerland"), (("Gothenburg", "Gotemburgo", "गोथेनबर्ग"), "Sweden"),
    (("Bergen", "Bergen", "बर्गन"), "Norway"), (("Salzburg", "Salzburgo", "साल्ज़बर्ग"), "Austria"),
    (("Antwerp", "Amberes", "एंटवर्प"), "Belgium"), (("Rotterdam", "Róterdam", "रॉटरडैम"), "the Netherlands"),
    (("Guadalajara", "Guadalajara", "ग्वादालाजारा"), "Mexico"), (("Medellín", "Medellín", "मेडेलिन"), "Colombia"),
    (("Dubai", "Dubái", "दुबई"), "the United Arab Emirates"), (("Abu Dhabi", "Abu Dabi", "अबू धाबी"), "the United Arab Emirates"),
    (("Kyiv", "Kiev", "कीव"), "Ukraine"),
]

DEFAULT_TEMPLATES = {
    "landmark_country": {"en": ["{S} is located in the country of", "{S} is in the country of", "{S} can be found in the country of"],
                         "es": ["{S} se encuentra en el país de", "{S} está en el país de", "{S} se halla en el país de"],
                         "hi": ["{s} इस देश में स्थित है:", "{s} इस देश में है:", "{s} जिस देश में स्थित है, उसका नाम"]},
    "city_country": {"en": ["{S} is a city in the country of", "{S} is in the country of", "{S} is located in the country of"],
                     "es": ["{S} es una ciudad del país de", "{S} está en el país de", "{S} se encuentra en el país de"],
                     "hi": ["{s} इस देश का शहर है:", "{s} इस देश में है:", "{s} जिस देश में स्थित है, उसका नाम"]},
}


def make(rel: str, idx: int, subj: tuple[str, ...], obj: tuple[str, ...], templates: dict) -> dict:
    fid = f"{rel}_{idx:03d}"
    seed = int(hashlib.sha256(fid.encode()).hexdigest(), 16) % (2**31)
    f = {"id": fid, "relation": rel, "seed": seed, "lang": {}}
    for k, lg in enumerate(LANGS):
        f["lang"][lg] = {"subject": subj[k], "object": obj[k], "object_alts": [],
                         "prompts": [fmt(t, lg, subj[k]) for t in templates[rel][lg]]}
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default=None, help="templates json containing these relations (else defaults)")
    ap.add_argument("--out", default="data/v2/facts_candidates_extra.json")
    a = ap.parse_args()
    templates = dict(DEFAULT_TEMPLATES)
    if a.templates and Path(a.templates).exists():
        templates |= {k: v for k, v in json.loads(Path(a.templates).read_text(encoding="utf8"))["templates"].items() if k in templates}
    lms = LANDMARKS + LANDMARKS2
    assert len(lms) == len(LM_COUNTRY), (len(lms), len(LM_COUNTRY))
    facts = [make("landmark_country", i, l, country(LM_COUNTRY[i]), templates) for i, (l, _c) in enumerate(lms)]
    facts += [make("city_country", i, c, country(cn), templates) for i, (c, cn) in enumerate(CITIES)]
    Path(a.out).write_text(json.dumps({"version": 2, "note": "A9 extra non-copyable relations", "facts": facts}, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print(len(facts), "candidates:", {r: sum(f["relation"] == r for f in facts) for r in templates})


if __name__ == "__main__":
    main()
