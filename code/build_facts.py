"""P1 (part 1): build candidate fact triples with EN/ES/HI surface forms and prompts -> data/facts_candidates.json.

Filtering to facts the model recalls correctly in all three languages is done later by filter_facts.py
(needs the model). Entity translations were written by hand; Hindi prompt templates carry two variants and the
variant is selected by a pre-declared rule (higher whole-set accuracy), never per fact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# (country (en, es, hi), capital (en, es, hi))
CAPITALS = [
    (("France", "Francia", "फ्रांस"), ("Paris", "París", "पेरिस")),
    (("Germany", "Alemania", "जर्मनी"), ("Berlin", "Berlín", "बर्लिन")),
    (("Italy", "Italia", "इटली"), ("Rome", "Roma", "रोम")),
    (("Spain", "España", "स्पेन"), ("Madrid", "Madrid", "मैड्रिड")),
    (("Portugal", "Portugal", "पुर्तगाल"), ("Lisbon", "Lisboa", "लिस्बन")),
    (("Japan", "Japón", "जापान"), ("Tokyo", "Tokio", "टोक्यो")),
    (("China", "China", "चीन"), ("Beijing", "Pekín", "बीजिंग")),
    (("India", "India", "भारत"), ("New Delhi", "Nueva Delhi", "नई दिल्ली")),
    (("Russia", "Rusia", "रूस"), ("Moscow", "Moscú", "मॉस्को")),
    (("Brazil", "Brasil", "ब्राज़ील"), ("Brasília", "Brasilia", "ब्रासीलिया")),
    (("Argentina", "Argentina", "अर्जेंटीना"), ("Buenos Aires", "Buenos Aires", "ब्यूनस आयर्स")),
    (("Mexico", "México", "मेक्सिको"), ("Mexico City", "Ciudad de México", "मेक्सिको सिटी")),
    (("Canada", "Canadá", "कनाडा"), ("Ottawa", "Ottawa", "ओटावा")),
    (("Australia", "Australia", "ऑस्ट्रेलिया"), ("Canberra", "Canberra", "कैनबरा")),
    (("Greece", "Grecia", "ग्रीस"), ("Athens", "Atenas", "एथेंस")),
    (("Turkey", "Turquía", "तुर्की"), ("Ankara", "Ankara", "अंकारा")),
    (("Sweden", "Suecia", "स्वीडन"), ("Stockholm", "Estocolmo", "स्टॉकहोम")),
    (("Norway", "Noruega", "नॉर्वे"), ("Oslo", "Oslo", "ओस्लो")),
    (("Finland", "Finlandia", "फ़िनलैंड"), ("Helsinki", "Helsinki", "हेलसिंकी")),
    (("Denmark", "Dinamarca", "डेनमार्क"), ("Copenhagen", "Copenhague", "कोपेनहेगन")),
    (("Poland", "Polonia", "पोलैंड"), ("Warsaw", "Varsovia", "वारसॉ")),
    (("Austria", "Austria", "ऑस्ट्रिया"), ("Vienna", "Viena", "वियना")),
    (("Switzerland", "Suiza", "स्विट्ज़रलैंड"), ("Bern", "Berna", "बर्न")),
    (("Belgium", "Bélgica", "बेल्जियम"), ("Brussels", "Bruselas", "ब्रसेल्स")),
    (("Ireland", "Irlanda", "आयरलैंड"), ("Dublin", "Dublín", "डबलिन")),
    (("Peru", "Perú", "पेरू"), ("Lima", "Lima", "लीमा")),
    (("Chile", "Chile", "चिली"), ("Santiago", "Santiago", "सैंटियागो")),
    (("Colombia", "Colombia", "कोलंबिया"), ("Bogotá", "Bogotá", "बोगोटा")),
    (("Venezuela", "Venezuela", "वेनेज़ुएला"), ("Caracas", "Caracas", "काराकास")),
    (("Thailand", "Tailandia", "थाईलैंड"), ("Bangkok", "Bangkok", "बैंकॉक")),
    (("Vietnam", "Vietnam", "वियतनाम"), ("Hanoi", "Hanói", "हनोई")),
    (("Indonesia", "Indonesia", "इंडोनेशिया"), ("Jakarta", "Yakarta", "जकार्ता")),
    (("Pakistan", "Pakistán", "पाकिस्तान"), ("Islamabad", "Islamabad", "इस्लामाबाद")),
    (("Bangladesh", "Bangladés", "बांग्लादेश"), ("Dhaka", "Daca", "ढाका")),
    (("Nepal", "Nepal", "नेपाल"), ("Kathmandu", "Katmandú", "काठमांडू")),
    (("Sri Lanka", "Sri Lanka", "श्रीलंका"), ("Colombo", "Colombo", "कोलंबो")),
    (("Iran", "Irán", "ईरान"), ("Tehran", "Teherán", "तेहरान")),
    (("Iraq", "Irak", "इराक"), ("Baghdad", "Bagdad", "बग़दाद")),
    (("Saudi Arabia", "Arabia Saudita", "सऊदी अरब"), ("Riyadh", "Riad", "रियाद")),
    (("Kenya", "Kenia", "केन्या"), ("Nairobi", "Nairobi", "नैरोबी")),
    (("Nigeria", "Nigeria", "नाइजीरिया"), ("Abuja", "Abuya", "अबुजा")),
    (("Ethiopia", "Etiopía", "इथियोपिया"), ("Addis Ababa", "Adís Abeba", "अदीस अबाबा")),
    (("Morocco", "Marruecos", "मोरक्को"), ("Rabat", "Rabat", "रबात")),
    (("Algeria", "Argelia", "अल्जीरिया"), ("Algiers", "Argel", "अल्जीयर्स")),
    (("South Africa", "Sudáfrica", "दक्षिण अफ़्रीका"), ("Pretoria", "Pretoria", "प्रिटोरिया")),
    (("Ghana", "Ghana", "घाना"), ("Accra", "Acra", "अक्रा")),
    (("Hungary", "Hungría", "हंगरी"), ("Budapest", "Budapest", "बुडापेस्ट")),
    (("Romania", "Rumania", "रोमानिया"), ("Bucharest", "Bucarest", "बुखारेस्ट")),
    (("Ukraine", "Ucrania", "यूक्रेन"), ("Kyiv", "Kiev", "कीव")),
    (("Bulgaria", "Bulgaria", "बुल्गारिया"), ("Sofia", "Sofía", "सोफिया")),
    (("Serbia", "Serbia", "सर्बिया"), ("Belgrade", "Belgrado", "बेलग्रेड")),
    (("Croatia", "Croacia", "क्रोएशिया"), ("Zagreb", "Zagreb", "ज़ाग्रेब")),
    (("Iceland", "Islandia", "आइसलैंड"), ("Reykjavik", "Reikiavik", "रेक्यावीक")),
    (("Mongolia", "Mongolia", "मंगोलिया"), ("Ulaanbaatar", "Ulán Bator", "उलानबटार")),
    (("Afghanistan", "Afganistán", "अफ़ग़ानिस्तान"), ("Kabul", "Kabul", "काबुल")),
    (("Cambodia", "Camboya", "कंबोडिया"), ("Phnom Penh", "Nom Pen", "नोम पेन")),
    (("South Korea", "Corea del Sur", "दक्षिण कोरिया"), ("Seoul", "Seúl", "सियोल")),
    (("Jordan", "Jordania", "जॉर्डन"), ("Amman", "Amán", "अम्मान")),
    (("Lebanon", "Líbano", "लेबनान"), ("Beirut", "Beirut", "बेरूत")),
    (("Syria", "Siria", "सीरिया"), ("Damascus", "Damasco", "दमिश्क")),
    (("Libya", "Libia", "लीबिया"), ("Tripoli", "Trípoli", "त्रिपोली")),
    (("Uruguay", "Uruguay", "उरुग्वे"), ("Montevideo", "Montevideo", "मोंटेवीडियो")),
    (("Ecuador", "Ecuador", "इक्वाडोर"), ("Quito", "Quito", "क्विटो")),
    (("Paraguay", "Paraguay", "पैराग्वे"), ("Asunción", "Asunción", "असुनसियोन")),
    (("Jamaica", "Jamaica", "जमैका"), ("Kingston", "Kingston", "किंग्स्टन")),
]

# (landmark (en, es, hi), city (en, es, hi))
LANDMARKS = [
    (("the Eiffel Tower", "la Torre Eiffel", "एफ़िल टॉवर"), ("Paris", "París", "पेरिस")),
    (("the Colosseum", "el Coliseo", "कोलोसियम"), ("Rome", "Roma", "रोम")),
    (("Big Ben", "el Big Ben", "बिग बेन"), ("London", "Londres", "लंदन")),
    (("the Statue of Liberty", "la Estatua de la Libertad", "स्टैच्यू ऑफ़ लिबर्टी"), ("New York", "Nueva York", "न्यूयॉर्क")),
    (("the Taj Mahal", "el Taj Mahal", "ताज महल"), ("Agra", "Agra", "आगरा")),
    (("the Sagrada Familia", "la Sagrada Familia", "सग्रादा फ़ैमिलिया"), ("Barcelona", "Barcelona", "बार्सिलोना")),
    (("the Brandenburg Gate", "la Puerta de Brandeburgo", "ब्रांडेनबर्ग गेट"), ("Berlin", "Berlín", "बर्लिन")),
    (("the Acropolis", "la Acrópolis", "एक्रोपोलिस"), ("Athens", "Atenas", "एथेंस")),
    (("the Red Fort", "el Fuerte Rojo", "लाल क़िला"), ("Delhi", "Delhi", "दिल्ली")),
    (("the Sydney Opera House", "la Ópera de Sídney", "सिडनी ओपेरा हाउस"), ("Sydney", "Sídney", "सिडनी")),
    (("the Burj Khalifa", "el Burj Khalifa", "बुर्ज ख़लीफ़ा"), ("Dubai", "Dubái", "दुबई")),
    (("the Kremlin", "el Kremlin", "क्रेमलिन"), ("Moscow", "Moscú", "मॉस्को")),
    (("the Louvre", "el Louvre", "लूव्र"), ("Paris", "París", "पेरिस")),
    (("the Leaning Tower", "la Torre inclinada", "पीसा की मीनार"), ("Pisa", "Pisa", "पीसा")),
    (("the Charminar", "el Charminar", "चारमीनार"), ("Hyderabad", "Hyderabad", "हैदराबाद")),
    (("the Hagia Sophia", "Santa Sofía", "हागिया सोफ़िया"), ("Istanbul", "Estambul", "इस्तांबुल")),
    (("the Alhambra", "la Alhambra", "अलहम्ब्रा"), ("Granada", "Granada", "ग्रेनेडा")),
    (("the Prado Museum", "el Museo del Prado", "प्राडो संग्रहालय"), ("Madrid", "Madrid", "मैड्रिड")),
    (("Christ the Redeemer", "el Cristo Redentor", "क्राइस्ट द रिडीमर"), ("Rio de Janeiro", "Río de Janeiro", "रियो डी जनेरियो")),
    (("the Space Needle", "la Space Needle", "स्पेस नीडल"), ("Seattle", "Seattle", "सिएटल")),
    (("the Golden Gate Bridge", "el Puente Golden Gate", "गोल्डन गेट ब्रिज"), ("San Francisco", "San Francisco", "सैन फ़्रांसिस्को")),
    (("the CN Tower", "la Torre CN", "सीएन टावर"), ("Toronto", "Toronto", "टोरंटो")),
    (("the Petronas Towers", "las Torres Petronas", "पेट्रोनास टावर्स"), ("Kuala Lumpur", "Kuala Lumpur", "कुआला लंपुर")),
    (("the Forbidden City", "la Ciudad Prohibida", "निषिद्ध नगर"), ("Beijing", "Pekín", "बीजिंग")),
    (("the Atomium", "el Atomium", "एटोमियम"), ("Brussels", "Bruselas", "ब्रसेल्स")),
    (("the Hermitage Museum", "el Museo del Hermitage", "हर्मिटेज संग्रहालय"), ("Saint Petersburg", "San Petersburgo", "सेंट पीटर्सबर्ग")),
    (("the Charles Bridge", "el Puente de Carlos", "चार्ल्स ब्रिज"), ("Prague", "Praga", "प्राग")),
    (("Edinburgh Castle", "el Castillo de Edimburgo", "एडिनबरा कैसल"), ("Edinburgh", "Edimburgo", "एडिनबरा")),
    (("the Trevi Fountain", "la Fontana de Trevi", "ट्रेवी फ़ाउंटेन"), ("Rome", "Roma", "रोम")),
    (("the Rialto Bridge", "el Puente de Rialto", "रियाल्टो पुल"), ("Venice", "Venecia", "वेनिस")),
    (("Westminster Abbey", "la Abadía de Westminster", "वेस्टमिंस्टर एबे"), ("London", "Londres", "लंदन")),
    (("Tower Bridge", "el Tower Bridge", "टावर ब्रिज"), ("London", "Londres", "लंदन")),
    (("Times Square", "Times Square", "टाइम्स स्क्वायर"), ("New York", "Nueva York", "न्यूयॉर्क")),
    (("the Hollywood Sign", "el Cartel de Hollywood", "हॉलीवुड साइन"), ("Los Angeles", "Los Ángeles", "लॉस एंजिल्स")),
    (("the Golden Temple", "el Templo Dorado", "स्वर्ण मंदिर"), ("Amritsar", "Amritsar", "अमृतसर")),
    (("the Hawa Mahal", "el Hawa Mahal", "हवा महल"), ("Jaipur", "Jaipur", "जयपुर")),
    (("the Victoria Memorial", "el Memorial Victoria", "विक्टोरिया मेमोरियल"), ("Kolkata", "Calcuta", "कोलकाता")),
    (("the Meenakshi Temple", "el Templo de Meenakshi", "मीनाक्षी मंदिर"), ("Madurai", "Madurai", "मदुरै")),
    (("the Mysore Palace", "el Palacio de Mysore", "मैसूर पैलेस"), ("Mysore", "Mysore", "मैसूर")),
]

TEMPLATES = {
    "capital_of": {  # subject = country, object = capital
        "en": ["The capital of {s} is"],
        "es": ["La capital de {s} es"],
        "hi": ["{s} की राजधानी", "{s} की राजधानी है"],
    },
    "landmark_city": {  # subject = landmark, object = city
        "en": ["{s} is located in the city of"],
        "es": ["{s} se encuentra en la ciudad de"],
        "hi": ["{s} इस शहर में स्थित है:", "{s} शहर में स्थित है, जिसका नाम"],
    },
    "capital_country": {  # subject = capital, object = country
        "en": ["{s} is the capital of"],
        "es": ["{s} es la capital de"],
        "hi": ["{s} इस देश की राजधानी है:", "{s} जिस देश की राजधानी है, उसका नाम"],
    },
}
LANGS = ("en", "es", "hi")


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def make(rel: str, idx: int, subj: tuple[str, ...], obj: tuple[str, ...]) -> dict:
    fid = f"{rel}_{idx:03d}"
    seed = int(hashlib.sha256(fid.encode()).hexdigest(), 16) % (2**31)
    f = {"id": fid, "relation": rel, "seed": seed, "lang": {}}
    for k, lg in enumerate(LANGS):
        s = subj[k]
        # sentence-initial subject is capitalised for EN/ES templates that start with {s}
        prompts = [t.format(s=cap(s) if (t.startswith("{s}") and lg != "hi") else s) for t in TEMPLATES[rel][lg]]
        f["lang"][lg] = {"subject": s, "object": obj[k], "prompts": prompts}
    return f


def main() -> None:
    facts = [make("capital_of", i, c, k) for i, (c, k) in enumerate(CAPITALS)]
    facts += [make("landmark_city", i, l, c) for i, (l, c) in enumerate(LANDMARKS)]
    facts += [make("capital_country", i, k, c) for i, (c, k) in enumerate(CAPITALS[:50])]
    Path("data").mkdir(exist_ok=True)
    with open("data/facts_candidates.json", "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "note": "unfiltered candidates; see filter_facts.py", "facts": facts},
                  fh, ensure_ascii=False, indent=1)
    print(len(facts), "candidate facts;", {r: sum(f["relation"] == r for f in facts) for r in TEMPLATES})


if __name__ == "__main__":
    main()
