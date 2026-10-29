"""A23 candidates: non-geography relations -> data/v2/facts_candidates_a23.json (docs/HYPOTHESIS.md A23).

official_language: country -> language.   currency: country -> currency.   profession: famous person -> profession.
book_author: book -> author.
Answers are written differently in EN/ES/HI, so they cannot be copied across languages. These are CANDIDATES: only facts the model
answers correctly in all three languages survive the filter, and the kept ones go to the native-speaker audit sheet
(code/make_audit_sheet.py). Nothing here has been checked by a native speaker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_facts import CAPITALS
from build_facts_extra import EXTRA_COUNTRIES
from build_facts_v2 import fmt

LANGS = ("en", "es", "hi")

# ---- official_language: (country en name, language key)
LANG = {  # key -> (en, es, hi, [es alts], [hi alts])
    "Japanese": ("Japanese", "japonés", "जापानी"), "Chinese": ("Chinese", "chino", "चीनी"), "Russian": ("Russian", "ruso", "रूसी"),
    "Portuguese": ("Portuguese", "portugués", "पुर्तगाली"), "Spanish": ("Spanish", "español", "स्पेनिश"), "German": ("German", "alemán", "जर्मन"),
    "Italian": ("Italian", "italiano", "इतालवी"), "French": ("French", "francés", "फ़्रेंच"), "Greek": ("Greek", "griego", "यूनानी"),
    "Turkish": ("Turkish", "turco", "तुर्की"), "Swedish": ("Swedish", "sueco", "स्वीडिश"), "Norwegian": ("Norwegian", "noruego", "नॉर्वेजियाई"),
    "Danish": ("Danish", "danés", "डेनिश"), "Polish": ("Polish", "polaco", "पोलिश"), "Dutch": ("Dutch", "neerlandés", "डच"),
    "Hungarian": ("Hungarian", "húngaro", "हंगेरियाई"), "Romanian": ("Romanian", "rumano", "रोमानियाई"), "Ukrainian": ("Ukrainian", "ucraniano", "यूक्रेनी"),
    "Bulgarian": ("Bulgarian", "búlgaro", "बुल्गारियाई"), "Croatian": ("Croatian", "croata", "क्रोएशियाई"), "Serbian": ("Serbian", "serbio", "सर्बियाई"),
    "Icelandic": ("Icelandic", "islandés", "आइसलैंडिक"), "Mongolian": ("Mongolian", "mongol", "मंगोलियाई"), "Korean": ("Korean", "coreano", "कोरियाई"),
    "Thai": ("Thai", "tailandés", "थाई"), "Vietnamese": ("Vietnamese", "vietnamita", "वियतनामी"), "Persian": ("Persian", "persa", "फ़ारसी"),
    "Arabic": ("Arabic", "árabe", "अरबी"), "Hebrew": ("Hebrew", "hebreo", "हिब्रू"), "Czech": ("Czech", "checo", "चेक"), "Slovak": ("Slovak", "eslovaco", "स्लोवाक"),
    "Slovenian": ("Slovenian", "esloveno", "स्लोवेनियाई"), "Estonian": ("Estonian", "estonio", "एस्टोनियाई"), "Latvian": ("Latvian", "letón", "लातवियाई"),
    "Lithuanian": ("Lithuanian", "lituano", "लिथुआनियाई"), "Nepali": ("Nepali", "nepalí", "नेपाली"), "Bengali": ("Bengali", "bengalí", "बांग्ला"),
    "Indonesian": ("Indonesian", "indonesio", "इंडोनेशियाई"), "Albanian": ("Albanian", "albanés", "अल्बानियाई"), "Georgian": ("Georgian", "georgiano", "जॉर्जियाई"),
    "Armenian": ("Armenian", "armenio", "अर्मेनियाई"),
}
COUNTRY_LANG = [("Japan", "Japanese"), ("China", "Chinese"), ("Russia", "Russian"), ("Brazil", "Portuguese"), ("Portugal", "Portuguese"), ("Argentina", "Spanish"),
                ("Mexico", "Spanish"), ("Spain", "Spanish"), ("Chile", "Spanish"), ("Colombia", "Spanish"), ("Cuba", "Spanish"), ("Venezuela", "Spanish"),
                ("Uruguay", "Spanish"), ("Germany", "German"), ("Austria", "German"), ("Italy", "Italian"), ("France", "French"), ("Greece", "Greek"),
                ("Turkey", "Turkish"), ("Sweden", "Swedish"), ("Norway", "Norwegian"), ("Denmark", "Danish"), ("Poland", "Polish"), ("the Netherlands", "Dutch"),
                ("Hungary", "Hungarian"), ("Romania", "Romanian"), ("Ukraine", "Ukrainian"), ("Bulgaria", "Bulgarian"), ("Croatia", "Croatian"), ("Serbia", "Serbian"),
                ("Iceland", "Icelandic"), ("Mongolia", "Mongolian"), ("South Korea", "Korean"), ("Thailand", "Thai"), ("Vietnam", "Vietnamese"), ("Iran", "Persian"),
                ("Saudi Arabia", "Arabic"), ("Egypt", "Arabic"), ("Libya", "Arabic"), ("Israel", "Hebrew"), ("the Czech Republic", "Czech"), ("Slovakia", "Slovak"),
                ("Slovenia", "Slovenian"), ("Estonia", "Estonian"), ("Latvia", "Latvian"), ("Lithuania", "Lithuanian"), ("Nepal", "Nepali"), ("Bangladesh", "Bengali"),
                ("Indonesia", "Indonesian"), ("Albania", "Albanian"), ("Georgia", "Georgian"), ("Armenia", "Armenian")]

# ---- currency: (country, key)
CUR = {"yen": ("yen", "yen", "येन"), "yuan": ("yuan", "yuan", "युआन"), "rupee": ("rupee", "rupia", "रुपया"), "ruble": ("ruble", "rublo", "रूबल"),
       "peso": ("peso", "peso", "पेसो"), "dollar": ("dollar", "dólar", "डॉलर"), "pound": ("pound", "libra", "पाउंड"), "franc": ("franc", "franco", "फ़्रैंक"),
       "lira": ("lira", "lira", "लीरा"), "won": ("won", "won", "वॉन"), "dong": ("dong", "dong", "डोंग"), "taka": ("taka", "taka", "टका"),
       "riyal": ("riyal", "riyal", "रियाल"), "rand": ("rand", "rand", "रैंड"), "naira": ("naira", "naira", "नाइरा"), "shilling": ("shilling", "chelín", "शिलिंग"),
       "euro": ("euro", "euro", "यूरो"), "dirham": ("dirham", "dírham", "दिरहम"), "dinar": ("dinar", "dinar", "दीनार")}
COUNTRY_CUR = [("Japan", "yen"), ("China", "yuan"), ("India", "rupee"), ("Russia", "ruble"), ("Mexico", "peso"), ("Argentina", "peso"), ("Chile", "peso"),
               ("Colombia", "peso"), ("Canada", "dollar"), ("Australia", "dollar"), ("New Zealand", "dollar"), ("the United States", "dollar"),
               ("the United Kingdom", "pound"), ("Switzerland", "franc"), ("Turkey", "lira"), ("South Korea", "won"), ("Vietnam", "dong"), ("Bangladesh", "taka"),
               ("Saudi Arabia", "riyal"), ("South Africa", "rand"), ("Nigeria", "naira"), ("Kenya", "shilling"), ("Germany", "euro"), ("France", "euro"),
               ("Italy", "euro"), ("Spain", "euro"), ("Greece", "euro"), ("Ireland", "euro"), ("Morocco", "dirham"), ("the United Arab Emirates", "dirham"),
               ("Jordan", "dinar")]

# ---- profession: (person (en, es, hi), profession key)
PROF = {"physicist": ("physicist", "físico", "भौतिक विज्ञानी", [], ["भौतिकशास्त्री", "भौतिकविद"]), "painter": ("painter", "pintor", "चित्रकार", [], ["पेंटर"]),
        "composer": ("composer", "compositor", "संगीतकार", [], []), "playwright": ("playwright", "dramaturgo", "नाटककार", [], []),
        "novelist": ("novelist", "novelista", "उपन्यासकार", [], []), "poet": ("poet", "poeta", "कवि", [], []),
        "footballer": ("footballer", "futbolista", "फुटबॉलर", [], ["फ़ुटबॉलर", "फुटबॉल खिलाड़ी", "फ़ुटबॉल खिलाड़ी"]),
        "cricketer": ("cricketer", "jugador de críquet", "क्रिकेटर", ["jugador de cricket", "criquetista"], ["क्रिकेट खिलाड़ी"]),
        "astronomer": ("astronomer", "astrónomo", "खगोलशास्त्री", [], ["खगोलविद"]), "mathematician": ("mathematician", "matemático", "गणितज्ञ", [], []),
        "philosopher": ("philosopher", "filósofo", "दार्शनिक", [], []), "economist": ("economist", "economista", "अर्थशास्त्री", [], []),
        "inventor": ("inventor", "inventor", "आविष्कारक", [], ["अविष्कारक", "आविष्कारकर्ता"]), "astronaut": ("astronaut", "astronauta", "अंतरिक्ष यात्री", [], ["अंतरिक्षयात्री"]),
        "singer": ("singer", "cantante", "गायक", [], [])}
PEOPLE = [(("Albert Einstein", "Albert Einstein", "अल्बर्ट आइंस्टीन"), "physicist"), (("Stephen Hawking", "Stephen Hawking", "स्टीफन हॉकिंग"), "physicist"),
          (("Pablo Picasso", "Pablo Picasso", "पाब्लो पिकासो"), "painter"), (("Vincent van Gogh", "Vincent van Gogh", "विंसेंट वैन गॉग"), "painter"),
          (("Claude Monet", "Claude Monet", "क्लॉड मोने"), "painter"), (("Salvador Dalí", "Salvador Dalí", "साल्वाडोर डाली"), "painter"),
          (("Mozart", "Mozart", "मोज़ार्ट"), "composer"), (("Beethoven", "Beethoven", "बीथोवन"), "composer"), (("Bach", "Bach", "बाख"), "composer"),
          (("William Shakespeare", "William Shakespeare", "विलियम शेक्सपियर"), "playwright"), (("Charles Dickens", "Charles Dickens", "चार्ल्स डिकेंस"), "novelist"),
          (("Leo Tolstoy", "León Tolstói", "लियो टॉल्स्टॉय"), "novelist"), (("Jane Austen", "Jane Austen", "जेन ऑस्टिन"), "novelist"),
          (("Rabindranath Tagore", "Rabindranath Tagore", "रवींद्रनाथ टैगोर"), "poet"), (("Pablo Neruda", "Pablo Neruda", "पाब्लो नेरूदा"), "poet"),
          (("Lionel Messi", "Lionel Messi", "लियोनेल मेसी"), "footballer"), (("Cristiano Ronaldo", "Cristiano Ronaldo", "क्रिस्टियानो रोनाल्डो"), "footballer"),
          (("Pelé", "Pelé", "पेले"), "footballer"), (("Diego Maradona", "Diego Maradona", "डिएगो माराडोना"), "footballer"),
          (("Sachin Tendulkar", "Sachin Tendulkar", "सचिन तेंदुलकर"), "cricketer"), (("Virat Kohli", "Virat Kohli", "विराट कोहली"), "cricketer"),
          (("Galileo Galilei", "Galileo Galilei", "गैलीलियो गैलिली"), "astronomer"), (("Nicolaus Copernicus", "Nicolás Copérnico", "निकोलस कॉपरनिकस"), "astronomer"),
          (("Pythagoras", "Pitágoras", "पाइथागोरस"), "mathematician"), (("Euclid", "Euclides", "यूक्लिड"), "mathematician"),
          (("Aristotle", "Aristóteles", "अरस्तू"), "philosopher"), (("Plato", "Platón", "प्लेटो"), "philosopher"), (("Socrates", "Sócrates", "सुकरात"), "philosopher"),
          (("Confucius", "Confucio", "कन्फ्यूशियस"), "philosopher"), (("Immanuel Kant", "Immanuel Kant", "इमैनुएल कांट"), "philosopher"),
          (("Adam Smith", "Adam Smith", "एडम स्मिथ"), "economist"), (("Thomas Edison", "Thomas Edison", "थॉमस एडिसन"), "inventor"),
          (("Alexander Graham Bell", "Alexander Graham Bell", "अलेक्जेंडर ग्राहम बेल"), "inventor"), (("Neil Armstrong", "Neil Armstrong", "नील आर्मस्ट्रांग"), "astronaut"),
          (("Elvis Presley", "Elvis Presley", "एल्विस प्रेस्ली"), "singer"), (("Michael Jackson", "Michael Jackson", "माइकल जैक्सन"), "singer"),
          (("Freddie Mercury", "Freddie Mercury", "फ्रेडी मर्करी"), "singer"), (("Lata Mangeshkar", "Lata Mangeshkar", "लता मंगेशकर"), "singer")]

# ---- book_author: ((title en, es, hi), (author en, es, hi), [en alts], [es alts], [hi alts])
BOOKS = [
    (("Hamlet", "Hamlet", "हैमलेट"), ("William Shakespeare", "William Shakespeare", "विलियम शेक्सपियर"), ["Shakespeare"], ["Shakespeare"], ["शेक्सपियर"]),
    (("Romeo and Juliet", "Romeo y Julieta", "रोमियो और जूलियट"), ("William Shakespeare", "William Shakespeare", "विलियम शेक्सपियर"), ["Shakespeare"], ["Shakespeare"], ["शेक्सपियर"]),
    (("Macbeth", "Macbeth", "मैकबेथ"), ("William Shakespeare", "William Shakespeare", "विलियम शेक्सपियर"), ["Shakespeare"], ["Shakespeare"], ["शेक्सपियर"]),
    (("Othello", "Otelo", "ओथेलो"), ("William Shakespeare", "William Shakespeare", "विलियम शेक्सपियर"), ["Shakespeare"], ["Shakespeare"], ["शेक्सपियर"]),
    (("Pride and Prejudice", "Orgullo y prejuicio", "प्राइड एंड प्रेजुडिस"), ("Jane Austen", "Jane Austen", "जेन ऑस्टिन"), [], [], []),
    (("Oliver Twist", "Oliver Twist", "ओलिवर ट्विस्ट"), ("Charles Dickens", "Charles Dickens", "चार्ल्स डिकेंस"), ["Dickens"], ["Dickens"], ["डिकेंस"]),
    (("Great Expectations", "Grandes esperanzas", "ग्रेट एक्सपेक्टेशंस"), ("Charles Dickens", "Charles Dickens", "चार्ल्स डिकेंस"), ["Dickens"], ["Dickens"], ["डिकेंस"]),
    (("War and Peace", "Guerra y paz", "युद्ध और शांति"), ("Leo Tolstoy", "León Tolstói", "लियो टॉल्स्टॉय"), ["Tolstoy"], ["Tolstói", "Tolstoi"], ["टॉल्स्टॉय"]),
    (("Anna Karenina", "Ana Karénina", "अन्ना कारेनिना"), ("Leo Tolstoy", "León Tolstói", "लियो टॉल्स्टॉय"), ["Tolstoy"], ["Tolstói", "Tolstoi"], ["टॉल्स्टॉय"]),
    (("Crime and Punishment", "Crimen y castigo", "अपराध और दंड"), ("Fyodor Dostoevsky", "Fiódor Dostoyevski", "फ्योदोर दोस्तोयेव्स्की"), ["Dostoevsky"], ["Dostoyevski", "Dostoievski"], ["दोस्तोयेव्स्की"]),
    (("Les Misérables", "Los miserables", "ले मिजरेबल्स"), ("Victor Hugo", "Victor Hugo", "विक्टर ह्यूगो"), ["Hugo"], ["Hugo"], ["ह्यूगो"]),
    (("The Odyssey", "la Odisea", "ओडिसी"), ("Homer", "Homero", "होमर"), [], [], []), (("The Iliad", "la Ilíada", "इलियड"), ("Homer", "Homero", "होमर"), [], [], []),
    (("The Divine Comedy", "La divina comedia", "द डिवाइन कॉमेडी"), ("Dante Alighieri", "Dante Alighieri", "दांते अलीघिएरी"), ["Dante"], ["Dante"], ["दांते"]),
    (("Faust", "Fausto", "फ़ॉस्ट"), ("Johann Wolfgang von Goethe", "Johann Wolfgang von Goethe", "गोएथे"), ["Goethe"], ["Goethe"], ["गेटे", "गोथे"]),
    (("Frankenstein", "Frankenstein", "फ्रैंकेनस्टाइन"), ("Mary Shelley", "Mary Shelley", "मैरी शेली"), [], [], []),
    (("Dracula", "Drácula", "ड्रैकुला"), ("Bram Stoker", "Bram Stoker", "ब्रैम स्टोकर"), [], [], []),
    (("Animal Farm", "Rebelión en la granja", "एनिमल फार्म"), ("George Orwell", "George Orwell", "जॉर्ज ऑरवेल"), ["Orwell"], ["Orwell"], ["ऑरवेल"]),
    (("Moby-Dick", "Moby Dick", "मोबी डिक"), ("Herman Melville", "Herman Melville", "हरमन मेलविल"), ["Melville"], ["Melville"], ["मेलविल"]),
    (("The Great Gatsby", "El gran Gatsby", "द ग्रेट गैट्सबी"), ("F. Scott Fitzgerald", "F. Scott Fitzgerald", "एफ. स्कॉट फिट्ज़गेराल्ड"), ["Fitzgerald"], ["Fitzgerald"], ["फिट्ज़गेराल्ड"]),
    (("The Old Man and the Sea", "El viejo y el mar", "द ओल्ड मैन एंड द सी"), ("Ernest Hemingway", "Ernest Hemingway", "अर्नेस्ट हेमिंग्वे"), ["Hemingway"], ["Hemingway"], ["हेमिंग्वे"]),
    (("Alice's Adventures in Wonderland", "Alicia en el país de las maravillas", "एलिस इन वंडरलैंड"), ("Lewis Carroll", "Lewis Carroll", "लुईस कैरोल"), ["Carroll"], ["Carroll"], ["कैरोल"]),
    (("Gulliver's Travels", "Los viajes de Gulliver", "गुलिवर्स ट्रैवल्स"), ("Jonathan Swift", "Jonathan Swift", "जोनाथन स्विफ्ट"), ["Swift"], ["Swift"], ["स्विफ्ट"]),
    (("Robinson Crusoe", "Robinson Crusoe", "रॉबिन्सन क्रूसो"), ("Daniel Defoe", "Daniel Defoe", "डेनियल डेफ़ो"), ["Defoe"], ["Defoe"], ["डेफ़ो", "डेफो"]),
    (("Gitanjali", "Gitanjali", "गीतांजलि"), ("Rabindranath Tagore", "Rabindranath Tagore", "रवींद्रनाथ टैगोर"), ["Tagore"], ["Tagore"], ["टैगोर"]),
    (("Godan", "Godán", "गोदान"), ("Premchand", "Premchand", "प्रेमचंद"), [], [], []),
    (("the Ramayana", "el Ramayana", "रामायण"), ("Valmiki", "Valmiki", "वाल्मीकि"), [], [], []),
    (("the Mahabharata", "el Mahabharata", "महाभारत"), ("Vyasa", "Vyasa", "व्यास"), [], [], []),
    (("Don Quixote", "Don Quijote de la Mancha", "डॉन क्विक्सोट"), ("Miguel de Cervantes", "Miguel de Cervantes", "मिगेल दे सर्वांतेस"), ["Cervantes"], ["Cervantes"], ["सर्वांतेस", "सरवेंटीस"]),
]

POOL_A23 = {
    "official_language": {
        "en": ["The official language of {s} is", "The main language spoken in {s} is", "People in {s} speak", "The national language of {s} is", "In {s}, most people speak",
               "The language of government in {s} is"],
        "es": ["La lengua oficial de {s} es", "El idioma oficial de {s} es", "En {s} se habla", "La lengua principal de {s} es", "En {s} la gente habla",
               "El idioma que más se habla en {s} es"],
        "hi": ["{s} की आधिकारिक भाषा", "{s} की राजभाषा", "{s} में बोली जाने वाली मुख्य भाषा", "{s} के लोग मुख्य रूप से बोलते हैं", "{s} की मुख्य भाषा", "{s} की राष्ट्रभाषा"],
    },
    "currency": {
        "en": ["The currency of {s} is", "The official currency of {s} is", "People in {s} pay with", "The money used in {s} is", "The national currency of {s} is",
               "In {s}, prices are paid in"],
        "es": ["La moneda de {s} es", "La moneda oficial de {s} es", "En {s} se paga con", "El dinero que se usa en {s} es", "La divisa nacional de {s} es",
               "En {s} los precios se pagan en"],
        "hi": ["{s} की मुद्रा", "{s} की आधिकारिक मुद्रा", "{s} में पैसे इस नाम से चलते हैं:", "{s} की राष्ट्रीय मुद्रा", "{s} में भुगतान किया जाता है", "{s} की करेंसी"],
    },
    "profession": {
        "en": ["{S} was a famous", "{S} is best known as a", "{S} worked as a", "By profession, {s} was a", "{S} became world-famous as a", "{S} is remembered as a great"],
        "es": ["{S} fue un famoso", "{S} es conocido como", "{S} trabajó como", "Por profesión, {s} fue", "{S} se hizo famoso como", "{S} es recordado como un gran"],
        "hi": ["{s} एक प्रसिद्ध", "{s} एक मशहूर", "{s} पेशे से एक", "{s} को एक महान", "{s} दुनिया भर में जाने जाते हैं एक", "{s} की पहचान एक"],
    },
    "book_author": {
        "en": ["{S} was written by", "The author of {s} is", "{S} is a book by", "{S} was authored by", "The writer of {s} is", "The book {s} was created by"],
        "es": ["{S} fue escrito por", "El autor de {s} es", "{S} es una obra de", "{S} fue obra de", "El escritor de {s} es", "El libro {s} fue creado por"],
        "hi": ["{s} को लिखा था", "{s} के लेखक", "{s} की रचना की थी", "{s} के रचनाकार", "{s} किसकी रचना है:", "{s} इन्होंने लिखी थी:"],
    },
}


def country(name: str) -> tuple[str, str, str]:
    if name in EXTRA_COUNTRIES:
        return EXTRA_COUNTRIES[name]
    for (c, _k) in CAPITALS:
        if c[0] == name:
            return c
    raise KeyError(name)


def make(rel: str, idx: int, subj: tuple[str, ...], obj: tuple[str, ...], alts: dict[str, list[str]]) -> dict:
    fid = f"{rel}_{idx:03d}"
    seed = int(hashlib.sha256(fid.encode()).hexdigest(), 16) % (2**31)
    f = {"id": fid, "relation": rel, "seed": seed, "lang": {}}
    for k, lg in enumerate(LANGS):
        f["lang"][lg] = {"subject": subj[k], "object": obj[k], "object_alts": alts.get(lg, []),
                         "prompts": [fmt(t, lg, subj[k]) for t in POOL_A23[rel][lg][:3]]}
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v2/facts_candidates_a23.json")
    a = ap.parse_args()
    facts = []
    for i, (c, k) in enumerate(COUNTRY_LANG):  # Spanish answers usually carry an article: accept "el <language>"
        o = LANG[k]
        facts.append(make("official_language", i, country(c), o, {"es": [f"el {o[1]}"], "en": [], "hi": []}))
    for i, (c, k) in enumerate(COUNTRY_CUR):
        o = CUR[k]
        facts.append(make("currency", i, country(c), o, {"en": [f"the {o[0]}"], "es": [f"el {o[1]}", f"la {o[1]}"]}))
    for i, (p, k) in enumerate(PEOPLE):
        o = PROF[k]
        facts.append(make("profession", i, p, o[:3], {"es": o[3], "hi": o[4]}))
    for i, (t, au, ae, se, he) in enumerate(BOOKS):
        facts.append(make("book_author", i, t, au, {"en": ae, "es": se, "hi": he}))
    Path(a.out).write_text(json.dumps({"version": 2, "note": "A23 non-geography candidates (unaudited)", "facts": facts}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(facts), "candidates:", {r: sum(f["relation"] == r for f in facts) for r in POOL_A23})


if __name__ == "__main__":
    main()
