"""Study v2 candidates (docs/HYPOTHESIS.md A5): 5 relations, 3 phrasings per fact and language -> data/v2/facts_candidates.json.

Reuses the v1 capital and landmark lists (ids and seeds unchanged) and adds more landmarks, inventions and historical
events. Entity translations are hand-written. Phrasing 0 is the canonical phrasing used for tracing and editing.
`object_alts` lists acceptable alternative surface forms of the object (any of them counts as correct).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from build_facts import CAPITALS, LANDMARKS

LANGS = ("en", "es", "hi")

# (landmark (en, es, hi), city (en, es, hi)) -- batch 2, appended after the 39 v1 landmarks
LANDMARKS2 = [
    (("the Pantheon", "el Panteón", "पैन्थियन"), ("Rome", "Roma", "रोम")),
    (("the Vatican Museums", "los Museos Vaticanos", "वेटिकन संग्रहालय"), ("Vatican City", "Ciudad del Vaticano", "वेटिकन सिटी")),
    (("the Blue Mosque", "la Mezquita Azul", "ब्लू मस्जिद"), ("Istanbul", "Estambul", "इस्तांबुल")),
    (("the Cologne Cathedral", "la Catedral de Colonia", "कोलोन कैथेड्रल"), ("Cologne", "Colonia", "कोलोन")),
    (("the Neuschwanstein Castle", "el Castillo de Neuschwanstein", "न्यूशवांस्टाइन कैसल"), ("Füssen", "Füssen", "फ्युसेन")),
    (("the Reichstag", "el Reichstag", "राइखस्टाग"), ("Berlin", "Berlín", "बर्लिन")),
    (("the Parthenon", "el Partenón", "पार्थेनन"), ("Athens", "Atenas", "एथेंस")),
    (("Buckingham Palace", "el Palacio de Buckingham", "बकिंघम पैलेस"), ("London", "Londres", "लंदन")),
    (("the Tower of London", "la Torre de Londres", "टॉवर ऑफ़ लंदन"), ("London", "Londres", "लंदन")),
    (("the London Eye", "el London Eye", "लंदन आई"), ("London", "Londres", "लंदन")),
    (("the Arc de Triomphe", "el Arco del Triunfo", "आर्क डी ट्रायम्फ"), ("Paris", "París", "पेरिस")),
    (("Notre-Dame Cathedral", "la Catedral de Notre Dame", "नोट्र-डेम कैथेड्रल"), ("Paris", "París", "पेरिस")),
    (("the Palace of Versailles", "el Palacio de Versalles", "वर्साय का महल"), ("Versailles", "Versalles", "वर्साय")),
    (("the Park Güell", "el Parque Güell", "पार्क ग्वेल"), ("Barcelona", "Barcelona", "बार्सिलोना")),
    (("the Royal Palace", "el Palacio Real", "रॉयल पैलेस"), ("Madrid", "Madrid", "मैड्रिड")),
    (("the Guggenheim Museum", "el Museo Guggenheim", "गुगेनहाइम संग्रहालय"), ("Bilbao", "Bilbao", "बिलबाओ")),
    (("the Empire State Building", "el Empire State", "एम्पायर स्टेट बिल्डिंग"), ("New York", "Nueva York", "न्यूयॉर्क")),
    (("the White House", "la Casa Blanca", "व्हाइट हाउस"), ("Washington", "Washington", "वॉशिंगटन")),
    (("Mount Rushmore", "el Monte Rushmore", "माउंट रशमोर"), ("Keystone", "Keystone", "कीस्टोन")),
    (("the Hollywood Walk of Fame", "el Paseo de la Fama de Hollywood", "हॉलीवुड वॉक ऑफ़ फ़ेम"), ("Los Angeles", "Los Ángeles", "लॉस एंजिल्स")),
    (("the Chichen Itza pyramid", "la pirámide de Chichén Itzá", "चिचेन इत्ज़ा पिरामिड"), ("Yucatán", "Yucatán", "युकातान")),
    (("Machu Picchu", "Machu Picchu", "माचू पिच्चू"), ("Cusco", "Cuzco", "कुस्को")),
    (("the Great Wall", "la Gran Muralla", "चीन की महान दीवार"), ("Beijing", "Pekín", "बीजिंग")),
    (("the Temple of Heaven", "el Templo del Cielo", "स्वर्ग का मंदिर"), ("Beijing", "Pekín", "बीजिंग")),
    (("the Oriental Pearl Tower", "la Torre de la Perla Oriental", "ओरिएंटल पर्ल टावर"), ("Shanghai", "Shanghái", "शंघाई")),
    (("the Tokyo Skytree", "la Tokyo Skytree", "टोक्यो स्काईट्री"), ("Tokyo", "Tokio", "टोक्यो")),
    (("Himeji Castle", "el Castillo de Himeji", "हिमेजी कैसल"), ("Himeji", "Himeji", "हिमेजी")),
    (("the Gyeongbokgung Palace", "el Palacio Gyeongbokgung", "ग्योंगबोकगुंग महल"), ("Seoul", "Seúl", "सियोल")),
    (("the Marina Bay Sands", "el Marina Bay Sands", "मरीना बे सैंड्स"), ("Singapore", "Singapur", "सिंगापुर")),
    (("Angkor Wat", "Angkor Wat", "अंगकोर वाट"), ("Siem Reap", "Siem Reap", "सिएम रीप")),
    (("the Lotus Temple", "el Templo del Loto", "लोटस टेंपल"), ("Delhi", "Delhi", "दिल्ली")),
    (("India Gate", "la Puerta de la India", "इंडिया गेट"), ("Delhi", "Delhi", "दिल्ली")),
    (("the Qutub Minar", "el Qutb Minar", "कुतुब मीनार"), ("Delhi", "Delhi", "दिल्ली")),
    (("the Gateway of India", "la Puerta de la India de Bombay", "गेटवे ऑफ़ इंडिया"), ("Mumbai", "Bombay", "मुंबई")),
    (("the Konark Sun Temple", "el Templo del Sol de Konark", "कोणार्क सूर्य मंदिर"), ("Konark", "Konarak", "कोणार्क")),
    (("the Ajanta Caves", "las cuevas de Ajanta", "अजंता की गुफाएँ"), ("Aurangabad", "Aurangabad", "औरंगाबाद")),
    (("the Mahabodhi Temple", "el Templo Mahabodhi", "महाबोधि मंदिर"), ("Bodh Gaya", "Bodh Gaya", "बोधगया")),
    (("the Lingaraja Temple", "el Templo Lingaraja", "लिंगराज मंदिर"), ("Bhubaneswar", "Bhubaneswar", "भुवनेश्वर")),
    (("the Fatehpur Sikri", "Fatehpur Sikri", "फ़तेहपुर सीकरी"), ("Agra", "Agra", "आगरा")),
    (("the Amber Fort", "el Fuerte Amber", "आमेर का किला"), ("Jaipur", "Jaipur", "जयपुर")),
    (("the Pyramids of Giza", "las pirámides de Guiza", "गीज़ा के पिरामिड"), ("Giza", "Guiza", "गीज़ा")),
    (("Table Mountain", "la Montaña de la Mesa", "टेबल माउंटेन"), ("Cape Town", "Ciudad del Cabo", "केप टाउन")),
]

# (invention (en, es, hi), inventor (en, es, hi), alternative object forms {lang: [...]})
INVENTIONS = [
    (("the telephone", "el teléfono", "टेलीफ़ोन"), ("Alexander Graham Bell", "Alexander Graham Bell", "अलेक्ज़ेंडर ग्राहम बेल"), {"en": ["Alexander Bell"], "es": ["Alexander Bell"], "hi": ["ग्राहम बेल"]}),
    (("the light bulb", "la bombilla", "बिजली का बल्ब"), ("Thomas Edison", "Thomas Edison", "थॉमस एडिसन"), {"en": ["Thomas Alva Edison"], "es": ["Thomas Alva Edison"], "hi": ["थॉमस अल्वा एडिसन"]}),
    (("the telegraph code", "el código telegráfico", "टेलीग्राफ़ कोड"), ("Samuel Morse", "Samuel Morse", "सैमुअल मोर्स"), {}),
    (("the radio", "la radio", "रेडियो"), ("Guglielmo Marconi", "Guglielmo Marconi", "गुग्लिएल्मो मार्कोनी"), {"hi": ["मार्कोनी"]}),
    (("the printing press", "la imprenta", "छापाखाना"), ("Johannes Gutenberg", "Johannes Gutenberg", "योहानेस गुटेनबर्ग"), {"hi": ["जोहान्स गुटेनबर्ग", "गुटेनबर्ग"]}),
    (("dynamite", "la dinamita", "डायनामाइट"), ("Alfred Nobel", "Alfred Nobel", "अल्फ्रेड नोबेल"), {"hi": ["आल्फ्रेड नोबेल"]}),
    (("penicillin", "la penicilina", "पेनिसिलिन"), ("Alexander Fleming", "Alexander Fleming", "अलेक्ज़ेंडर फ्लेमिंग"), {"hi": ["एलेक्जेंडर फ्लेमिंग", "अलेक्जेंडर फ्लेमिंग"]}),
    (("the smallpox vaccine", "la vacuna contra la viruela", "चेचक का टीका"), ("Edward Jenner", "Edward Jenner", "एडवर्ड जेनर"), {}),
    (("pasteurization", "la pasteurización", "पाश्चुरीकरण"), ("Louis Pasteur", "Louis Pasteur", "लुई पाश्चर"), {}),
    (("the periodic table", "la tabla periódica", "आवर्त सारणी"), ("Dmitri Mendeleev", "Dmitri Mendeléyev", "दिमित्री मेंडलीव"), {"es": ["Dmitri Mendeleev", "Dimitri Mendeléyev"], "hi": ["दमित्री मेंडेलीव", "मेंडलीव"]}),
    (("the World Wide Web", "la World Wide Web", "वर्ल्ड वाइड वेब"), ("Tim Berners-Lee", "Tim Berners-Lee", "टिम बर्नर्स-ली"), {"hi": ["टिम बर्नर्स ली"]}),
    (("the electric battery", "la pila eléctrica", "विद्युत बैटरी"), ("Alessandro Volta", "Alessandro Volta", "अलेस्सांद्रो वोल्टा"), {"hi": ["एलेसांद्रो वोल्टा", "एलेसेंड्रो वोल्टा"]}),
    (("the X-ray", "los rayos X", "एक्स-रे"), ("Wilhelm Röntgen", "Wilhelm Röntgen", "विल्हेल्म रॉन्टगेन"), {"en": ["Wilhelm Roentgen"], "hi": ["विल्हेम रॉन्टजेन", "रॉन्टजन"]}),
    (("the stethoscope", "el estetoscopio", "स्टेथोस्कोप"), ("René Laennec", "René Laennec", "रेने लेनेक"), {"en": ["Rene Laennec"], "hi": ["रेने लैनेक"]}),
    (("the diesel engine", "el motor diésel", "डीज़ल इंजन"), ("Rudolf Diesel", "Rudolf Diesel", "रूडोल्फ़ डीज़ल"), {"hi": ["रुडोल्फ डीजल", "रूडोल्फ डीज़ल"]}),
    (("the spinning jenny", "la hiladora jenny", "स्पिनिंग जेनी"), ("James Hargreaves", "James Hargreaves", "जेम्स हार्ग्रीव्ज़"), {"hi": ["जेम्स हरग्रीव्स"]}),
    (("the cotton gin", "la desmotadora de algodón", "कपास ओटने की मशीन"), ("Eli Whitney", "Eli Whitney", "एली व्हिटनी"), {}),
    (("the lightning rod", "el pararrayos", "तड़ित चालक"), ("Benjamin Franklin", "Benjamin Franklin", "बेंजामिन फ्रैंकलिन"), {}),
    (("the mechanical calculator", "la calculadora mecánica", "यांत्रिक कैलकुलेटर"), ("Blaise Pascal", "Blaise Pascal", "ब्लेज़ पास्कल"), {"hi": ["ब्लेस पास्कल"]}),
    (("the Analytical Engine", "la máquina analítica", "एनालिटिकल इंजन"), ("Charles Babbage", "Charles Babbage", "चार्ल्स बैबेज"), {"hi": ["चार्ल्स बैबिज"]}),
    (("the typewriter", "la máquina de escribir", "टाइपराइटर"), ("Christopher Sholes", "Christopher Sholes", "क्रिस्टोफर शोल्स"), {}),
    (("the revolver", "el revólver", "रिवॉल्वर"), ("Samuel Colt", "Samuel Colt", "सैमुअल कोल्ट"), {}),
    (("the Rubik's Cube", "el cubo de Rubik", "रूबिक क्यूब"), ("Ernő Rubik", "Ernő Rubik", "एर्नो रुबिक"), {"en": ["Erno Rubik"], "es": ["Erno Rubik"]}),
    (("the Linux kernel", "el núcleo Linux", "लिनक्स कर्नेल"), ("Linus Torvalds", "Linus Torvalds", "लिनस टोरवाल्ड्स"), {"hi": ["लिनुस टोरवाल्ड्स", "लाइनस टोरवाल्ड्स"]}),
    (("the Python programming language", "el lenguaje de programación Python", "पायथन प्रोग्रामिंग भाषा"), ("Guido van Rossum", "Guido van Rossum", "गाइडो वैन रॉसम"), {"hi": ["गुइडो वैन रॉसम", "गीडो वान रोसम"]}),
    (("the C programming language", "el lenguaje de programación C", "सी प्रोग्रामिंग भाषा"), ("Dennis Ritchie", "Dennis Ritchie", "डेनिस रिची"), {"hi": ["डेनिस रिचे"]}),
    (("the Java programming language", "el lenguaje de programación Java", "जावा प्रोग्रामिंग भाषा"), ("James Gosling", "James Gosling", "जेम्स गोसलिंग"), {}),
    (("Velcro", "el velcro", "वेल्क्रो"), ("George de Mestral", "George de Mestral", "जॉर्ज डी मेस्ट्राल"), {"hi": ["जॉर्ज दे मेस्त्राल"]}),
    (("Kevlar", "el kevlar", "केवलर"), ("Stephanie Kwolek", "Stephanie Kwolek", "स्टेफनी क्वोलेक"), {"hi": ["स्टेफ़नी क्वोलेक"]}),
    (("the ballpoint pen", "el bolígrafo", "बॉलपॉइंट पेन"), ("László Bíró", "László Bíró", "लास्ज़लो बिरो"), {"en": ["Laszlo Biro"], "es": ["Laszlo Biro", "László Biró"]}),
    (("the zipper", "la cremallera", "ज़िप"), ("Gideon Sundback", "Gideon Sundback", "गिडियन सुंडबैक"), {}),
    (("the microwave oven", "el horno microondas", "माइक्रोवेव ओवन"), ("Percy Spencer", "Percy Spencer", "पर्सी स्पेंसर"), {}),
    (("nylon", "el nailon", "नायलॉन"), ("Wallace Carothers", "Wallace Carothers", "वालेस कैरोथर्स"), {"hi": ["वॉलेस कैरोथर्स"]}),
    (("paper", "el papel", "कागज़"), ("Cai Lun", "Cai Lun", "साई लुन"), {"es": ["Cai Lung", "Tsai Lun"], "hi": ["त्साई लुन", "काई लुन"]}),
    (("the seismograph", "el sismógrafo", "भूकंपमापी"), ("Zhang Heng", "Zhang Heng", "झांग हेंग"), {"hi": ["ज़ांग हेंग"]}),
    (("the helicopter", "el helicóptero", "हेलीकॉप्टर"), ("Igor Sikorsky", "Igor Sikorsky", "इगोर सिकोर्स्की"), {"hi": ["इगोर सिकोरस्की"]}),
    (("the jet engine", "el motor a reacción", "जेट इंजन"), ("Frank Whittle", "Frank Whittle", "फ्रैंक व्हिटल"), {}),
    (("the safety razor", "la maquinilla de afeitar de seguridad", "सेफ्टी रेज़र"), ("King Gillette", "King Gillette", "किंग जिलेट"), {"en": ["King C. Gillette"], "hi": ["किंग जिलेट", "किंग कैंप जिलेट"]}),
    (("the cinematograph", "el cinematógrafo", "सिनेमैटोग्राफ़"), ("the Lumière brothers", "los hermanos Lumière", "ल्यूमियर बंधु"), {"en": ["Lumière brothers", "the Lumiere brothers"], "es": ["hermanos Lumière"], "hi": ["लुमियर बंधु", "ल्युमियर बंधु"]}),
    (("the airplane", "el avión", "हवाई जहाज़"), ("the Wright brothers", "los hermanos Wright", "राइट बंधु"), {"en": ["Wright brothers", "Orville and Wilbur Wright"], "es": ["hermanos Wright"], "hi": ["राइट बंधुओं"]}),
]

# (event (en, es, hi), year) -- years chosen distinct
EVENTS = [
    (("the Battle of Hastings", "la batalla de Hastings", "हेस्टिंग्स का युद्ध"), "1066"),
    (("the signing of the Magna Carta", "la firma de la Carta Magna", "मैग्ना कार्टा पर हस्ताक्षर"), "1215"),
    (("the fall of Constantinople", "la caída de Constantinopla", "कॉन्स्टेंटिनोपल का पतन"), "1453"),
    (("the first voyage of Columbus", "el primer viaje de Colón", "कोलंबस की पहली यात्रा"), "1492"),
    (("Martin Luther's Ninety-five Theses", "las Noventa y cinco tesis de Lutero", "मार्टिन लूथर की पचानवे थीसिस"), "1517"),
    (("the defeat of the Spanish Armada", "la derrota de la Armada Invencible", "स्पेनिश आर्मडा की हार"), "1588"),
    (("the Mayflower landing", "el desembarco del Mayflower", "मेफ़्लावर का उतरना"), "1620"),
    (("the Great Fire of London", "el Gran Incendio de Londres", "लंदन की महान आग"), "1666"),
    (("the Battle of Plassey", "la batalla de Plassey", "प्लासी का युद्ध"), "1757"),
    (("the American Declaration of Independence", "la Declaración de Independencia de Estados Unidos", "अमेरिकी स्वतंत्रता की घोषणा"), "1776"),
    (("the French Revolution", "la Revolución francesa", "फ़्रांसीसी क्रांति"), "1789"),
    (("the Battle of Trafalgar", "la batalla de Trafalgar", "ट्राफलगर का युद्ध"), "1805"),
    (("the Battle of Waterloo", "la batalla de Waterloo", "वाटरलू का युद्ध"), "1815"),
    (("the Irish Potato Famine", "la Gran Hambruna irlandesa", "आयरलैंड का आलू अकाल"), "1845"),
    (("the Indian Rebellion", "la rebelión de los cipayos", "1857 का भारतीय विद्रोह"), "1857"),
    (("the start of the American Civil War", "el inicio de la guerra de Secesión", "अमेरिकी गृहयुद्ध की शुरुआत"), "1861"),
    (("the opening of the Suez Canal", "la inauguración del canal de Suez", "स्वेज़ नहर का उद्घाटन"), "1869"),
    (("the eruption of Krakatoa", "la erupción del Krakatoa", "क्राकाटोआ का विस्फोट"), "1883"),
    (("the first modern Olympic Games", "los primeros Juegos Olímpicos modernos", "पहले आधुनिक ओलंपिक खेल"), "1896"),
    (("the Wright brothers' first flight", "el primer vuelo de los hermanos Wright", "राइट बंधुओं की पहली उड़ान"), "1903"),
    (("the sinking of the Titanic", "el hundimiento del Titanic", "टाइटैनिक का डूबना"), "1912"),
    (("the start of the First World War", "el inicio de la Primera Guerra Mundial", "प्रथम विश्व युद्ध की शुरुआत"), "1914"),
    (("the Russian Revolution", "la Revolución rusa", "रूसी क्रांति"), "1917"),
    (("the Treaty of Versailles", "el Tratado de Versalles", "वर्साय की संधि"), "1919"),
    (("the Wall Street Crash", "el crac del 29", "वॉल स्ट्रीट क्रैश"), "1929"),
    (("the Salt March", "la Marcha de la Sal", "नमक सत्याग्रह"), "1930"),
    (("the start of the Second World War", "el inicio de la Segunda Guerra Mundial", "द्वितीय विश्व युद्ध की शुरुआत"), "1939"),
    (("the independence of India", "la independencia de la India", "भारत की स्वतंत्रता"), "1947"),
    (("the assassination of Mahatma Gandhi", "el asesinato de Mahatma Gandhi", "महात्मा गांधी की हत्या"), "1948"),
    (("the first human spaceflight by Yuri Gagarin", "el primer vuelo espacial tripulado de Yuri Gagarin", "यूरी गगारिन की पहली मानव अंतरिक्ष उड़ान"), "1961"),
    (("the assassination of John F. Kennedy", "el asesinato de John F. Kennedy", "जॉन एफ़. कैनेडी की हत्या"), "1963"),
    (("the Apollo 11 Moon landing", "el alunizaje del Apolo 11", "अपोलो 11 का चंद्रमा पर उतरना"), "1969"),
    (("the Chernobyl disaster", "el desastre de Chernóbil", "चेर्नोबिल आपदा"), "1986"),
    (("the fall of the Berlin Wall", "la caída del Muro de Berlín", "बर्लिन की दीवार का गिरना"), "1989"),
    (("the dissolution of the Soviet Union", "la disolución de la Unión Soviética", "सोवियत संघ का विघटन"), "1991"),
    (("the September 11 attacks", "los atentados del 11 de septiembre", "11 सितंबर के हमले"), "2001"),
    (("the Indian Ocean tsunami", "el tsunami del océano Índico", "हिंद महासागर की सुनामी"), "2004"),
]

# batch 2 (A8): appended so earlier ids and seeds are unchanged; years distinct from batch 1 and from each other
EVENTS2 = [
    (("the arrival of the Black Death in Europe", "la llegada de la peste negra a Europa", "यूरोप में ब्लैक डेथ का आगमन"), "1347"),
    (("the Treaty of Tordesillas", "el Tratado de Tordesillas", "टोरडेसिलस की संधि"), "1494"),
    (("the fall of Tenochtitlan", "la caída de Tenochtitlan", "तेनोच्तितलान का पतन"), "1521"),
    (("the First Battle of Panipat", "la primera batalla de Panipat", "पानीपत की पहली लड़ाई"), "1526"),
    (("the Battle of Lepanto", "la batalla de Lepanto", "लेपांटो का युद्ध"), "1571"),
    (("the start of the Thirty Years' War", "el inicio de la guerra de los Treinta Años", "तीस वर्षीय युद्ध की शुरुआत"), "1618"),
    (("the Peace of Westphalia", "la Paz de Westfalia", "वेस्टफेलिया की शांति"), "1648"),
    (("the Great Plague of London", "la Gran Plaga de Londres", "लंदन की महान प्लेग"), "1665"),
    (("the start of the Haitian Revolution", "el inicio de la Revolución haitiana", "हैती क्रांति की शुरुआत"), "1791"),
    (("the coronation of Napoleon as emperor", "la coronación de Napoleón como emperador", "नेपोलियन का सम्राट के रूप में राज्याभिषेक"), "1804"),
    (("the Congress of Vienna", "el Congreso de Viena", "वियना की कांग्रेस"), "1814"),
    (("the start of the Greek War of Independence", "el inicio de la guerra de independencia de Grecia", "यूनानी स्वतंत्रता संग्राम की शुरुआत"), "1821"),
    (("the opening of the Stockton and Darlington Railway", "la inauguración del ferrocarril de Stockton y Darlington", "स्टॉकटन और डार्लिंगटन रेलवे का उद्घाटन"), "1825"),
    (("the start of the First Opium War", "el inicio de la primera guerra del Opio", "प्रथम अफ़ीम युद्ध की शुरुआत"), "1839"),
    (("the Revolutions of 1848", "las revoluciones de 1848", "1848 की क्रांतियाँ"), "1848"),
    (("the start of the Crimean War", "el inicio de la guerra de Crimea", "क्रीमिया युद्ध की शुरुआत"), "1853"),
    (("the publication of On the Origin of Species", "la publicación de El origen de las especies", "ऑन द ओरिजिन ऑफ़ स्पीशीज़ का प्रकाशन"), "1859"),
    (("the Meiji Restoration", "la restauración Meiji", "मेजी पुनर्स्थापना"), "1868"),
    (("the proclamation of the German Empire", "la proclamación del Imperio alemán", "जर्मन साम्राज्य की घोषणा"), "1871"),
    (("the Berlin Conference on Africa", "la Conferencia de Berlín sobre África", "अफ़्रीका पर बर्लिन सम्मेलन"), "1884"),
    (("the completion of the Eiffel Tower", "la finalización de la Torre Eiffel", "एफ़िल टॉवर का निर्माण पूर्ण होना"), "1889"),
    (("the Spanish-American War", "la guerra hispano-estadounidense", "स्पेन-अमेरिका युद्ध"), "1898"),
    (("the Boxer Rebellion", "la rebelión de los bóxers", "बॉक्सर विद्रोह"), "1900"),
    (("the San Francisco earthquake", "el terremoto de San Francisco", "सैन फ़्रांसिस्को भूकंप"), "1906"),
    (("the start of the Mexican Revolution", "el inicio de la Revolución mexicana", "मैक्सिकन क्रांति की शुरुआत"), "1910"),
    (("the start of the Spanish flu pandemic", "el inicio de la pandemia de gripe española", "स्पेनिश फ़्लू महामारी की शुरुआत"), "1918"),
    (("the start of the Spanish Civil War", "el inicio de la guerra civil española", "स्पेनी गृहयुद्ध की शुरुआत"), "1936"),
    (("the founding of the People's Republic of China", "la fundación de la República Popular China", "चीनी जनवादी गणराज्य की स्थापना"), "1949"),
    (("the start of the Korean War", "el inicio de la guerra de Corea", "कोरियाई युद्ध की शुरुआत"), "1950"),
    (("the Suez Crisis", "la crisis de Suez", "स्वेज़ संकट"), "1956"),
    (("the Cuban Missile Crisis", "la crisis de los misiles de Cuba", "क्यूबा मिसाइल संकट"), "1962"),
    (("the Six-Day War", "la guerra de los Seis Días", "छह दिवसीय युद्ध"), "1967"),
    (("the Bangladesh Liberation War", "la guerra de liberación de Bangladés", "बांग्लादेश मुक्ति संग्राम"), "1971"),
    (("the Iranian Revolution", "la Revolución iraní", "ईरानी क्रांति"), "1979"),
    (("the start of the Gulf War", "el inicio de la guerra del Golfo", "खाड़ी युद्ध की शुरुआत"), "1990"),
    (("the Rwandan genocide", "el genocidio de Ruanda", "रवांडा नरसंहार"), "1994"),
    (("the global financial crisis", "la crisis financiera mundial", "वैश्विक वित्तीय संकट"), "2008"),
    (("the Fukushima nuclear disaster", "el desastre nuclear de Fukushima", "फुकुशिमा परमाणु आपदा"), "2011"),
    (("the Brexit referendum", "el referéndum del Brexit", "ब्रेक्ज़िट जनमत संग्रह"), "2016"),
]

TEMPLATES = {
    "capital_of": {
        "en": ["The capital of {s} is", "The capital city of {s} is", "Q: What is the capital of {s}? A:"],
        "es": ["La capital de {s} es", "La ciudad capital de {s} es", "P: ¿Cuál es la capital de {s}? R:"],
        "hi": ["{s} की राजधानी", "{s} की राजधानी है", "प्रश्न: {s} की राजधानी क्या है? उत्तर:"],
    },
    "capital_country": {
        "en": ["{S} is the capital of", "{S} is the capital city of", "Q: Which country has {s} as its capital? A:"],
        "es": ["{S} es la capital de", "{S} es la ciudad capital de", "P: ¿De qué país es capital {s}? R:"],
        "hi": ["{s} इस देश की राजधानी है:", "{s} जिस देश की राजधानी है, उसका नाम", "प्रश्न: {s} किस देश की राजधानी है? उत्तर:"],
    },
    "landmark_city": {
        "en": ["{S} is located in the city of", "{S} can be found in the city of", "Q: In which city is {s} located? A:"],
        "es": ["{S} se encuentra en la ciudad de", "{S} se halla en la ciudad de", "P: ¿En qué ciudad se encuentra {s}? R:"],
        "hi": ["{s} इस शहर में स्थित है:", "{s} शहर में स्थित है, जिसका नाम", "प्रश्न: {s} किस शहर में स्थित है? उत्तर:"],
    },
    "invention_inventor": {
        "en": ["The inventor of {s} is", "{S} was invented by", "Q: Who invented {s}? A:"],
        "es": ["El inventor de {s} es", "El creador de {s} es", "P: ¿Quién inventó {s}? R:"],
        "hi": ["{s} का आविष्कार करने वाले व्यक्ति का नाम", "{s} के आविष्कारक", "प्रश्न: {s} का आविष्कार किसने किया? उत्तर:"],
    },
    "event_year": {
        "en": ["{S} took place in the year", "{S} happened in", "Q: In what year did {s} take place? A:"],
        "es": ["{S} tuvo lugar en el año", "{S} ocurrió en", "P: ¿En qué año tuvo lugar {s}? R:"],
        "hi": ["{s} सन्", "{s} वर्ष में हुआ था:", "प्रश्न: {s} किस वर्ष हुआ था? उत्तर:"],
    },
}


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def es_fix(t: str) -> str:
    return t.replace(" de el ", " del ").replace(" a el ", " al ")


def fmt(t: str, lg: str, s: str) -> str:
    out = t.format(s=s, S=cap(s)) if lg != "hi" else t.format(s=s, S=s)
    return es_fix(out) if lg == "es" else out


def make(rel: str, idx: int, subj: tuple[str, ...], obj: tuple[str, ...], alts: dict | None = None) -> dict:
    fid = f"{rel}_{idx:03d}"
    seed = int(hashlib.sha256(fid.encode()).hexdigest(), 16) % (2**31)
    f = {"id": fid, "relation": rel, "seed": seed, "lang": {}}
    for k, lg in enumerate(LANGS):
        f["lang"][lg] = {"subject": subj[k], "object": obj[k], "object_alts": (alts or {}).get(lg, []),
                         "prompts": [fmt(t, lg, subj[k]) for t in TEMPLATES[rel][lg]]}
    return f


def load_templates(path: str = "data/v2/templates.json") -> dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf8"))["templates"]
    return TEMPLATES


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default="data/v2/templates.json")
    ap.add_argument("--out", default="data/v2/facts_candidates.json")
    ap.add_argument("--with-events2", action="store_true", help="append the A8 event batch (Gemma pool)")
    a = ap.parse_args()
    global TEMPLATES
    TEMPLATES = load_templates(a.templates)
    facts = [make("capital_of", i, c, k) for i, (c, k) in enumerate(CAPITALS)]
    facts += [make("capital_country", i, k, c) for i, (c, k) in enumerate(CAPITALS)]
    facts += [make("landmark_city", i, l, c) for i, (l, c) in enumerate(LANDMARKS + LANDMARKS2)]
    facts += [make("invention_inventor", i, inv, who, alts) for i, (inv, who, alts) in enumerate(INVENTIONS)]
    facts += [make("event_year", i, ev, (y, y, y)) for i, (ev, y) in enumerate(EVENTS + (EVENTS2 if a.with_events2 else []))]
    Path("data/v2").mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"version": 2, "note": "Study v2 candidates (A5); 3 phrasings per fact and language; phrasing 0 is canonical", "facts": facts},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(facts), "candidates:", {r: sum(f["relation"] == r for f in facts) for r in TEMPLATES})


if __name__ == "__main__":
    main()
