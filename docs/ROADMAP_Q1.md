# Roadmap to a Q1-level paper (written 2026-10-26)
Goal (user, m00927): target a Q1 journal. Honest gap: today = 2 small models (3B/2B), 4 geography relations for interchange, EN/ES/HI, mostly descriptive edits. Q1 reviewers will want breadth (models, languages, relations), a mechanism, and an intervention result. Venue is decided after the evidence; no quartile is asserted here (check SCImago/JCR).
Constraint check: $0 and one 8 GB GPU. Free 16 GB GPUs (Kaggle/Colab T4) would allow 7-9B models in 4/8-bit; needs the user's OK (outward-facing: uploads code/data).

## Phase A - close the known holes (local, 8 GB, ~3 weeks)
A1 E4 Llama-3.2-3B base (overlap + interchange)       4-6 GPU h   removes instruct/base confound
A2 A18 EN->HI span-wise subject patch (pre-register)   2-3 GPU h   closes the one-direction Hindi gap
A3 Edits n=100 x {traced, L//2, A9 layer} + EN-paraphrase control   8-12 GPU h
A4 Paper-text fixes (abstract vs 4.7, A11 vs A13b, Hindi as estimate, inline n/CI, p-floor)   CPU
## Phase B - breadth (what Q1 reviewers ask first)
B1 Models: add >=2 more families at >=7B (e.g. Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B) - needs 16 GB GPU or 4-bit; keep 2 small ones   ~30-40 GPU h
B2 Languages: add 2-3 typologically different (e.g. Chinese, Arabic, Swahili/low-resource) to EN/ES/HI; native-speaker audit of kept facts
B3 Relations: add non-geography (people/occupations, language-of-country, element symbols; Wikidata name-dissimilar facts = E3) so the claim is not "geography only"
## Phase C - mechanism + intervention (what makes it a Q1 contribution, not a measurement note)
C1 Which components carry the shared state: attention-head vs MLP interchange, path patching on the subject->answer route; where language-specific parts (tokenisation/name) enter
C2 Predictive use: does the shared-ness score predict cross-lingual edit transfer per fact (P6-style validity across models)? A model of when edits transfer
C3 Practical method: a cross-lingual edit that targets the shared layer, with specificity/damage, compared with MEMIT/ROME-multilingual baselines (Khandelwal/Wang style)
## Phase D - rigor and release
Pre-register each phase in docs/HYPOTHESIS.md before data; keep manifest/verify; external timestamp (Zenodo/OSF DOI - user); replication package; ablate every choice (thresholds, clustering, templates).
## Decision gates
G-A (after Phase A): EN-ES shared holds on base Llama and EN->HI is not much lower than HI->EN -> continue to B. If not: narrow to the audit/pitfalls paper (TMLR/workshop).
G-B (after B): claim holds on >=4 models, >=5 languages, >=3 relation families -> do C. If it holds only on some, the paper becomes "when is it shared" (still publishable, stronger).
G-C: a mechanism or an edit method that beats baselines -> submit to Q1; otherwise TMLR.
