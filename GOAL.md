# GOAL: The Babel Effect

Master tracking file. Updated 2026-10-30 (replaces the 2026-10-29 version). Detail lives in `docs/HYPOTHESIS.md` (pre-registrations A1-A28), `docs/EXPERIMENT_LOG.md`, `docs/LOG.md`, `docs/NEXT.md`, `docs/ROADMAP_Q1.md`. Nothing here is committed or pushed until the user says so.

## 1. Goal
**Win:** the strongest results we can, aimed at a Q1 journal if the results allow, otherwise Q2-Q3 (venue decided later). The work comes first; the paper is written as well as possible after. Plan B if no positive method result appears: a diagnostic paper (layer map, wording gap, null shared-key result).

**Question:** are facts stored in language-specific circuits or in a shared space, and what does the answer mean for editing a fact in one language? Languages: English, Spanish, Hindi.

**Evidence rule (2026-10-30):** all new evidence comes from the Kaggle runs (2x T4, fp16). Results are compared with published work only, never with the earlier laptop runs. A new result is strengthened (controls, more facts, layer map) before it is claimed.

**Acceptance rule (2026-10-31, from the user):** everything handed over is run on Kaggle by the user, and the paper is accepted only if it has novelty. So every notebook and every claim is judged by what it adds that no paper has; the novelty claim is worded as "not found in prior work" until the five closest papers (2310.10378, 2601.17397, 2406.16416, 2504.09378, 2510.10280) are read in full.

**Model rule (2026-10-31, from the user):** run the models the published papers used so our numbers can be compared with theirs like for like, and use them to show our work with more work (baselines ROME, MEMIT and in-context on the same facts, the same benchmark slice and metrics). Whether ours is best is decided by the results and reported as found: today we have a new explanation (key similarity predicts transfer), not a better editor; the one place we could have beaten published editors, the shared-key fix, was null.

## 2. Where we are
- Kaggle run 1 (Llama-3.2-3B-Instruct, 74 min, 2026-10-30):
  - shared-key edit gives no improvement (paired median gain 0.06-0.08 for Spanish at the traced layer, about 0 elsewhere): Plan B stands
  - loss beyond rewording: Hindi 0.83 (traced layer) and 0.66 (layer 14), Spanish traced 0.37, Spanish layer 14 0.13 (inconclusive)
  - Spanish keeps 0.50 of the English effect at the traced layer and 0.85 at layer 14; Hindi loses most of it
  - non-geography facts: untestable (53 English-Spanish and 13 English-Hindi facts, only 2 relations survived)
- **Extra result (first pass, 28 facts per cell):** at the traced layer, facts whose Spanish or Hindi subject key is closer to the English key transfer more: rho 0.55 (Spanish) and 0.52 (Hindi), Holm p 0.006 and 0.010. At layer 14: -0.06 and 0.02. With the edit layer also controlled: Spanish 0.45 [-0.01, 0.75], Hindi 0.06 [-0.32, 0.57]. Not a claim yet.
- **Next run (A28, registered before its data exist):** one notebook that traces the facts itself, edits about 85 facts at the traced layer and at layer 14 (English key and shared key), maps layers 2/6/10/18/22 on 40 facts, and runs the key-similarity test, wording control, shared-key analysis and graphs. About 6 h on 2x T4 (limit 10.5 h).
- **A28 result (2026-10-31, 86 facts, layer map on 40; verdicts in `docs/HYPOTHESIS.md`):** key similarity predicts which facts carry over at the traced layer: Spanish rho 0.41 [0.20, 0.59], Hindi 0.42 [0.19, 0.62] (Holm p 0.001 each); none at layer 14. Beyond the edit layer both pass narrowly (Spanish 0.26 [0.004, 0.49], Hindi 0.22 [0.011, 0.44]). Within-layer layer map: Spanish supported (0.29 [0.12, 0.44]), Hindi not (0.17 [-0.02, 0.36]). Shared-key edit: gain at most 0.047, null. Wording loss beyond rewording in all four cells (Spanish 0.39 and 0.26, Hindi 0.79 and 0.73). 7 of 8 registered sub-predictions met. One model so far.

## 3. Plan (status after the Kaggle run)
| # | What | Status |
|---|---|---|
| 1 | Paraphrase control: does an English edit carry to other English phrasings, and is Spanish as good? | done on Kaggle; repeated at about 3x the facts in the next run |
| 2 | More fact types (non-geography relations, 100+ facts) | open: first try untestable; needs more relations, registered first |
| 3 | Native-speaker check of Hindi and Spanish facts (`data/audit/native_speaker_audit.csv`, 362 facts) | open: needs a Hindi checker and a Spanish checker from the user |
| 4 | More models (section 5) | planned |
| 5 | New edit method: key built from several languages | done on Kaggle, null: no improvement |
| 5b | **Key similarity predicts which facts transfer** (new possible Q1 hook) | run on Llama-3.2-3B-Instruct: supported for Spanish (both tests), Hindi narrow; needs a second model |
| 6 | More languages (2-3, one non-Latin non-Indic script) | not started |
| 7 | Standard benchmark slice (Bi-ZsRE English-Chinese on Qwen2.5-3B) and baselines (MEMIT-style, in-context) | registered as A26 Tests 2 and 3, code not built |
| 8 | Paper repairs | done 2026-10-29 |
| 9 | OSF/Zenodo timestamp; revoke the Hugging Face token; delete the old Kaggle notebooks | open |

## 4. Run path (what to run, in order)
1. **Done 2026-10-31:** `C:\Users\Som Praneeth\Desktop\reasearch\babel-effect\kaggle\babel_llama.ipynb` (code id `bundle-f8371a452538`; the run already in progress is `bundle-21a9baef1028`, same science, only two Spanish analysis jobs fail there). Llama-3.2-3B-Instruct.
   - Only input: the Kaggle Model Meta Llama 3.2, Transformers, 3b-instruct (the same as run 1). No dataset: the code, fact files and tokenizer files are packed inside the notebook. Do not attach the old run's Output.
   - Needs: GPU T4 x2 and Internet on (package install, WikiText-2).
   - Download the Output and send it back; then I read the A28 verdicts.
2. Gemma-2-2B notebook: built and CPU-checked 2026-10-31 (`kaggle/babel_gemma2.ipynb`, A29 registered); waiting for the user to run it. Input: Kaggle Model Gemma 2, Transformers, gemma-2-2b.
3. Llama-3.2-3B base notebook (needs its Kaggle Model path).
4. Qwen2.5-3B, then BLOOM-1b7 (pilot kept only 34 facts, needs a larger fact pool).
5. One 7-8B model only after a T4 test (both T4s or 4-bit; untested; 4-bit changes the edited weights).
Each new model is registered in `docs/HYPOTHESIS.md` before its first run.

## 5. Models
**Ours (committed):** Llama-3.2-3B-Instruct (primary), Gemma-2-2B, Llama-3.2-3B base.
**To add (from the models existing work used):** Qwen2.5-3B, BLOOM-1b7; later, if a T4 test passes: Llama-2-7B, Llama-3.1-8B, Qwen2.5-7B, Mistral-7B-v0.1, optionally BLOOMZ-7b1.
**Dropped:** Llama-3.2-1B-Instruct (kept 38 facts) and BLOOM-1b7's first pilot (34 facts): below the 50 needed.

**What existing work used** (read from retrieved paper passages on 2026-10-30, not full reads; XGLM/EuroLLM/mT5 sizes and CLEVER-CKE models were not in the passages):
| Paper | Models |
|---|---|
| Bi-ZsRE (2309.08952) | Chinese-LLaMA-Plus-7B, Chinese-LLaMA-2-7B, Baichuan-7B, Baichuan2-7B |
| XME (2401.10521) | BLOOM-560M, mBERT-base, XLM-RoBERTa-base |
| MLaKE (2404.04990) | Vicuna-7B-v1.5, Qwen1.5-7B-Chat (edits); Llama-2-7B-chat, Gemma-7B-IT, Mistral-7B-Instruct-v0.2 (QA only) |
| CLM-Bench (2601.17397) | Llama2-7B-Chinese, Qwen2-7B, Llama3-8B, Mistral-7B |
| Fierro (2410.14387) | XGLM, EuroLLM, mT5; GPT2-XL (English only) |
| Ifergan (2408.10646) | BLOOM-7B, Qwen-7B, Llama-2-7B, Mistral-7B-v0.1, Chinese-Llama-2-7B, Hebrew-Mistral-7B |
| Dumas (2411.08745) | Llama-2 7B (main), Llama-3 8B, Llama-2 70B, Mistral 7B v0.3, Aya 23 8B, Gemma 2B |
| LU-LAFNs (2406.16416) | Llama-3.1-8B, Qwen2-7B, bloomz-7b1-mt; Llama-2-7B (appendix) |
| ReMaKE (2312.13040) | Llama-2-7B, BLOOMZ-7b1-mt |
| LangEdit (2506.10800) | GPT-J-6B, Llama3-8B, Qwen2.5-7B |
| X-KDE (2502.14645) | Llama-2-7B-chat, Qwen2.5-7B-Instruct (named in its appendix) |
| MEMAT (2502.02173) | Aguila-7B (English and Catalan) |
| MEMLA (2406.11566) | mGPT-1.3B |
| Arabic editing (2507.09629) | Llama-2-7B-Chat |
Facts: almost every paper uses 6-8B models; only BLOOM-560M, mGPT-1.3B and Gemma 2B are as small as ours. None used Llama-3.2 or Gemma-2 in these passages, so none of their numbers is directly comparable to ours. Hindi appears only in XME and Ifergan; Bi-ZsRE and MzsRE have no Hindi. Model counts: Llama-2-7B in 6 papers, Qwen in 6, Llama-3/3.1-8B in 4, Mistral-7B in 4, BLOOM family in 4.

## 6. Published results vs what the next notebook gives
| Topic | Published | Our Kaggle first pass | Next notebook (registered odds, not results) |
|---|---|---|---|
| One-language edit carries over? | LU-LAFNs paper, Table 7 (ROME, Llama-2-7B, Bi-ZsRE, average F1 of four metrics): English 72.96, Chinese 35.11 after an English edit (ratio about 0.48; my arithmetic). MLaKE, MEMIT: English to German/French about 0.4, to Japanese/Chinese about 0.12-0.15. XME, BLOOM-560M, MEND: English to Spanish 88.8 vs 93.0, to Hindi 65.6 (50 is chance). MEMLA, mGPT-1.3B, ROME, English edit: 100 in English, 78 Spanish, 56 Chinese, 60 Thai. | Spanish 0.50 (traced layer), 0.85 (layer 14); Hindi loses most | edits at layers 2/6/10/18/22 plus traced and 14; Spanish below 0.5 at layers 2 and 6 and at least 0.5 by layer 14 (65%) |
| Which layer | one default layer per paper | 0.50 vs 0.85 for Spanish | full transfer-by-layer curve (descriptive) |
| Loss beyond rewording | Bi-ZsRE has a same-language rewording column, so partly known | Hindi 0.83/0.66, Spanish traced 0.37, Spanish layer 14 0.13 | Hindi stays above 0.15 in both conditions (90%); Spanish layer 14 gets a clear answer |
| Shared multi-language key fix | one published multilingual fix gains about 2 F1 over ROME (passage, not full read) | gain 0.06-0.08, no improvement | gain below 0.15 in all 8 cells (75%), likely null again |
| Edit geometry | CLM-Bench: English vs Chinese edit-vector cosine 0.027-0.042, mixed edits additive (cosine above 0.976) | - | key cosine measured per fact and layer |
| **Key similarity predicts transfer per fact** | **not in the papers read**; CLM-Bench reports only an average | traced layer rho 0.55/0.52, layer 14 about 0; with layer control Spanish 0.45, Hindi 0.06 | Spanish traced rho at least 0.30 again (60%); Hindi survives layer control (30%); layer 14 near 0 (65%); within-layer rho holds for Spanish (50%), Hindi (30%) |
| Facts outside geography | not reported | untestable (2 relations) | reruns unchanged, expect the same |
Published shared-representation findings (not rerun here): Fierro (finding the subject is language-independent, object extraction language-dependent), Ifergan (sharing depends on script; consistent answers do not mean shared representations), Dumas (concept patching works across languages).

## 7. What decides the paper
- Spanish and Hindi both pass the layer-controlled key-similarity test: key similarity predicts transfer; this is the contribution (Q1 candidate).
- Only Spanish passes: partial claim (Q2).
- Neither passes: the diagnostic paper with the layer map, the wording gap and the null shared-key result (Q2-Q3).

## 8. Checks done on the next notebook (2026-10-30)
- Full suite: 49 tests pass, 1 skipped; also run on the code unpacked from the notebook exactly as Kaggle assembles it.
- Every job command: script exists, every flag exists in the script, every dependency exists; the folder pattern each job reads is made by a job it depends on (unit test).
- Key-similarity and layer-map scripts rerun on the real Kaggle run-1 data; the layer-map graph tested including undefined and empty layers.
- Fixed: both analysis scripts print results after saving them; a print error would have failed the job and the retry would have hit the write-once folder. Fixed a stale help line.
- Untested on a T4: the tracing job in fp16 (no resume flag, a retry restarts it), the larger scale, the time estimate. The tracing code ran in fp16 on the laptop GPU for 2 facts (7.5 GiB), which is not a T4 test.

## 9. Standing rules
Pre-register in `docs/HYPOTHESIS.md` before any data exist; write-once run folders; every number through `results/manifest.json`; never touch `..\RRAP`; commit or push only when the user says so; plain names in chat (no run codes); quote paths (they contain spaces). Kaggle notebooks must run with zero errors and keep both GPUs busy; rebuild the notebook (`.venv/Scripts/python kaggle/make_notebook.py`) after every code change.

## 10. Needed from the user
Run the notebook and send back the Output; the Kaggle Model paths for Gemma-2-2B and Llama-3.2-3B base; someone to check the Hindi and Spanish facts; an OSF/Zenodo account.
