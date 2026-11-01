# NEXT (to-do list, updated 2026-10-30)

## 2026-10-31 (latest): A28 run read, next steps
- [x] Llama-3.2-3B-Instruct notebook run finished and read (verdicts in docs/HYPOTHESIS.md, A28). Key similarity predicts which facts carry over: Spanish in both tests, Hindi only in the layer-controlled test (narrow). Shared-key fix: no gain. Wording loss in all cells.
- [ ] User rule 2026-10-31: use the models published papers used (order: Gemma-2-2B, Llama-3.2-3B base, BLOOM-560M (XME, has Hindi), mGPT-1.3B (MEMLA), BLOOM-1b7, Qwen2.5-3B, then 7-8B after a T4 test) plus baselines and a benchmark slice on the same facts; Kaggle weekly GPU quota (about 30 h, unconfirmed) limits this to about 4 notebooks a week.
- [ ] **Run next:** `kaggle/babel_gemma2.ipynb` (Gemma-2-2B, registered as A29 in docs/HYPOTHESIS.md, built and CPU-checked 2026-10-31; see the last block of docs/LOG.md). Only input: Kaggle Model Gemma 2, Transformers, gemma-2-2b (base). NOT the dataset leimeng46/gemma-2-2b (it is the 18-layer first Gemma, refused by the notebook). Expected 3 to 5 h on 2x T4 if fp16 is accepted. Untested on a T4. After the Output: read the A29 verdicts, then Llama-3.2-3B base (same pattern, needs its own entry in kaggle/steps.py MODEL, common.py, a config pack and a fingerprint).
- [ ] Replicate on a second model: Gemma-2-2B first (built), then Llama-3.2-3B base. The key-similarity claim needs at least one more model before it is a headline.
- [ ] Rebuild the notebook for new models from the current source (layout fix included); run every job command in a clean unpacked payload first.
- [x] Manifest (2126 claims) and paper Section 4.8 for the A28 numbers, figures 7-8 added (2026-10-31). Still open: full read of the five closest papers before the novelty wording is firmed up; the same section for Gemma-2-2B after the A29 Output.
- [ ] Still open: Hindi and Spanish checker for the audit sheet; OSF/Zenodo timestamp; revoke the Hugging Face token and delete old Kaggle notebooks; more non-geography relations (register first); baselines and benchmark slice (A26 Tests 2 and 3).
- Commit and push only when the user says so.

## 2026-10-30 (latest): run path and models (master file: GOAL.md)
- [ ] **Run now:** `kaggle/babel_llama.ipynb`, code id bundle-f8371a452538 (Llama-3.2-3B-Instruct; the run in progress is bundle-21a9baef1028). Only input: Kaggle Model Meta Llama 3.2, Transformers, 3b-instruct; no dataset; do not attach the old Output. About 6 h on 2x T4. Send back the Output.
- [ ] After the Output arrives: run the Spanish non-geography analysis twice (plain and leak-free) on the Output's e2a23_A run with the current code; the two Kaggle jobs failed because the laptop tracing run was not packed (fixed in kaggle/layout.py, notebook not rebuilt; rebuild before the Gemma and Llama-base notebooks).
- [ ] Read the A28 verdicts from the Output (key similarity beyond the edit layer, layer map, shared-key and wording rules at about 3x the facts).
- [ ] Next notebooks, in order: Gemma-2-2B, Llama-3.2-3B base (both need their Kaggle Model paths), Qwen2.5-3B, BLOOM-1b7 (needs a larger fact pool), then one 7-8B model after a T4 test. Register each model in docs/HYPOTHESIS.md before its run.
- [ ] Still open: Hindi and Spanish checker for `data/audit/native_speaker_audit.csv`; OSF/Zenodo timestamp; revoke the Hugging Face token and delete the old Kaggle notebooks; more non-geography relations (register first); benchmark slice and baselines (A26 Tests 2 and 3).
- Commit and push only when the user says so.

## 2026-10-30 (later): Kaggle only, make the new results the best
- [x] Key-similarity test run on the Kaggle data: traced layer rho 0.55 (Spanish) and 0.52 (Hindi), layer 14 none; weakens when the edit layer is controlled (Hindi 0.06). Registered A28 to settle it. New notebook `kaggle/babel_llama.ipynb` (code id bundle-6d5af7908026): traces on Kaggle, about 85 facts, layer map, key similarity, all in one run (about 6 h on 2x T4).
- [ ] User runs the notebook and sends back the Output; then read the A28 verdicts (key similarity beyond the edit layer, layer map, shared-key and wording rules on the larger runs).
- [ ] If the within-layer relation holds: write the contribution as "key similarity predicts which facts transfer, at early layers"; if not: the diagnostic paper reports the layer map and the null shared-key result.
- [ ] Then the next model notebooks (Gemma-2-2B, Llama base; need their Kaggle Model paths) with the same step; baselines (MEMIT-style, in-context on held-out wordings) and the benchmark slice stay registered in A26 Test 2 and 3.
- Commit and push only when the user says so.

## 2026-10-30: first Kaggle results (Llama-3.2-3B-Instruct, verdicts in docs/HYPOTHESIS.md)
- [x] Paraphrase control, shared-key edit, non-geography facts, tokenizer comparison run on Kaggle (73.9 min, 2x T4, fp16). Shared-key edit: no improvement (Plan B stands). Non-geography facts: untestable (53 EN/ES and 13 EN/HI facts, two relations). Paraphrase control: cross-language loss beyond wording for Hindi and for Spanish at the traced layer; single-edit numbers do not reproduce the laptop's (median-level numbers do).
- [ ] Decide with the user: next model notebook (Gemma-2-2B, then Llama base; same pattern, needs the Kaggle Model path), and what to do about the single-edit non-reproduction (rerun the laptop edits in fp16? report as a limitation?).
- [ ] A23 needs more relations that Llama answers (currency and profession failed the English template filter); register before any new facts are run.
- [ ] Key-similarity per-fact test (register first): the A24 records already hold key cosines per fact.
- [ ] Hindi + Spanish checker, OSF/Zenodo timestamp (still open). Commit and push only when the user says so.

## 2026-10-29 (late): Kaggle rerun and the contribution
Chosen contribution: mechanism + fix. Claim: the read-out is shared, but a rank-1 edit is keyed on a language-specific form of the subject; key similarity across languages predicts per-fact edit transfer; a key built from several languages closes the gap without training. Compare with ROME, MEMIT, in-context and one published multilingual editor (language-agnostic factual neurons 2406.16416 or MEMAT 2502.02173) on our controlled facts and a Bi-ZsRE/MzsRE slice. Plan B: diagnostic paper (likely Q2-Q3).
- [x] Kaggle engine: per-model notebooks `kaggle/notebooks/babel_llama.ipynb`, `babel_gemma2.ipynb` (code from GitHub, no dataset to upload). Stage 1 (fact preparation) only; GPU parts untested until the first run.
- [ ] User runs both; I compare kept facts with the laptop (Llama 93, Gemma 58) and read the timings.
- [ ] Add the stages to the same notebooks: tracing, noise/phrasing, swap tests, edit sweep, in-context, paraphrase control, new relations, shared-key edit; then the Llama-base notebook.
- [ ] Pre-register before running: the key-similarity test (per-fact rho between EN-ES/EN-HI key similarity at the edit layer and transfer), the baselines, the benchmark slice (check which languages MzsRE has).
- [ ] Read in full: Fierro 2410.14387, Dumas 2411.08745, Ifergan 2408.10646, CLM-Bench 2601.17397, LU-LAFNs 2406.16416, MEMAT 2502.02173; update related work.
- [ ] Needs the user: Hindi + Spanish checker, OSF/Zenodo timestamp, Kaggle weekly GPU quota.

## 2026-10-29: plan to raise the level (see `GOAL.md`)
Order: A21 -> A23 -> A24 (needs A21) -> bigger models (7-9B, 4-bit; needs a feasibility check and the user's decision on a free cloud GPU) -> more languages -> benchmark slice.
- [x] CPU: paper repairs (A22 reconciliation, 2 citations, p-floor note, abstract/discussion/limitations, PDF rebuilt); A23/A24 pre-registered and coded; native-speaker audit sheet built.
- [ ] GPU, needs the user's laptop window and "go": `./scripts_a21.sh` (about 3 h), `./scripts_a23.sh` (about 3-4 h), then `./scripts_a24.sh` (about 6 h, after A21).
- [ ] Needs the user: a Hindi and a Spanish checker for `data/audit/native_speaker_audit.csv`; yes/no on a free cloud GPU for 7-9B; Zenodo/OSF timestamp.
- [ ] CPU, after each run: manifest blocks (`a21_*`, `a23_*`, `a24_*`), paper paragraphs, rebuild PDF, refresh `..\evaluator_pack`.
- [ ] CPU later: 2-3 more languages (candidate facts), a mLAMA/Bi-ZsRE slice builder, 4-bit feasibility test script.

## Older list (2026-10-25)

## 2026-10-25 status
- [x] A12/A12a confound analysis, A13a independent-noise re-trace, A13b covariates, A13 E2 interchange (verdict SHARED EN-ES, both models; docs/EXPERIMENT_LOG.md). Paper section 4.7 drafted.
- [ ] GPU (needs laptop): E2 Hindi; E3 name-dissimilar facts (more relations); E4 base vs instruct. (Gemma F7 done 2026-10-25; Llama-1B pilot kept 38 facts and BLOOM-1b7 pilot 34, both below the 50 needed, so no full run.)
- [ ] CPU: related work + arXiv-verified citations; Llama in-context baseline anomaly (EN 0/10); main tables with 10,000 permutations; Fig. 4 buckets; reproduce script + clean-clone run; external pre-registration timestamp (Zenodo/OSF); full paper rewrite and abstract update.
- [ ] Open science question: tracing increment does not predict interchange per fact in Llama (rho -0.04); explain or drop the link.


Legend: [x] done, [~] running, [ ] to do. GPU items wait for the layer sweep (scripts_run_sweeps.sh). It was interrupted on 10-15 and 10-18 (laptop sleep) and resumed on 10-19 03:07 from saved rows; scripts_watchdog.sh restarts it if it dies. Estimate: Llama about 4-5 h more, Gemma about 6 h, then the queued chains.

## Done
- [x] P0-P5 on Llama-3.2-3B (v1 pilot) and paper draft v1; repo public (praann07/babel-effect)
- [x] Model research report (reports/Models for multilingual fact editing.md), Gemma-2-2B weights verified
- [x] A5-A8 pre-registered; v2 data (5 relations, 3 phrasings); full-answer tracing; Llama v2 (n=92) and Gemma-2-2B v2 (n=58) overlap; cross-model table (33 shared facts); related work (20 verified refs); manifest 356 claims

## Fixes to make the paper reviewer-proof (do one at a time, commit after each)
- [x] F1 Pre-register the relation-stratified null as the primary test for everything from now on (A9). CPU, now.
- [x] F2 Hindi diagnostic: Hindi accuracy and overlap versus Hindi token length (answers "is it a tokenizer artifact?"). CPU, now.
- [~] F3 In-context editing baseline (put the new fact in the prompt) to put the weight-edit transfer numbers in context. Code now, run after the sweep.
- [~] F4 Non-copyable facts: add relations whose answers differ across languages (landmark_country, city_country); target >= 50 non-copyable valid facts per model. Data now, GPU after the sweep.
- [~] F5 Phrasing robustness (H8): trace phrasings 1 and 2 on 30 facts per model. GPU after the sweep.
- [~] F6 Third model: forward-only pilot (Llama-3.2-1B first; Qwen3-1.7B-Base / BLOOM-1b7 next) and run the full pipeline only if it keeps >= 50 facts.
- [~] F10 BLOOM-1b7 as a third family (A10 pre-registered; weights downloading, chain scripts_run_bloom.sh queued behind scripts_run_post.sh). Tests whether weak Hindi overlap is a tokenisation effect.
- [ ] F7 More edit facts: after the sweep, rerun the best layers on about 30 facts per model.
- [ ] F8 Paper rewrite for v2 (tables, figures, honest verdicts on H6-H9), HYPOTHESIS verdicts, README.
- [~] F9 reproduce.sh: `./reproduce.sh check` (tests + manifest + figures) done; the full v2 end-to-end rerun from a clean clone is still to do.
- [x] CPU extras done 2026-10-19: paper figures (paper/figures), unit tests (tests/), watchdog.

## Hypotheses status (docs/HYPOTHESIS.md, Verdicts section)
- H6 supported (both models). H7 supported in Gemma, contradicted for ES in Llama. H8 supported (both). H9 not supported. F4 met for Llama, not Gemma. F3/F7 descriptive (Llama F7 done).
- B1-B3 (A10, BLOOM-1b7): pending.

## Left to run (GPU; start with Desktop "Start Experiments.bat" or the two nohup commands in the queue2 script header)
1. Gemma F7 (20 new facts; partial dir analysis/run_20261020_p4f7_gemma2 is kept unused, the queue starts a fresh one)
2. BLOOM-1b7 pilot (A10; TransformerLens BLOOM path untested)
3. Llama-3.2-1B pilot (full pipeline only if >= 50 facts)

## Left to write (CPU)
- Add Gemma F7 and the pilots to paper section 4.5/5 once run; related-work additions after verifying 2411.08745; README refresh; reproduce.sh for the v2 pipeline + one clean-clone rerun; fix Start Experiments.bat.

## Waiting on the user
- Venue choice; optional Hindi-speaker spot check; keep the laptop plugged in while the GPU queue runs.

## 2026-10-29 (night): laptop only; comparison with published work done
- Kaggle dropped. Registered in docs/HYPOTHESIS.md as A26: key-similarity test, baselines (MEMIT-style, in-context, published editor), Bi-ZsRE English-Chinese slice on Qwen2.5-3B.
- Next, in order: (1) go from the user, then the paraphrase control, shared-key edit and non-geography runs; (2) CPU: key-similarity analysis script; (3) write MEMIT-style editor, Bi-ZsRE loader and scorer, Qwen entry in code/common.py, with CPU tests; (4) read MEMAT (2502.02173) and the Fierro, Dumas, Ifergan papers in full; (5) native-speaker audit and OSF/Zenodo timestamp.
- Finding from the reading: Bi-ZsRE has English and Chinese only, so Spanish and Hindi stay on our controlled facts. LU-LAFNs needs 30 paraphrases per question (the authors used a 72B API), which is a deviation under the $0 rule.
