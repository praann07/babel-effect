---
title: "The Babel Effect: master record"
subtitle: "Goal, story, problems faced, paper, pre-registrations, logs and paths: everything in one file"
date: "Built 2026-11-01 from the repository at paper/build_master.py"
---


# About this document

This is the **master record** of the project "The Babel Effect": one long file that holds everything we did, formal and informal, so that nothing lives only in a chat or in someone's head. It is the go-to document. The short paper is one chapter of it (Part 5). It is written to be revised: change a source file, rerun one command, get a new PDF.

**Written:** 2026-11-01 (your clock, IST). **State of the work at writing time:** Llama-3.2-3B-Instruct finished on Kaggle and in the paper; Gemma-2-2B running on Kaggle (fourth attempt), no Gemma result read yet; Llama-3.2-3B base and four more models not started.

**Time zones.** Kaggle console times are UTC. Your clock is IST, which is UTC plus 5 hours 30 minutes. Every time in this record says which one it is, or is a date only.

## How this record is built (so you can revise it)

| What | Where | How to change it |
|---|---|---|
| Chapters written by hand (Parts 1 to 4, 6, 7, 9) | `paper/master/*.md` | Edit the file, rebuild. |
| The paper (Part 5) | `paper/paper_v2_draft.md` | Edit the paper; everything here follows. |
| Master goal and plan file (copied into Part 10) | `GOAL.md` | Edit `GOAL.md`. |
| Pre-registrations, decision log, experiment log, plans, external review (Parts 8 to 10) | `docs/*.md` | Edit those files; they are copied in word for word. |
| Lists of paths, run folders and git history (Part 9) | generated | Rebuilt from the repository each time. |

**Rebuild:** `.venv/Scripts/python.exe paper/build_master.py` from the repo root. Needs `pandoc` and Microsoft Edge. Output: `paper/The_Babel_Effect_MASTER_RECORD.pdf` (and the `.md` it was built from).

## What is checked and what is not

- The numbers in the paper chapter (Part 5) are checked against files by `results/manifest.json` (2126 claims verified on 2026-11-01 before the first commit of this record).
- Numbers in the hand-written chapters are copied from the files named next to them (`docs/HYPOTHESIS.md`, `GOAL.md`, `docs/LOG.md`). They are not machine-checked; if one disagrees with the paper chapter, the paper chapter and the files win.
- Literature statements come from paper abstracts and retrieved passages, not full reads, unless the text says otherwise. Venues were not checked. Papers are cited by arXiv identifier.
- Copied documents (Parts 8 to 10) are unedited. They contain run codes (A1 to A30) and many dates; the chapters before them give the plain-words version.

## Contents

1. **Part 1. Goal, rules and what we are trying to prove.** The founding brief, how the goal changed, the standing rules.
2. **Part 2. The story in order.** What we did from 2026-10-12 to 2026-11-01, with the key numbers.
3. **Part 3. Problems faced.** Every setback we hit, its cause, the fix, and whether it is closed.
4. **Part 4. In your words, and my mistakes.** The instructions you gave in your own words, and the errors I made, stated plainly.
5. **Part 5. The paper.** Current draft, word for word.
6. **Part 6. Results at a glance, and what decides the paper.** One page of numbers.
7. **Part 7. Open items and next steps.** Including what to do when the Gemma Output arrives.
8. **Part 8. Pre-registrations.** `docs/HYPOTHESIS.md`, word for word.
9. **Part 9. Where everything is.** Paths, commands, notebooks, run folders, git history.
10. **Part 10. Logs and plans.** `docs/LOG.md`, `docs/EXPERIMENT_LOG.md`, `docs/NEXT.md`, `docs/ROADMAP_Q1.md`, the master goal file, the external review of 2026-10-23, and the draft of the control analysis (A12a).


# Part 1. Goal, rules and what we are trying to prove

## 1.1 The question

Does a language model keep a fact in separate per-language circuits, or in one shared, language-independent space? And what does the answer mean for **editing** a fact in one language (change "the capital of X is Y" in English and see whether Spanish and Hindi follow)?

Languages: English (EN), Spanish (ES), Hindi (HI). Hindi is the hard case: a different script, longer token sequences.

## 1.2 The founding brief (2026-10-12), word for word

This is the instruction the project started from. It sets the method, the phases and the gates. Parts of it were later overtaken (the laptop-only rule gave way to Kaggle for new evidence; see 1.3), but the method and the honesty rules still hold.

```text
You are an expert mechanistic interpretability researcher and elite PyTorch engineer. Project: "The Babel Effect" — cross-lingual localization and editing of factual knowledge in small open-weight LLMs. Target: Full conference or journal submission (ACL / EMNLP / ICLR / TMLR). Undergraduate first-author paper; extreme methodological rigor must compensate for compute scale.
RESEARCH QUESTION
Are stored facts in language-specific circuits or a language-agnostic conceptual space? Test: (1) causally trace factual recall in EN, ES, HI; (2) quantify circuit overlap against a permutation null; (3) apply a localized rank-1 MLP edit at the traced site in one language; (4) measure cross-lingual transfer, specificity, and damage.
HARD CONSTRAINTS
$0 API spend; all experiments on one laptop GPU (<=8GB VRAM).
Open weights only. Primary: Llama-3.2-3B-Instruct. Fallback ladder (only if absolute OOM after all optimizations): Gemma-2-2B -> Pythia-1.4B -> GPT-2. Log every substitution with reason in docs/LOG.md.
TransformerLens HookedTransformer (or nnsight) for all activation access.
VRAM Protocol (Push the limit): Aggressive optimization is required. Primary attempt: 4-bit weights (bitsandbytes) + fp16/bf16 activations + chunked activation caching + CPU offloading for large cached tensors. Free caches explicitly after every run. Use torch.no_grad() for all tracing. Do not downscale the model unless OOM persists after chunking and offloading are fully implemented.
Architecture Note: Adapt rank-1 ROME-style updates for SwiGLU architectures (e.g., targeting the down_proj matrix while holding gate_proj/up_proj constant, or using a unified MLP abstraction).
METHOD PROTOCOL (follow exactly; cite Meng et al. 2022 for tracing/ROME)
Corruption: Gaussian noise (sigma = 3x subject-embedding std; seed fixed per fact, stored in facts.json) on subject-token embeddings; verify P(object) collapses.
Tracing: Restore clean activations per (layer, token) site on residual stream and MLP outputs; attribution = indirect effect = logit(object) - logit(top competitor), patched minus corrupted.
Critical Site: Per language = argmax attribution at that language's last subject token. Crucial: Account for BPE tokenization length differences across EN/ES/HI (e.g., verify token span alignment; use proportional indexing if token counts diverge >2). Store full grids.
Cross-lingual Comparison: Primary = per-layer attribution profiles at each language's own critical token (cosine + Spearman correlation); secondary = span-pooled top-k grid IoU (k in {5,10,20}). Null = shuffle fact IDs across languages, 1000 permutations (vectorized via NumPy/SciPy) -> compute z-scores and p-values.
Editing: Rank-1 ROME-style MLP update at (a) EN traced peak layer, (b) fixed mid layer L//2, (c) random layer control.
Metrics: dP(new object) per language; transfer ratio = dP_L2 / dP_EN; specificity = max |dP| over 20 unrelated control facts; damage = perplexity ratio on 100 held-out Wikipedia sentences.
Statistics: n >= 50 facts post-filter; bootstrap 95% CIs over facts; report median +/- CI; log all seeds.
PHASES WITH GATES (never skip; update docs/NEXT.md at each gate)
P0 Env: venv, pinned requirements.txt, GPU/VRAM report, TransformerLens smoke test. Gate: Probe prompt predicts "Paris" under 4-bit + chunked caching without OOM.
P1 Data: /data/facts.json, 150 triples templated in EN/ES/HI; keep only facts correct in all 3 languages (target n>=50). Gate: Accuracy table saved, token span alignment verified.
P2 Tracing: /code/causal_trace.py on 10 facts; heatmaps to /analysis/. Gate: Mid-layer peak visible, consistent with literature.
P3 Overlap: /code/circuit_overlap.py on all filtered facts + null. BEFORE computing, write predicted directions in docs/HYPOTHESIS.md and commit (pre-registration). Gate: Overlap z-scores with CIs. (NOTE: P0-P3 forms the core empirical backbone for a full conference/journal paper. Rigor must meet full venue standards, not just workshop level).
P4 Editing: /code/knowledge_edit.py, 3 conditions x 10 facts. Gate: EN edit success >= 80%.
P5 Transfer: /code/eval_transfer.py full battery. Gate: Transfer + specificity + damage tables complete.
P6 Writing: /paper full draft; results/manifest.json mapping every numeric claim to an artifact path; reproduce.sh runs P1-P5 from a clean clone. Gate: reproduce.sh passes.
STANDING TASKS
CODE: Modular, type-hinted, argparse configs; one config JSON saved per run; run dirs /analysis/run_<date>_<tag>/ are write-once, never overwrite raw outputs.
ANALYSIS: Seaborn heatmaps and tables; every claim gets a number + figure + artifact path.
WRITING: Strict academic tone, zero filler; cite only references you verified exist (check the arXiv ID), else mark [CITATION NEEDED].
HYGIENE: docs/LOG.md (decisions), docs/EXPERIMENT_LOG.md (run ID | hypothesis | config | result | verdict), README with exact commands, git commit at every gate.
OPERATING RULES
Work autonomously; fix your own errors; make routine implementation choices yourself.
Before any GPU run >5 min: provide a one-line VRAM + runtime estimate. If over budget, implement chunking/offloading first before downscaling the model.
Null results are results. If evidence contradicts the hypothesis, update docs/HYPOTHESIS.md honestly and continue; never massage or fabricate numbers; never delete raw outputs.
Message me only for: scope/budget/venue changes; a blocker persisting after 3 distinct fix attempts; or a gate failure that invalidates the core empirical backbone.
Status reports: max 10 lines, numbers first.
```

## 1.3 How the goal changed

| Date | What you said or decided | Effect |
|---|---|---|
| 2026-10-12 | Founding brief above. | Laptop GPU only, Llama-3.2-3B-Instruct primary. |
| 2026-10-25 | "we will do all the work, the conference we will decide later to which one we should submit... q1 or q3 or q2 journal that's the goal" | Target a journal (Q1, Q2 or Q3). Do not discuss the venue yet; do the science. |
| 2026-10-26 | "target q1 plan accordingly"; "it is about the work, not the paper; if the work is good we write the paper as good as we can" | `docs/ROADMAP_Q1.md` written. Science first, paper text after. |
| 2026-10-29 | Move the heavy reruns to Kaggle (2x T4 GPUs). One self-contained notebook per model, no GitHub download inside it. | `kaggle/driver.py` and `kaggle/make_notebook.py`. |
| 2026-10-30 | "forget about our laptop run, dont bring it into the picture, focus on the Kaggle run, make it the best" | New evidence comes from Kaggle only; compare with published work, never with the laptop runs. |
| 2026-10-31 | "whatever u give me I will run in the Kaggle, we will only get accepted if we have novelty, that's it." | Every notebook and claim is judged by what it adds that no paper has. |
| 2026-10-31 | Use the models the published papers used, plus baselines and a benchmark slice. | Model order set (Gemma-2-2B, Llama base, BLOOM-560M, mGPT-1.3B, BLOOM-1b7, Qwen2.5-3B, 7 to 8B after a T4 test). |
| 2026-11-01 | "write everything into the paper ... document each and everything formal informal ... we can revise it ... our go to huge pdf file" | This master record. |

## 1.4 What we claim, and what we will not claim

**Established (our data, Llama-3.2-3B-Instruct; details in Parts 5 and 6):**

- Facts are read from a shared subject representation (swap test: Spanish about 0.90 to 1.21, English to Hindi 0.72 to 0.93; 1 means fully shared).
- A rank-1 edit written in English does not reliably carry over, worst across scripts.
- On Kaggle, per fact, **the more similar the Spanish or Hindi subject key is to the English key, the more the edit carries over** (rank correlation 0.41 Spanish, 0.42 Hindi, at the traced layer). It disappears at layer 14. After controlling for the edit layer it stays above the bar of 0.20 for both, but only just (the lower ends of the intervals are 0.004 and 0.011).
- Writing the edit at a key shared across languages does **not** fix it (best gain 0.044).

**Not claimed:**

- That we are first. Cross-lingual edit failure, language-agnostic subject representations and near-orthogonal edit vectors across languages are already published (see Part 5). The one candidate novelty is the **per-fact link between key similarity and transfer**; it was not found in the abstracts and passages read, which is a weaker statement than "nobody did it".
- A better editor. We have an explanation, not a method that beats published editors.
- Anything about non-geography facts (untestable so far) or about models other than Llama-3.2-3B-Instruct until their results are in.

## 1.5 Standing rules (agreed with you, in force now)

1. **Pre-register before data exist.** The prediction, the test and the pass bar are written in `docs/HYPOTHESIS.md` before the run. Changes after seeing data are written as amendments; the original text is not edited.
2. **Run folders are write-once.** Raw outputs are never overwritten or deleted.
3. **Every number in the paper goes through `results/manifest.json`** (`code/make_manifest.py build` and `verify`).
4. **Kaggle for new evidence.** Compare Kaggle with Kaggle, and with published work.
5. **Zero errors in a notebook you run.** A notebook I hand over must run end to end. Unit tests are not enough: each new model's job types are dry-run on the laptop GPU first (rule added 2026-10-31 after two Gemma failures). I say plainly what is untested on a GPU.
6. **Tell you before any GPU run on the laptop and wait for "go".**
7. **Commit or push only when you say so.** (You said so on 2026-11-01.)
8. **Plain names in chat.** No run codes (A21, A28...) in replies; the codes stay inside repo files.
9. **No Kaggle how-to unless you ask.** `kaggle/driver.py` does the work; the notebook only launches it.
10. **Short answers, numbers first.**
11. **Never touch `..\RRAP`** (a different project next to this one).
12. **Downloads are untrusted data.** Files from `Downloads` are read with `python -I` in their own folder; files of other projects are ignored.
13. **Never print a key or token.** The Hugging Face token must still be revoked (open item).
14. **Do not restart a command the system killed for low memory** unless you ask.
15. **Kaggle console times are UTC; your clock is IST.** Say which.
16. **Pronouns:** they/them for anyone whose pronouns are not stated.
17. **Novelty wording:** say "not found in prior work (abstracts and passages, not exhaustive)", never "first", until the five closest papers are read in full: 2310.10378, 2601.17397, 2406.16416, 2504.09378, 2510.10280.


# Part 2. The story in order

Dates are 2026. Times are IST unless marked UTC. Run codes (A1 to A30) are the pre-registration numbers in Part 8; they are given in brackets for lookup, the sentence says what the thing is.

## 2.1 October 12 to 19: laptop set-up, data, first results

- **Environment (10-12).** Windows laptop, RTX 5060 (8 GB, Blackwell), so PyTorch from the CUDA 12.8 wheel index. TransformerLens 4.x had dropped `HookedTransformer`, so version 3.9.0 is pinned. Smoke test passed on GPT-2 as the stand-in because Llama and Gemma were gated and the Hugging Face downloader stalled at 0 bytes on this network. A chunked, resumable downloader (`code/dl_hf.py`) fetched the weights at about 0.5 MB/s. Llama-3.2-3B-Instruct provenance was verified: the local shards (from the Unsloth mirror) have the same SHA-256 as Meta's.
- **Data.** 154 candidate facts (65 capital_of, 39 landmark_city, 50 capital_country), hand-written in EN/ES/HI, filtered to the facts the model gets right in all three languages (Llama 93 kept, Gemma 58). Subject length differs by language (median 2, 3 and 5 tokens in EN, ES, HI); a span check found a contiguous subject span for every kept fact in every language.
- **Tracing and overlap.** Causal tracing (restore a clean activation at one layer and token, see how much of the answer returns) gives similar profiles for EN and ES in both models (z about 4 to 4.8 against a shuffled-fact null).
- **Edits.** Rank-1 edits on Llama at the traced layer, at layer 14 and at layer 21. English works (success 0.80 / 0.93 / 0.72). Spanish carries over only partly (0.58 / 0.75 / 0.00) and Hindi hardly at all (0.00 / 0.10 / 0.00).
- **Setbacks.** The layer sweep died twice (a reboot on 10-15; Modern Standby on 10-18). BLOOM-1b7 was added as a third model family on 10-19.

## 2.2 October 20 to 25: an outside review and the swap test

- A harsh outside review on 10-21 (another AI, with a hostile prompt, `docs/REVIEW_PROMPT.txt`) rated the draft as a reject for the main venues and named the missing controls: same-language, centred, and lexical-overlap. A second outside review followed (`docs/REVIEW_2026-10-23.md`). The second reviewer was partly wrong (a claim of 46 identical-prefix facts in Llama; the real number is 1) and partly right (shared noise seed across languages, unnormalised profiles, a Spearman tie bug). The controls were pre-registered (A12, A12a) before computing.
- **Swap test (interchange, A13).** Take the subject's internal state from one language, put it into another, and see whether the answer survives (1 = fully shared). EN to ES: 0.90 to 1.21, shared, robust to answer leakage and four clusterings of the bootstrap. My prediction that shared noise inflated the alignment was wrong.
- Pilots with Llama-3.2-1B and BLOOM-1b7 kept only 38 and 34 facts, below the 50 needed, so they were dropped.

## 2.3 October 26 to 29: Hindi, base model, literature

- **Hindi.** First attempt underpowered (23 and 12 source entities, below the floor of 25). A larger set (A16) gave a last-token result of 0.74 to 0.76, a lower bound because Hindi subjects span several tokens. The whole-subject version (A18) gave Llama 0.86 / 0.87 and Gemma 0.93 / 0.81: shared under four clusterings.
- **Base model (A19).** Llama-3.2-3B base replicates the Instruct result (EN to ES; HI to EN 0.86; EN to HI 0.72, partial).
- **Name-overlap reading revised (A22).** The earlier explanation that shared subject pieces explain the EN-ES overlap does not hold for Llama: the EN-ES increment is present in both groups of facts (0.30 and 0.40, with and without shared subject pieces).
- **Literature check (10-29, abstracts only).** Cross-lingual edit failure is already published (Bi-ZsRE, MLaKE, XME, CLM-Bench), as are language-agnostic subject representations (Fierro, Dumas) and near-orthogonal edit vectors (CLM-Bench). So these are not our novelty. The one candidate: per-fact edit-key similarity against transfer.
- **Three experiments registered, not yet run:** a paraphrase control (A21), new non-geography relations (A23), and a shared-key edit that writes the edit at a key built from several languages (A24).
- **Decision night (10-29).** Moving the reruns to Kaggle was first dropped, then reinstated the same night once the Llama weights were found as a Kaggle Model, which needs no token (A25, A27).

## 2.4 October 30: the first Kaggle runs (Llama-3.2-3B-Instruct, 2x T4)

- A first notebook crashed (numpy upgraded under a live kernel). The fix was a driver that runs every job in its own process (`kaggle/driver.py`).
- **Run 1** (code `bundle-05074174f20e`, 81.5 min): 13 of 15 jobs done. **Run 2** (code `bundle-f122bd5c1df0`, 73.9 min): all 15 done.
- **Findings on 28 facts:** the shared-key edit gives no improvement (gain 0.06 to 0.08 for Spanish at the traced layer, about 0 elsewhere); non-geography facts untestable (only two relations survive); wording explains part of the loss but not all (Hindi 0.83 and 0.66, Spanish traced 0.37, Spanish layer 14 inconclusive).
- **A sanity check failed:** single edits differ between laptop and Kaggle by 0.10 (median), with no median shift. So the rule from here: Kaggle is compared with Kaggle only.
- **Extra result, first pass:** per-fact key similarity predicts transfer at the traced layer (Spearman 0.55 Spanish, 0.52 Hindi on 28 facts). Not claimable yet: with the edit layer controlled, Hindi fell to 0.06.

## 2.5 October 30 to 31: scale-up to 86 facts

- **Registered before the data (A28):** one notebook that traces the facts itself, edits about 85 facts at the traced layer and at layer 14, maps five more layers, and runs the key-similarity test, wording control, shared-key analysis and graphs.
- **First attempt (code `bundle-21a9baef1028`, started 10-30 09:42):** tracing took 108 minutes (my guess was 30) and GPU 1 sat idle for 73 minutes. Two Spanish non-geography analysis jobs failed 3 of 3 because a run folder was missing from the layout list (fixed in source).
- **Fresh run (code `bundle-f8371a452538`):** 20 of 20 jobs, 348 minutes. Results, Llama-3.2-3B-Instruct, 86 facts (layer map on 40):

| Test | Spanish | Hindi | Bar |
|---|---|---|---|
| Key similarity vs transfer, traced layer (Spearman, 95% CI) | 0.41 [0.20, 0.59] | 0.42 [0.19, 0.62] | 0.30, CI above 0, Holm p |
| Same at layer 14 | 0.06 | 0.02 | none expected |
| With edit layer controlled (partial) | 0.26 [0.004, 0.49] | 0.22 [0.011, 0.44] | 0.20 |
| Layer map, within-layer, pooled | 0.29 [0.12, 0.44] supported | 0.17 [-0.02, 0.36] not supported | |
| Loss beyond rewording | 0.39 and 0.26 | 0.79 and 0.73 | 0.15 |
| Shared-key edit gain, best | 0.044 | | 0.15 (null) |

Seven of eight registered sub-predictions were met. Holm p was 0.0012 (Spanish) and 0.0008 (Hindi).

- **Paper:** Section 4.8 added (Tables 5 to 8, Figures 7 and 8), PDF rebuilt (20 pages), manifest 2126 claims verified.
- **Literature comparison (10-31, abstracts and passages only):** our direction agrees with published work (the edit reaches the same script, not the other); numbers are not comparable (published work uses 6 to 8B models). Closest to our key-similarity claim: 2310.10378 (language-pair level, six queries, BLOOM-3b), 2504.09378, 2510.10280, 2601.03168. Per-fact key similarity against transfer was not found.

## 2.6 October 31 to November 1: Gemma-2-2B on Kaggle

| Attempt | What happened |
|---|---|
| 1 (code `bundle-18a19884bb78`, 7.6 min) | Stopped at the smoke job. A base model answers "The capital of France is" with " a" (Paris 0.05); the question form "Question: What is the capital of France? Answer:" gives Paris 0.83. My check was wrong; the prompt was changed and fp16 is now also tested under tracing noise. |
| 2 (code `bundle-b7d6962e96d7`, 238 min) | 8 of 12 jobs done (tracing 44 min, English-key edits 80 min, both layer-map shards, wording analysis). The shared-key edit failed 3 of 3 times, so key similarity, the key-mode comparison and the graphs were skipped. The Output zip was not sent. |
| 3 | Stopped after 1 minute: the Kaggle Model for Gemma was not attached (setup, not a bug). |
| 4 (code `bundle-e186cf5d5e1b`, started 19:09 UTC on 10-31 = 00:39 IST on 11-01) | Running. At 21:19 UTC: no failure, shared-key edit at 35 of 52 facts. Expected end about 23:50 UTC (05:20 IST), an estimate. |

- **Cause of the attempt-2 failure, found on the laptop GPU (with your "go"):** for a fact whose subject text and left context are the same in two languages (e.g. "Madrid is the capital of" and "Madrid es la ciudad capital de"), the English and Spanish keys are the same vector (cosine exactly 1.0 for the Sri Lanka fact). The joint edit then has the same constraint twice; the matrix is singular and the solver raises. **Fix:** identical keys are kept once and the small solve is done in float64. Also: a later key mode may now fail without stopping the job (recorded in `mode_failures.json`), the console shows the last 14 lines of a job that fails, and a timing-fragile self-test got wide margins. 69 tests pass.
- **Dry runs on the laptop GPU (RTX 5060, fp16, peak 7.25 GiB):** both key modes ran on 4 Gemma facts with no failure. The run of the English-key base edits was killed by the system for low memory after 1 of 8 edits, so the key-similarity, key-mode and graph jobs have not yet been run on Gemma data.
- **Gemma-2-2B facts:** 26 layers, middle layer 13, edit cap 52 facts, post-MLP norm; fp16 chosen on the T4 (1.68 to 1.80 times faster, top-1 agreement 1.00, gradient cosine 0.999). Model input is the Kaggle Model `google/gemma-2/transformers/gemma-2-2b/1`, found by folder name and layer count and checked tensor by tensor against the laptop copy (288 tensors, 10.457 GB).
- **A suggested shortcut that did not work:** the Kaggle dataset `leimeng46/gemma-2-2b` is really `google/gemma-2b` (18 layers, the first Gemma); the notebook refuses it.

## 2.7 November 1: one folder, then commit

- All Gemma material and the Llama Kaggle results were copied into `C:\Users\Som Praneeth\Desktop\reasearch\BABEL_EFFECT_ORGANIZED\` (a copy, 69 MB; the repo stays the source of truth).
- At your word, the work was committed and pushed to GitHub (praann07/babel-effect, branch `main`) in groups: code and tests, notebooks and data, run outputs, then docs and paper with this record.


# Part 3. Problems faced

Every setback we hit, what caused it, what we did, and whether it is closed. "Mine" marks an error I made. Details and exact numbers are in the decision log (Part 10) and the pre-registrations (Part 8).

## 3.1 Set-up and hardware

| # | Problem | Cause | Fix | Status |
|---|---|---|---|---|
| 1 | Llama and Gemma could not be downloaded | Gated on Hugging Face; the stock downloader stalled at 0 bytes on this network | Access requested; chunked resumable downloader `code/dl_hf.py` (about 0.5 MB/s); GPT-2 used only as a pipeline smoke test | Closed. Later runs use Kaggle Models, no token |
| 2 | PyTorch had no GPU support | The RTX 5060 (Blackwell) needs the CUDA 12.8 wheels | Installed from the cu128 index | Closed |
| 3 | TransformerLens would not import `HookedTransformer` | Version 4.x removed it | Pinned `transformer-lens==3.9.0` | Closed |
| 4 | Memory blow-up on the 8 GB GPU | One forward pass peaked at 7.49 GiB; TransformerLens copies weight matrices on every pass and autograd kept about 1.2 GiB alive per pass, spilling a debug run to 14.5 GiB in shared memory | Weights laid out so the copies are not kept; caches freed per fact; chunked sweeps (details in the log, 2026-10-12 to 14) | Closed |
| 5 | The layer sweep died twice | A reboot (10-15) and Windows Modern Standby (10-18) | Resumable sweeps; `keep_awake.ps1`, watchdog and queue scripts | Closed |
| 6 | Weights provenance unclear | The laptop copy came from the Unsloth mirror | SHA-256 matched Meta's LFS objects; config and tokenizer files differ from Meta's, so the small files from the repo are used in the notebooks | Closed |

## 3.2 Science and statistics

| # | Problem | Cause | Fix | Status |
|---|---|---|---|---|
| 7 | Token counts differ by language | Median subject length 2 / 3 / 5 tokens in EN / ES / HI | Span check (`code/check_spans.py`): a contiguous subject span for every kept fact and language, 0 errors | Closed |
| 8 | Outside review: "reject" for the main venues | Missing same-language, centred and lexical-overlap controls; shared noise seed across languages; unnormalised profiles; a Spearman tie bug | Controls registered before computing (A12, A12a); the review's one false premise (46 identical-prefix facts) checked and rejected (real number: 1) | Closed |
| 9 | My prediction was wrong (mine) | I predicted shared noise inflated the EN-ES alignment | It matters only for Gemma equal-span facts; stated in the log | Closed |
| 10 | In-context baseline anomaly | Llama English statement fails in English (0/10) but works in another form | Diagnosed and written up (EXPERIMENT_LOG, 2026-10-25) | Closed |
| 11 | Hindi underpowered | 23 and 12 source entities, below the floor of 25 | Larger fact set (A16) | Closed |
| 12 | Last-token Hindi result looked low (0.74 to 0.76) | Hindi subjects span several tokens | Whole-subject (span-wise) version (A18): 0.86 / 0.87 Llama, 0.93 / 0.81 Gemma | Closed |
| 13 | The "name overlap explains it" reading | Not true for Llama | Revised after A22; paper Section 4.6 rewritten | Closed |
| 14 | Tracing does not predict which individual facts are shared | rho = -0.04 | Reported as a result, not hidden | Closed (a finding) |
| 15 | Laptop and Kaggle single edits differ | Median difference 0.10, no median shift (fp16 on T4 vs laptop) | Rule: compare Kaggle with Kaggle only; the notebook traces the facts itself | Closed |
| 16 | The shared-key edit gave no gain | It does not fix transfer (best gain 0.044) | Reported as a null; the diagnostic paper (Plan B) stands | Closed (a finding) |
| 17 | Non-geography facts untestable | Only two relations survive (official_language, book_author); 53 EN-ES and 13 EN-HI facts | Needs more relations, registered first | **Open** |
| 18 | The Hindi result is narrow | Lower end of the layer-controlled interval 0.011; layer-map pooled value 0.17 misses the bar | Needs a second model | **Open** (Gemma running) |
| 19 | Novelty is thin | Cross-lingual edit failure, shared representations and orthogonal edit vectors are already published | One candidate novelty: per-fact key similarity vs transfer; worded as "not found in prior work" | **Open** (five closest papers still to be read in full) |
| 20 | Hindi and Spanish facts not checked by a native speaker | No checker | `data/audit/native_speaker_audit.csv` (362 facts) is ready | **Open** (needs a Hindi and a Spanish checker) |

## 3.3 The Kaggle notebooks

| # | Problem | Cause | Fix | Status |
|---|---|---|---|---|
| 21 | First notebook crashed | Numpy upgraded under a live kernel | `kaggle/driver.py` runs every job in its own process (stdlib until pip, 3 attempts, resume) | Closed |
| 22 | Config and tokenizer files differ from Meta's | Laptop copy from a mirror | The notebook carries the laptop's small files | Closed |
| 23 | Two Spanish non-geography jobs failed 3 of 3 (Llama run, 10-30) | A run folder was missing from `kaggle/layout.py` | Fixed; the notebook was then checked unpacked in a clean folder; those jobs passed in the next run | Closed. My pre-run check covered scripts and flags, not the data files each job reads (mine) |
| 24 | Tracing took 108 min (guess 30); GPU 1 idle for 73 min | My estimate was off; tracing runs on one GPU | Re-estimated: whole run 348 min | **Open**: split tracing across both GPUs for later models |
| 25 | Progress bars count to 200 although 87 facts qualify | Bar total is the cap, not the number that qualifies | Noted | Cosmetic |
| 26 | Unit tests passed inside the repo but failed inside the notebook | Tests read repo-only files | Those tests skip inside a notebook; the notebook's own tests are now run unpacked after every rebuild | Closed |
| 27 | A timing self-test failed twice inside the Llama notebook folder (mine) | Margins too tight (1.17 vs 1.4) | Widened; a failing self-test stops the whole Kaggle run, so timing tests need wide margins | Closed |

## 3.4 Gemma-2-2B

| # | Problem | Cause | Fix | Status |
|---|---|---|---|---|
| 28 | Smoke job failed 3 times | A base model completes "The capital of France is" with " a" (mine: I had not checked the base model with this prompt) | Question-style prompt; smoke prints the top five tokens on a wrong answer; fp16 tested under tracing noise | Closed |
| 29 | Shared-key edit failed 3 of 3 | Identical EN/ES keys (cosine 1.0) make the joint-edit matrix singular | Identical keys kept once; solve in float64; later key modes may fail without stopping the job; failure text visible in the console | Closed in code; not yet seen on the GPU in Kaggle |
| 30 | My first theory was wrong (mine) | I wrote that identical subject names were ruled out because Llama had 24 such facts with key cosines below 0.995. What matters is an identical key (same subject text and same left context), not the same name | The amendment text was corrected before the final notebook | Closed |
| 31 | The failure had no traceback in the console | The driver hid the job log | Driver prints the last 14 log lines of a failed job | Closed |
| 32 | I handed over a notebook that failed twice on the real model (mine) | Unit tests and flag checks do not run the real model | Rule: dry-run every new model's job types on the laptop GPU first (done for Gemma, 4 facts) | Closed (rule in force) |
| 33 | My dry-run harness failed (mine) | A one-fact file left no answer pool; a three-fact file left too few valid facts for controls | Harness `run_one_fact.py` patches the fact selection and uses the full trace file | Closed |
| 34 | Base-edit dry run killed | The system ended it for low memory after 1 of 8 edits | Not restarted (standing rule) | **Open** (optional: rerun, about 25 min, other programs closed) |
| 35 | Third attempt stopped after 1 minute | The Gemma Kaggle Model was not attached | Notebook says so plainly | Closed (setup) |
| 36 | A suggested dataset was the wrong model | `leimeng46/gemma-2-2b` is Gemma 1 (18 layers) | Notebook refuses it; Google's own Kaggle Model is used | Closed |
| 37 | Key-similarity, key-mode and graph jobs never run on Gemma data | Attempt 2 stopped before them; attempt 4 not finished | Run on the real Output when it arrives; CPU-only, can be rerun locally | **Open** |

## 3.5 Communication and process

| # | Problem | Fix |
|---|---|---|
| 38 | You were angry about errors ("how code shouldnt have errors"). | Zero-error rule; dry runs before hand-over (Part 1, rule 5). |
| 39 | Run codes in my replies confused you. | Plain names in chat; codes stay in files. |
| 40 | I stated Kaggle times in UTC without saying so, so "NOW ITS 2:50" did not match. (mine) | Always say UTC or IST. |
| 41 | A detached background subshell made a task notification arrive early. (mine) | Run the process directly in the background. |
| 42 | Closing the assistant app would stop the background run, because it is a child of the session. | Told you not to; the Kaggle run itself is unaffected by closing the laptop app, but my laptop jobs are. |
| 43 | The Kaggle weekly GPU quota is not confirmed (about 30 hours is a guess). | **Open**: check on Kaggle before the next models. |
| 44 | A Hugging Face token was used earlier. | **Open**: revoke it. |


# Part 4. In your words, and my mistakes

This part keeps the informal side of the project: what you asked for in your own words (typed as you typed them, spelling kept), how the requests came, and the errors I made, stated plainly. Nothing here is edited to sound better.

## 4.1 Your instructions, in your words

| Date | You wrote | What it changed |
|---|---|---|
| 2026-10-25 | "we will do all the work, the conference we will decide later to which one we should submit... q1 or q3 or q2 journal that's the goal" | Journal target; no venue talk. |
| 2026-10-26 | "target q1 plan accordingly" and "it is about the work, not the paper; if the work is good we write the paper as good as we can" | Q1 roadmap; science first. |
| 2026-10-29 | "dont use A12 A33 i dont like it make it easier so i will understand" | No run codes in chat. |
| 2026-10-29 | "SHOULDNT HAVE ANY ERRORS ... CONSUME ALL OF THE KAGGLE GPU" | Zero errors; both GPUs busy. |
| 2026-10-29 | One unified notebook per model that runs the checks first, then everything; you run it, download the Output, send it back. No GitHub inside the notebook; all code packed inside. | `kaggle/make_notebook.py`. |
| 2026-10-30 | "forget about our laptop run, dont bring it into the picture, focus on the Kaggle run, make it the best" and "if we get extra results make them best" | Kaggle-only evidence; strengthen a new result before claiming it. |
| 2026-10-31 | "whatever u give me I will run in the Kaggle, we will only get accepted if we have novelty, that's it." | Every notebook judged by what is new. |
| 2026-10-31 | "so we are the first doing work of this kind?" | My answer: no for the field; "not found in prior work" only for the per-fact key test. |
| 2026-10-31 | Use the models the published papers used; add baselines and a benchmark slice. | Model order and the plan in `GOAL.md`. |
| 2026-10-31 | "ok...SEE BY THIS TIME U SHOULD HAVE UNDEESTAND HOW CODE SHOULDNT HAVE ERRORS OMGG" | Dry-run rule (Part 1, rule 5). I said "go" was needed for GPU use; you gave it ("go"). |
| 2026-10-31 | "can i close munderdufflin i mean u" | My answer: no, the background run is a child of this session. |
| 2026-10-31 | "give in one line", "DATASET LINK", "CAN I RUN IN ANOTHER LAPOTP BUT SAME ACCOUNT?", "BRO SORRY TELL ME WHAT TO DO", "OR GIVE ME DATAET LINK", "NOTEBOOK PATHH" | Short answers; the notebook path every time it changes; the Kaggle Model route instead of a dataset. |
| 2026-11-01 | "NOW ITS 2:50 WHEN WILL THIS BE DONE?" | I had given UTC times without saying so. Now always UTC or IST. |
| 2026-11-01 | "I HAVE SENT U THE PATHS OF THE NOTEBOOK AND REUSLTS WRITE KEEP THEM IN ONE SINGLE FOLDER ORGANZE WHATEEVER WE DID EVERYTHING UNITL NOW" | `BABEL_EFFECT_ORGANIZED\` (a copy, 69 MB). |
| 2026-11-01 | "push and commit to the github" | Commits and push (the standing rule was: only when you say so). |
| 2026-11-01 | "now write eeveeyrhitng into the paper ...so that we are documenting everythign its okay document each and eerythign formal informal into the paper we can revise it make it bettwe lter in the same paper ui can also put paths we are assuming it as our go to huge pdf file which says eeeverytihg about the paper include goal and probelms faced everything too" | This master record. |
| 2026-11-01 | "COMMIT AND PUSH TO GITHUB DONT BE LATE..." | Pushed the earlier commits at once; this record is a later commit. |

## 4.2 How we worked

- You run the notebook on Kaggle and send the result back: a pasted console, a saved console notebook from `Downloads`, or the Output zip. I read the files, I do not see Kaggle live.
- Questions came as short messages, often in capitals when something was wrong or urgent. The pattern: a status question ("how much time left?"), a path request ("NOTEBOOK PATHH"), a worry ("does this notebook have any erorrs like the previous one?"). I answer with numbers first and one decision.
- You prefer fewer round trips: one message per task, and the notebook doing its own checks.
- You said "go" before each laptop GPU run (rule: I tell you first and wait).
- You are the first author. The acceptance goal drives every choice: novelty first, then rigour.

## 4.3 My mistakes, plainly

1. **Wrong smoke-test prompt for the Gemma base model.** I had not checked it; the first Gemma attempt died at the smoke job after 7.6 minutes. Fixed with a question-style prompt.
2. **Handed over a notebook that failed on the real model, twice.** My checks covered scripts, flags and unit tests, not a real run of each job type. After your anger I adopted the dry-run rule and ran every Gemma job type on the laptop GPU.
3. **A wrong theory about the cause.** I wrote that identical subject names were ruled out because Llama had 24 such facts. The property that matters is an identical key, not an identical name. Corrected in the amendment text before the final notebook.
4. **Wrong time guess.** I guessed tracing at 30 minutes; it took 108.
5. **Wrong prediction in the science.** I predicted that shared noise inflated the EN-ES alignment; it did not (except for Gemma equal-span facts).
6. **UTC without saying so.** My "expected end" time was in UTC while you read it as your own clock. Now always labelled.
7. **A fragile self-test.** A timing test I wrote failed twice inside the Llama notebook folder; since a failing self-test stops the whole Kaggle run, that would have cost a run. Widened.
8. **Harness mistakes during the dry runs.** Two bad fact files (no answer pool; too few valid facts for the controls) cost time before I used the real fact list.
9. **An early pre-run check missed data files.** The first A28 Llama run lost two Spanish analysis jobs because a run folder was missing from the layout list. My check had covered scripts and flags only.
10. **Reading a shortcut too quickly.** The Kaggle dataset you suggested for Gemma was Gemma 1; I checked file sizes against the maker's repo and refused it.

## 4.4 Things that are true but easy to forget

- Nothing in the A28 result is a claim about a better editor: the shared-key fix was null.
- The Hindi layer-controlled result passes by a hair (lower end 0.011). It is a narrow pass, not a clean one.
- All strong numbers are from one model so far.
- The laptop experiments and the Kaggle experiments are different evidence; the paper keeps them apart (Section 4.8 is Kaggle only).
- The Gemma run that is running now was started from a notebook whose key-similarity, key-mode and graph jobs have not yet run on Gemma data anywhere.


# Part 5. The paper

*Copied word for word from `paper/paper_v2_draft.md` when this record was built. To change it, edit that file and rebuild.*

## The Babel Effect: Cross-Lingual Causal Profiles of Factual Recall in Two Small Language Models

*Draft v2 (work in progress; 2026-10-31: Section 4.8 added with the Kaggle run (A21 rewording control, A24 shared-key edit, A28 key similarity and layer map), with matching changes to the abstract, Sections 2, 3, 5-8; 2026-10-29 CPU pass: A22 reconciliation of the lexical split with the covariate model in Section 4.6, two added citations, p-floor note, updated discussion and limitations; Section 4.7 added 2026-10-25, updated 2026-10-28 with the span-wise English-Hindi test A18 and 2026-10-29 with the base-model replication A19 and 20 further edit facts A20). A first Hindi interchange (A15) was underpowered; pre-registered re-tests (A16, A18) are reported in Section 4.7. The BLOOM-1b7 and Llama-3.2-1B pilots kept too few facts (34 and 38 < 50) to run (A10). Every number is read from `results/manifest.json` or from a logged run (see `docs/LOG.md`). Hypotheses were pre-registered in `docs/HYPOTHESIS.md` (amendments A1-A19; A19, the Llama base-model replication, and A20, more edit facts, are reported in Sections 4.7 and 4.5).*

### Abstract

Do language models store a fact once, in a language-neutral form, or separately per language? We trace the causal effect of restoring clean activations when the subject of a factual prompt is corrupted, in English (EN), Spanish (ES) and Hindi (HI), for Llama-3.2-3B-Instruct (Grattafiori et al., 2024; 92 facts) and Gemma-2-2B (58 facts), and compare per-layer profiles with a shuffled-fact permutation null. English and Spanish profiles of the same fact are aligned beyond chance in both models (MLP-profile cosine z = 4.22 and 4.81, p <= 0.001; 0.0001 with 10,000 permutations). Hindi overlap is weaker: with a pooled null, Llama's EN-HI and ES-HI pairs are above chance (z = 2.45, 2.47) and Gemma's EN-HI marginally so (z = 2.19), but with a null restricted to the same relation only EN-ES remains significant after correction in both models. The weakness depends partly on Hindi subject length: with short Hindi subject spans the EN-HI cosine is 0.60 (Llama) against 0.20 with long spans. The layer where the fact is looked up is model-dependent (median layer 3 in Llama, 6 in Gemma). Edit transfer over every layer depends on the model: in Gemma mid-layer edits transfer more to Spanish and Hindi than early-layer edits, in Llama the opposite holds for Spanish, and Hindi transfer is small in both. The alignment is robust to rewording (within-language profile cosine exceeds the cross-language cosine in both models) and, in Llama, holds on 104 facts whose answers are not copyable across languages (EN-ES z = 4.87); an English statement in the prompt changes Spanish and Hindi answers in Llama more often than the weight edit does. A residual-stream interchange test then swaps the subject-token state between languages (Section 4.7): the answer follows the swapped state nearly as much as for a same-language swap for English-Spanish (ratio 0.90-1.17 in both models; 95% lower bounds >= 0.83 under four bootstrap clusterings) and, when the whole Hindi subject span is patched, for English-Hindi in both directions (ratio 0.81-0.93; every lower bound >= 0.70 under four clusterings; the single-token Hindi-to-English estimate of 0.74-0.80 was a lower bound). Controls added after review show that the English-Spanish alignment survives centring of the profiles and entity clustering in both models, but in Llama it is stronger in facts whose English and Spanish subject names share subword pieces (increment 0.40 against 0.30) and still present without them (Section 4.6). A Llama-3.2-3B base replicate keeps all four interchange labels within the registered margin of 0.15 of the instruct model (English to Hindi is the weakest, 0.72, PARTIAL). Pooled over 40 Llama facts, a rank-1 edit at layer 14 reaches Spanish with a median ratio of 0.75 against 0.58 at the traced layer and Hindi with 0.10 against 0.00, at about four times the side-effect cost (Section 4.5). A second run (Kaggle, 2x T4, fp16; Llama only; Section 4.8) re-fitted the edits on 86 facts and asked which facts carry over. When the edit is written at each fact's traced layer, the cosine between the English and the Spanish (or Hindi) MLP input at the subject token is associated with held-out transfer (Spearman rho 0.41 [0.20, 0.59] for Spanish and 0.42 [0.19, 0.62] for Hindi, 80 facts; Holm p <= 0.0012); with the edit layer as a further control the partial rho is 0.26 and 0.22 (lower bounds 0.004 and 0.011), and at layer 14 there is no association (0.06, 0.02). Within five fixed layers the association is supported for Spanish (pooled rho 0.29 [0.12, 0.44]) and not for Hindi (0.17 [-0.02, 0.36]). Writing the edit at a key shared by the three languages does not close the gap (paired median gain at most 0.044 against a registered 0.15), and Spanish and Hindi transfer stay below transfer to unseen English wording (difference 0.26-0.39 and 0.73-0.79). The association is a correlation on one model; the intervention built on it did not help. Profile similarity is consistent with partly shared lookup for English and Spanish and cannot be separated from subject-name overlap in Llama; the interchange test supports shared retrieval of the fact at the subject token across all three languages in both models, on four geography-type relations and two small models, and does not by itself show a single language-agnostic space.

### 1. Introduction

Multilingual language models answer factual questions in many languages, yet it is unclear whether a fact is stored once and read out through language-specific front and back ends, or stored separately per language. The answer determines whether an edit made in one language should carry over to others. We test it with two measurements on open-weight models that fit one 8 GB GPU: overlap of causal-tracing profiles across languages, and transfer of a localised edit from English to Spanish and Hindi.

Contributions: (i) a cross-lingual causal-tracing comparison with a shuffled-fact permutation null, a relation-stratified variant of it and a Hindi tokenisation diagnostic; (ii) results on two models with different lookup layers; (iii) an edit-transfer measurement over every layer, with robustness checks (rewording, non-copyable relations, an in-context baseline); (iv) a per-fact test of whether the similarity of the English and the Spanish or Hindi edit key predicts which facts an English edit carries over to, with a wording control, layer controls and a negative result for a shared-key edit (Section 4.8; one model); (v) a pre-registered design with logged deviations, all code, write-once run directories and a verifiable claim manifest.

### 2. Related work

**Locating and editing facts.** Causal tracing and rank-one editing of MLP weights follow Meng et al. (2022, ROME, arXiv:2202.05262) and its multi-edit extension MEMIT (Meng et al., 2022, arXiv:2210.07229). Geva et al. (2023, arXiv:2304.14767) describe recall as subject enrichment, relation propagation and attribute extraction. Hase et al. (2023, arXiv:2301.04213) find that the layer where tracing peaks does not reliably predict the best layer to edit, which motivates sweeping edit layers rather than relying on the traced layer. For activation patching itself we follow the advice of Zhang and Nanda (2023, arXiv:2309.16042) and Heimersheim and Nanda (2024, arXiv:2404.15255) on metrics and corruption choices, and we use their point that results depend on these choices to report both a same-language ceiling and a lexical-overlap control. Gupta et al. (2024, arXiv:2403.07175) and Yang et al. (2024, arXiv:2406.11263) trace model collapse after ROME edits to implementation details, which is relevant to how damage is measured.

**Multilingual recall mechanisms.** Wendler et al. (2024, arXiv:2402.10588) report that Llama-2 middle layers decode towards English before language-specific output. Fierro et al. (2024, arXiv:2410.14387) use patching on XGLM, EuroLLM and mT5 and find subject enrichment to be language-independent and object extraction language-dependent. Zhang et al. (2024, arXiv:2406.16416) locate language-agnostic factual neurons mostly in middle layers. Wang et al. (2025, arXiv:2504.04264) study cross-lingual factual inconsistency in LLaMA2-7B and BLOOM-560M. Lu et al. (2025, arXiv:2505.20546) analyse the multilingual factual-recall pipeline of Llama-3.2-3B, the model used here, in English, French, Spanish, Chinese, Japanese and Korean, without editing and without Hindi. Schut et al. (2025, arXiv:2502.15603) report English-like internal decisions in multilingual models. Dumas et al. (2024, arXiv:2411.08745) patch latents from a source translation prompt into a target prompt and find that the output language is encoded earlier than the concept, so the concept can be changed without changing the language; their task is word translation, whereas our interchange test (Section 4.7) swaps the subject of a factual-recall prompt across languages. Multilingual factual probes are provided by X-FACTR (arXiv:2010.06189) and mLAMA (arXiv:2102.00894).

**Cross-lingual knowledge editing.** Wang et al. (2023, arXiv:2309.08952) and Khandelwal et al. (2024, arXiv:2407.10275) find that parameter-based editing transfers poorly across languages while in-context editing transfers much better; Nie et al. (2024, arXiv:2406.17764) report in-context editing across 53 languages and a penalty for non-Latin scripts. Qi et al. (2023, arXiv:2310.10378) edit BLOOM-3b with ROME in English and find that the edit propagates to languages whose factual knowledge is more consistent with English, including Spanish. Banerjee et al. (2024, arXiv:2406.11139) apply ROME and MEMIT in English, Spanish, Hindi and five other languages on 7B models and report low portability across languages. Beniwal et al. (2024, arXiv:2401.10521) study cross-lingual editing in BLOOM, mBERT and XLM-RoBERTa, including Hindi, and find the largest limitations when the two languages use different scripts. Mousi et al. (2025, arXiv:2507.09629) find on Llama-2-7B-chat that parameter-based editing (ROME, MEMIT) generalises poorly from English to Arabic. Ifergan et al. (2024, arXiv:2408.10646) use editing itself as a probe of shared representation and report that high cross-lingual answer consistency does not imply shared representation, with script similarity the dominant factor; their result and ours agree that script matters, but we measure sharing with an interchange test at the subject token and relate it to edit transfer on the same model. Durrani et al. (2025, arXiv:2505.14393) survey the area.

Hu et al. (2026, arXiv:2601.17397, CLM-Bench) report that edits in one language do not propagate to the other on Llama-3 and Qwen2 and show that the edit vectors for Chinese and English are nearly orthogonal; this is a geometric statement about the edits, not about which individual facts carry over.

**Position of this work.** The studies above either trace or edit, mostly on 7B-class models, and none of them compares per-layer causal profiles across English, Spanish and Hindi against a shuffled-fact null and then measures edit transfer, specificity and damage over every layer on the same model. We searched for such a combination and did not find one; this is an absence in a bounded search, not proof that none exists. The same holds for Section 4.8: in the abstracts and passages we read, language-level similarity has been related to edit propagation (Qi et al., 2023, per language pair), but we did not find a test that relates the similarity of the English and the other-language edit key of each single fact to the transfer of that fact's edit. We have not read the closest papers in full, so this is a statement about what we found, not a claim of priority.

### 3. Method

**Models.** Llama-3.2-3B-Instruct (28 layers) and Gemma-2-2B (26 layers, base model), plus Llama-3.2-3B base (same architecture and tokenizer as the instruct model) for the interchange replication of Section 4.7 only, bf16, TransformerLens 3.9.0 without weight processing; one 8 GB GPU (peak 7.5 GiB). Weights are byte-identical (SHA-256) to the official repositories. Gemma is a base model and Llama an Instruct model; this confound is not separable here. A third family (BLOOM-1b7, pre-registered as A10, to test whether the weak Hindi overlap is a tokenisation effect) had not been run when this draft was written.

**Facts and phrasings.** Hand-written EN/ES/HI facts in five relations (capital_of, capital_country, landmark_city, invention_inventor, event_year; 408 candidates, 447 for Gemma after adding 39 events), three phrasings per fact and language. For each relation and language the three best of six answer-first templates are chosen on a 30-fact tuning set (per model). A fact is kept if the canonical phrasing begins with the answer (strict, after spelling normalisation) in all three languages and one other phrasing contains it in its first line. Kept: 93 (Llama: capital_of 27, capital_country 38, landmark_city 6, invention_inventor 0, event_year 22) and 58 (Gemma: 11, 11, 6, 0, 30).

**Causal tracing.** Gaussian noise (sigma = 3 x pooled subject-embedding std, per-fact seed) is added to the subject-token embeddings. A clean activation is restored at one (layer, prompt token) site of the residual stream or MLP output, and the effect is the change in the teacher-forced log-probability of the whole answer relative to the corrupted run (full-answer scoring; the first-token variant is stored). A pair is valid if the answer probability falls below half its clean value and the total effect is positive (92 of 93 and 58 of 58 facts).

**Overlap.** Per fact and language pair, cosine and Spearman correlation of the 28 (or 26) layer MLP-output profile at each language's own last subject token; statistic = median over facts; null = fact i in one language paired with a permuted fact in the other (1000 permutations; pooled across relations, and, as a post-hoc sensitivity analysis that becomes primary for later data, within relation); z and one-sided p from the null; 95% CIs by 2000 bootstrap resamples of facts. Six primary tests per model (3 pairs x 2 statistics), Bonferroni alpha 0.0083.

**Editing.** Rank-1 update of the MLP down-projection (ROME-style) at every layer for 10 facts per model, English prompt only, fitted on the full new answer; transfer ratio, specificity (20 controls) and damage (perplexity ratio on 100 WikiText-2 test sentences; Merity et al., 2016) as in A5. An in-context baseline (A9) is run for comparison, and a follow-up edit run (20 further facts) uses the traced layer, the middle layer (L//2, with L the number of layers; layer 14 in Llama, 13 in Gemma) and a random layer.

**Extra relations and rewording.** Two relations whose answers differ across languages (landmark_country, city_country) add 34 facts for Llama and one for Gemma (A9). Phrasings 1 and 2 are traced for the first 30 valid facts of each model to test robustness to wording (H8).

**Kaggle run (Section 4.8).** Section 4.8 comes from one notebook run on Kaggle (2x Tesla T4, fp16) with Llama-3.2-3B-Instruct; Sections 4.1-4.7 were run on a laptop GPU in bf16. The weights are those of Meta's Kaggle Model, compared tensor by tensor with the laptop copy before any job starts; fp16 was chosen by a registered equivalence check against bf16 on 24 fixed prompts. The notebook traces the kept facts itself, so the traced layers of Section 4.8 are its own and not those of Section 4.2. Config and tokenizer files are the laptop copy's, taken from a public mirror. Token ids of the text are identical in the mirror's tokenizer and in Meta's release for every one of 12,689 strings from the fact, template and candidate files (35 files); with special tokens every string differs, because the mirror prepends the beginning-of-text token by default. Edits: rank-1 ROME-style update of the MLP down-projection with the English key (the MLP hidden state at the last English subject token), fitted on one English phrasing, for 86 facts, at the fact's traced layer and at layer 14. Transfer to a language is the change in probability of the new answer in that language divided by the change in English, scored on the two other hand-written phrasings of the prompt in that language (held out from the fit); facts whose English edit changed the probability by 0.1 or less are dropped. The language key is the same hidden state at the last subject token of the Spanish or Hindi subject name, so it needs the translated subject. Key similarity is the cosine of the English and the other-language key; the statistic is Spearman rho over facts with a bootstrap 95% CI over facts, Holm-corrected over four cells (two languages, two layer conditions); the registered rule is rho >= 0.30 with a CI above 0. A partial rho controls for English edit strength, subject token count and subject-piece overlap, and for the edit layer. The layer map edits the first 40 valid facts at layers 2, 6, 10, 18 and 22; its statistic is the mean of the per-layer rho over layers with at least 20 usable facts (bootstrap over facts, layers resampled together; supported if >= 0.20 with a CI above 0 and Holm p < 0.05). The shared-key edit writes the update at the mean of the English, Spanish and Hindi keys (rank 1) or sets the output at each of the three keys (rank 3); its registered improvement rule is a paired median gain in held-out transfer of at least 0.15 with a CI above 0. All of this was registered in `docs/HYPOTHESIS.md` (A21, A24, A26-A28) before the data existed.

### 4. Results

#### 4.1 Overlap against the null

| model | pair | MLP cosine median [95% CI] | null mean | z (pooled) | z (within relation) |
|---|---|---|---|---|---|
| Llama (n=92) | EN-ES | 0.627 [0.557, 0.702] | 0.519 | 4.22 | 3.29 |
| Llama | EN-HI | 0.509 [0.335, 0.594] | 0.429 | 2.45 | 1.79 |
| Llama | ES-HI | 0.423 [0.313, 0.529] | 0.328 | 2.47 | 1.98 |
| Gemma (n=58) | EN-ES | 0.464 [0.329, 0.542] | 0.292 | 4.81 | 3.85 |
| Gemma | EN-HI | 0.226 [0.168, 0.283] | 0.136 | 2.19 | 1.68 |
| Gemma | ES-HI | 0.137 [0.014, 0.267] | 0.134 | 0.08 | -0.45 |

*p-value floor.* With 1000 permutations the smallest possible p is 0.001, so "p = 0.001" means "p <= 0.001"; with 10,000 permutations (Section 6, A14) the EN-ES p is 0.0001 in both models, the new floor.

English-Spanish overlap is the only pair that survives the relation-stratified null and correction in both models. Within relation and after correction, only EN-ES remains significant in both models. Spearman statistics agree in direction (Llama EN-ES z = 5.15; Gemma EN-ES z = 4.55). Per relation with its own null the samples are small (6-38 facts) and mostly not significant (Llama capital_country EN-ES z = 2.75, capital_of 2.08, event_year 0.91). On the 33 facts both models keep, EN-ES is above the null in both (Llama z = 1.88, Gemma z = 4.09) and EN-HI is at the null in both (z = -0.21, 0.55).

#### 4.2 Where the fact is looked up

The critical MLP layer (argmax at the last subject token) has median 3 (EN), 4 (ES), 4 (HI) in Llama (28 layers; 2-11% of facts in layers 8-19) and 6, 6, 6 in Gemma (26 layers; 26-43% in layers 8-19). The median absolute layer difference between languages is 2 in Llama and 2.5-4 in Gemma. The pre-registered expectation of a peak in layers 8-19 is not met for Llama. The median hides structure in Gemma (Figure 4, exploratory): the critical layer has modes at layers 4, 6 and 14; layer 14 holds the argmax for 17 of 58 English and 13 of 58 Spanish facts but for only 2 of 58 Hindi facts, and the cluster appears in three relations (English counts 3, 5, 9 in capital_country, capital_of, event_year). Llama has no such late cluster (3%, 2%, 11% of facts in layers 8-19 for EN, ES, HI).

#### 4.3 Hindi tokenisation

Hindi subjects span a median of 5 (Llama) and 8 (Gemma) tokens against 2 for English. The EN-HI profile cosine correlates negatively with Hindi subject span length (Spearman -0.35 Llama, -0.39 Gemma). Splitting at the median span, Llama's EN-HI cosine is 0.597 for short spans (z = 1.69 pooled) and 0.199 for long spans (z = -1.22). So part of the weak Hindi result is a span-length effect; it does not explain it away, because EN-HI stays below EN-ES with short spans in both models. Hindi accuracy drops mainly for answers of 7 or more tokens (Llama 0.23 against 0.40-0.42; Gemma 0.02 against 0.21).

#### 4.4 Edit transfer over layers

For 10 facts per model (stratified by relation, English prompt only) we fitted a rank-1 edit of the MLP down-projection at every layer (280 edits in Llama, 260 in Gemma), toward a new object of the same relation, and scored greedy generation of the full new answer (Figures 5 and 6). Per-layer rates rest on 10 facts and move in steps of 0.1.

*English.* The edit makes the model produce the new answer for 84% of Llama edits and 74% of Gemma edits; it often fails at the last layers (final three layers: 53% success in Llama, 20% in Gemma).
*Transfer.* Averaged over all layers, the new answer appears in Spanish for 41% (Llama) and 48% (Gemma) of edits and in Hindi for 18% and 20%. Per layer, Hindi success never exceeds 0.5 (Llama layer 4; Gemma layers 12, 13, 17).
*Conditions (A5).* At the fact's English traced layer, English success is 0.9 in both models; median Spanish transfer ratio (dP_ES / dP_EN, over successful edits) is 0.66 in Llama and 0.38 in Gemma, and Hindi 0.0001 and 0.095. At L//2 (14 in Llama, 13 in Gemma) English success is 1.0 and 0.8, Spanish transfer 0.52 and 0.50, Hindi 0.010 and 0.016; specificity loss (maximum absolute probability change over 20 unrelated facts) is 0.084 and 0.089 against 0.030 and 0.027 at the traced layer. Median perplexity on 100 held-out sentences changes by at most 0.2% (Llama) and 0.6% (Gemma) at any layer.
*H7 (mid-layer edits transfer more than layers 0-5, and cost more specificity).* The result depends on the model. In Gemma, near-mid layers (L//2 +/- 3) transfer more than layers 0-5 to Spanish (median paired difference +0.34, 95% CI [0.01, 0.62]) and to Hindi (+0.07 [0.03, 0.13]) and lose more specificity (+0.04 [0.008, 0.19]); 8 of 10 and 9 of 10 facts have a positive Spanish and Hindi difference. In Llama the Spanish difference is negative (-0.31 [-0.47, -0.09]; 1 of 10 facts positive), the Hindi difference is negative or zero (-0.013 [-0.095, 0.000]) and the specificity difference includes zero (+0.004 [-0.023, 0.035]). H7 is therefore supported in Gemma and contradicted in Llama; we make no pooled claim.
#### 4.5 Robustness: phrasings (H8), non-copyable relations (F4), in-context baseline (F3), follow-up edits (F7)

*Phrasings (H8).* For the first 30 valid facts of each model we traced two further phrasings in all three languages. Within a language, the MLP-profile cosine between phrasings of the same fact exceeds the cross-language cosine of the canonical phrasings: Llama 0.754 against 0.609 (median per-fact difference 0.139, 95% CI [0.101, 0.228]; 26 of 30 facts positive), Gemma 0.741 against 0.338 (0.341 [0.287, 0.463]; 30 of 30). H8 is supported in both models. By language the within-language median is 0.985 (EN), 0.744 (ES), 0.695 (HI) in Llama and 0.819, 0.778, 0.713 in Gemma. The profile therefore reflects the fact and language more than the wording, and cross-language overlap is lower than the change of wording alone.

*Non-copyable relations (F4).* We added landmark_country and city_country, whose answers differ in form across languages. Llama keeps 34 extra facts (126 valid in total; 104 without event_year and invention_inventor, so the target of at least 50 is met). On the 104 non-copyable facts the MLP-profile cosine is 0.653 (EN-ES, z = 4.87), 0.529 (EN-HI, z = 2.78) and 0.488 (ES-HI, z = 3.51) against a shuffled-fact null (pooled within this set). With the relation-stratified null on all 126 facts the z-values are 4.24, 2.67 and 1.98. Within single relations in Llama, EN-ES z is 2.75 (capital_country, n = 38), 2.08 (capital_of, 26), 1.96 (city_country, 27) and 0.91 (event_year, 22). Gemma keeps only one extra fact, so its non-copyable set has 29 facts and the target is not reached; there EN-ES is 0.507 (z = 2.53), EN-HI 0.314 (z = 1.16) and ES-HI 0.280 (z = 0.42). The English-Spanish alignment therefore does not depend on copyable answers in Llama; for Gemma the non-copyable sample is too small for a conclusion.

*In-context baseline (F3).* On 10 facts per model we placed the new fact in an English statement before the question (descriptive, no hypothesis). In Gemma the English statement changed the answer in English for 8/10 facts, Spanish 7/10 and Hindi 6/10 (median probability change 0.48, 0.56, 0.27). In Llama it changed Spanish and Hindi answers for 10/10 facts each (0.75, 0.69) but English for 0/10 (0.003); a diagnostic on the first 6 facts (greedy generations saved in `analysis/run_20261025_incontext_diag_llama/generations.txt`) shows why: in English the model writes the original answer and rejects the new one (for example "France, not Denmark"), whereas in Spanish it repeats the new object. The English test prompt repeats the context sentence word for word, which invites a correction; the 0/10 is therefore a behaviour of this prompt design in an instruct model, not a code error, and the English figure should not be read as the model being unable to use the context. In Llama this prompt-level transfer to Spanish and Hindi exceeds that of the weight edit (41% and 18% of edits on average over layers, section 4.4).

*More edit facts (A20).* Twenty further Llama facts were edited under the same three conditions (traced layer, layer 14, layer 21) and pooled with the F7 facts (40 facts; the original 10 sweep facts are not pooled). English success is 0.80 (traced), 0.93 (layer 14) and 0.72 (layer 21). Median Spanish transfer is 0.58 [0.11, 0.88], 0.75 [0.64, 0.91] and 0.00; Hindi transfer is 0.00, 0.10 [0.01, 0.36] and 0.00. Median specificity loss is 0.038, 0.148 and 0.029; the perplexity ratio stays within 0.9996-1.0004. The layer-14 edit therefore transfers more to Spanish and succeeds more often, at about four times the specificity loss; the registered prediction that traced and layer-14 transfer would be within 0.15 was not met. Descriptive, one model, instruct only; the traced-layer CI is wide (transfer is bimodal across facts).

*Follow-up edits (F7).* For 20 further Llama facts (30 with the sweep facts) edited at the fact's traced layer, L//2 and the A9-selected layer (layer 21, labelled "random" in the run files because it is drawn from the swept layers other than the traced and L//2 ones, and it was layer 21 for all 20 facts), English success is 0.80, 0.95 and 0.65. Median Spanish transfer ratio is 0.70 (traced) and 0.78 (L//2), Hindi 0.005 and 0.105. Median specificity loss is 0.062 at the traced layer and 0.568 at L//2 (95% CI [0.08, 0.78]); median perplexity changes by less than 0.05% at the traced layer and L//2. Spanish transfer at layer 21 is near zero (median ratio 6.6e-5). The same follow-up for Gemma (20 facts per condition; 19 at layer 3) gives English success 0.80 (traced), 0.65 (L//2) and 0.89 (layer 3). Median Spanish transfer ratio is 0.65, 0.42 and 0.66; Hindi 0.10, 0.20 and 0.0005. Median specificity loss is 0.084, 0.078 and 0.032; median perplexity changes by less than 0.06% in all three. Unlike Llama layer 21, the Gemma A9-selected layer (layer 3) edit transfers to Spanish. In this follow-up the "random" condition is drawn from the swept layers other than the traced and L//2 ones; for Gemma those are the A9-selected layer 3 and, for some facts, a few other layers, and it was layer 3 for 19 of the 20 facts, so this condition is effectively the A9-selected layer, not an independent random layer. These are descriptive medians on 19-20 facts. H7 is not re-tested on them.

#### 4.6 Controls added after review (A11)

An external review argued that the English-Spanish similarity could reflect generic profile shape or subject-name overlap. We pre-registered controls (A11, before computing) and ran them on the cached traces (Llama 126 facts, Gemma 59; 10,000 permutations; no new model run).

*Same-language baselines.* In Gemma the same-fact EN-ES cosine is 0.424 against 0.242 for different facts of the same relation (EN-EN 0.247, ES-ES 0.234). In Llama it is 0.570 against 0.480 for different-fact EN-ES, but 0.606 for EN-EN and 0.413 for ES-ES: English profiles of different facts are more alike than English and Spanish profiles of the same fact, so in Llama the profile shape is strongly language-specific and the fact-specific increment (0.09) is a small part of the cosine.

*Centred profiles.* After subtracting each language's mean profile, the EN-ES median cosine is 0.284 (null 0.113, z = 3.61) in Llama and 0.410 (null 0.082, z = 5.43) in Gemma. The alignment is not an artefact of the shared mean profile.

*Lexical overlap.* After removing accents and case, 36.5% of Llama subjects and 11.9% of Gemma subjects are identical in English and Spanish. The per-fact EN-ES cosine correlates with the similarity of the two subject strings in Llama (Spearman 0.29, permutation p < 0.001) and weakly in Gemma (0.22, p = 0.097). Split at the median similarity, Llama's EN-ES alignment is z = 4.50 (uncentred) and 3.36 (centred) in the high-similarity half (n = 62) but 2.04 and 1.44 in the low-similarity half (n = 64); the centred value is below the pre-registered threshold of 2. Gemma's low-similarity half gives z = 2.80 and 3.42 (n = 30). On these single-seed profiles, a substantial part of Llama's EN-ES similarity is therefore associated with subject-name overlap, and the remainder is not distinguishable from zero after centring at this sample size; in Gemma it is. The independent-noise re-analysis below revises this for Llama.

*Reconciling this split with the later covariate model (A22).* The split above used the original single-seed profiles, which Section 4.7 shows to be noisier (A13a: mean increment 0.239 against 0.346 with independent noise, Llama). We therefore repeated the split on the independent-noise profiles (registered as A22 before the computation; same facts, no new model run). Splitting the 103 informative Llama facts by whether the English and Spanish subject spans share any subword piece (Jaccard 0 against above 0), the fact-specific increment is present in both halves: 0.303 [0.192, 0.423] (n = 54, no shared piece) and 0.397 [0.291, 0.505] (n = 49, shared piece); leak-free 0.312 [0.202, 0.441] (n = 53). In Gemma the no-shared-piece half has 0.264 [0.139, 0.402] (n = 21) and the shared-piece half has only 4 facts. So in Llama subject-name overlap raises the increment by about 0.09 but is not required for it; the sentence above that the remainder "is not distinguishable from zero" applies to the single-seed profiles and is superseded by this result. The pre-registered prediction that Gemma's halves would be mostly unresolved was not met (the no-shared-piece half is present). Of the four pre-registered predictions, the two about Llama held, the one about Gemma did not, and the leak-free prediction held for Llama but not for the small Gemma halves (one changed from present to unresolved, one has no facts left).

*Entity dependence.* Clustering facts that share an English subject or answer (81 clusters in Llama, 56 in Gemma) leaves the 95% CI of the median EN-ES cosine essentially unchanged (Llama [0.596, 0.688]; Gemma [0.323, 0.536]).

#### 4.7 Is the alignment shared retrieval? Noise re-trace and interchange test (A12-A13)

*Shared-noise check (A13a).* The original tracer used one noise seed per fact for every language, which could inflate EN-ES similarity. We re-traced both models with independent noise per language (10 draws each; the last-token MLP profile, same score). The fact-specific EN-ES increment D (same-fact cosine minus the mean for other facts of the same relation) stays clearly above zero: Llama 0.346 (97.5% CI [0.271, 0.424], n = 103 informative facts; 0.367 [0.293, 0.445] on all 126), Gemma 0.257 ([0.130, 0.381], n = 25; 0.297 [0.214, 0.374] on all 59). With the original single seed the same facts gave 0.239 (Llama) and 0.128 (Gemma), so the single-draw profiles were noisier, not inflated. Within-language noise ceilings (draws 0-4 against 5-9) are 0.82/0.79 (Llama EN/ES) and 0.59/0.54 (Gemma), so most of the remaining gap is language difference, not noise.

*Covariate control (A13b).* Adding subject-token overlap, answer identity, span length and effect size as covariates leaves Llama's increment at 0.319 ([0.210, 0.428], n = 103). Gemma is unresolved at n = 25 informative facts (0.138, [-0.086, 0.362]) and present on the 55 non-identical-prefix facts (0.216, [0.047, 0.385]).

*Interchange (A13).* Profile similarity does not show that the same representation is read out. We therefore swapped the residual stream at the last subject token (site A) or final token (site B), one layer at a time, from a source prompt in one language into a target prompt in the other, and measured the change in log-probability of the source's answer minus the target's answer. The statistic I is the cross-language effect divided by the same-language effect, summed over the layers where the same-language effect is at least half its maximum. Confirmatory pairs have source subjects with no shared token between English and Spanish (Llama 142 pair-directions, 67 source entities; Gemma 162, 71); CIs resample source entities. The same-language gate (flip rate at least 60%, effect at least 2 nats) is met in all eight cells (flip 0.68-0.95, effect 19-27 nats).

At site A the ratio is Llama EN-to-ES 1.17 [1.09, 1.28] and ES-to-EN 0.90 [0.86, 0.94]; Gemma 1.07 [1.03, 1.12] and 0.94 [0.91, 0.97]. All four lower bounds exceed the pre-registered 0.7, so the pre-registered rule gives SHARED. The answer flips in 85% and 67% of Llama cases and 90% and 87% of Gemma cases; swaps from the same fact (self) and from a wrong token (control) flip none. At site B the ratio is also near 1 (0.90-1.11) but the cross-language flip rate is only 34-45%, against 67-94% within a language, so the final-token state is less interchangeable than the subject-token state. The lexical-overlap prediction is not met: the fitted ratio falls from 1.17 to 0.80 (Llama) and 1.03 to 0.91 (Gemma) from zero to full token overlap.

*Caveats.* Four relations (capital of country, capital country, city country, landmark city), English and Spanish only, and the identity check differs from zero by up to 0.119 nats (bf16 batch numerics). The tracing increments above do not predict interchange success per fact: the Spearman correlation is -0.04 in Llama (n = 40, p = 0.83), so the pre-registered metric-validity check fails there; in Gemma it is 0.23 on only 9 facts. Tracing similarity and interchange therefore support the same conclusion for these relations without being shown to measure the same quantity.

*Hindi interchange (A15, underpowered).* We repeated the test with Hindi in place of Spanish. Hindi recall limited the fact set to 30 Llama facts (all capital-of) and 16 Gemma facts (all capital-country), giving 23 and 12 source entities, below the pre-registered floor of 25, so we make no claim. In the pre-registered primary direction (English source into a Hindi prompt) the same-language ceiling is too weak (answer flips in 12% of Llama and 36% of Gemma cases against the required 60%), so the gate fails in both models. In the reverse direction the gate passes (flip 0.80 and 0.82) and the site-A ratio is 0.74 [0.67, 0.81] in Llama and 0.74 [0.69, 0.82] in Gemma, with cross-language flip rates of 0.26 and 0.45; by the rule this is PARTIAL, but we report it as descriptive only. The Hindi question therefore stays open.

*Hindi interchange, re-test (A16).* Part of the A15 failure was format: the Hindi templates made Llama continue with a question mark and a new line, which the strict answer match counted as wrong. We added question-and-answer templates, dropped the per-relation threshold, tested only Hindi-to-English (the direction whose ceiling passed) and used four source facts per target. Llama kept 66 facts (three relations) and Gemma 55 (four), giving 66 and 54 source entities, above the floor of 25. At site A the ratio is 0.74 [0.69, 0.78] in Llama (answer flips 0.48 against 0.80 within Hindi; PARTIAL, since the lower bound is below 0.7) and 0.76 [0.71, 0.81] in Gemma (flips 0.64 against 0.90; SHARED, by a margin of 0.01). On facts that were not in the A15 set the ratios are 0.76 and 0.77. Per relation the ratio ranges from 0.53 (Gemma, landmark city) to 0.87 (Gemma, city country). At site B the cross-language flip rate is 0.00, so the ratio is not interpretable. Hindi-to-English retrieval is therefore largely but not fully shared, weaker than English-Spanish (0.90-0.94), and the two models fall on either side of the 0.7 threshold with the same point estimate; English-to-Hindi was not tested here because patching one token cannot carry a 5-8 token Hindi entity; the span-wise test below does.

*Hindi interchange, span-wise (A18).* We patched the whole subject span: the last k subject positions of the target prompt (k = the shorter of the two subject lengths) receive the source's last k subject-position states, end-aligned, in both directions on the A16 fact sets (200 pairs per model and direction; 190 confirmatory pairs and 60 source entities in Llama, 186 and 53 in Gemma). The same-language ceiling now passes in all four cells (answer flips 0.71 EN to HI and 0.94 HI to EN in Llama; 0.81 and 0.95 in Gemma). The site-A ratio is 0.86 [0.80, 0.93] (EN to HI) and 0.87 [0.82, 0.92] (HI to EN) in Llama, and 0.93 [0.88, 0.99] and 0.81 [0.76, 0.86] in Gemma; all four are SHARED by the pre-registered rule and the two directions differ by 0.01 and 0.13, within the registered symmetry margin of 0.15. Excluding facts with the answer inside the subject changes the ratios by at most 0.05 (largest: Gemma HI to EN 0.81 to 0.85, EN to HI 0.93 to 0.95). The label is the same under all four bootstrap clusterings (source, target, connected component, two-way) in all eight cells (plain and leak-free); the smallest lower bound is 0.703 (Gemma HI to EN, component clustering), so that cell is close to the threshold. Patching the span moves the Hindi-to-English estimate up from 0.74-0.76 (last token only), so the earlier number understated the shared fraction. *Base-model replication (A19).* To test whether the result depends on instruction tuning we repeated the English-Spanish and English-Hindi interchange on Llama-3.2-3B base (same architecture and tokenizer; 81 English-Spanish facts and 76 Hindi facts kept, 200-pair cap, 79 and 190 confirmatory pairs, 33 and 69 source entities). The same-language ceiling passes in all four directions (flip 0.66-0.91). Site-A ratios: English to Spanish 1.21 [1.11, 1.37], Spanish to English 0.87 [0.81, 0.92], Hindi to English 0.86 [0.82, 0.90] (all SHARED) and English to Hindi 0.72 [0.68, 0.77] (PARTIAL: lower bound below 0.7). They differ from the instruct model's by 0.04, 0.03, 0.01 and 0.14, all within the registered margin of 0.15, so all four cells replicate; English to Hindi is the weakest and the one nearest the margin. Labels are the same under all four bootstrap clusterings and with answer-leaking facts removed (changes <= 0.01). The Spanish estimates rest on fewer facts than the instruct run (the 100-fact prediction was missed).

The broadcast variant (last-token state copied over every position) fails the ceiling in Llama (flip 0.41) and is not interpreted. These ratios are estimates from 3-4 relations and two models, not a general rate.

#### 4.8 Which facts does an English edit carry over to? Wording control, key similarity, layer map and shared-key edit (A21, A24, A28; Kaggle run)

All numbers in this section come from one Kaggle run of Llama-3.2-3B-Instruct (Section 3, last paragraph; 86 facts edited, 348 minutes, 20 of 20 jobs finished). They are not compared per fact with the laptop edits of Section 4.5: re-fitting the edits of an earlier Kaggle run against the stored laptop edits missed the registered limit (median absolute difference per edit 0.10 in English and 0.11 in Spanish against a limit of 0.02, 60 fact-layer pairs), although the group medians did not shift. Possible causes (fp16 on a T4 against bf16, the 25-step fit) were not separated. Median-level statements are comparable, per-fact statements are not.

*Wording control (A21 rule).* An English edit fitted on one phrasing may fail on a new Spanish or Hindi phrasing simply because it does not generalise to new wording. We therefore compare transfer to a language with transfer to unseen English wording (Table 5). At the traced layer the edit reaches unseen English wordings with a median ratio of 0.94 and the Spanish main prompt with 0.59, the Hindi one with 0.002; at layer 14 the values are 1.00, 0.70 and 0.12. The difference D (English wording minus language) is 0.39 [0.29, 0.49] for Spanish and 0.79 [0.67, 0.89] for Hindi at the traced layer, and 0.26 [0.15, 0.36] and 0.73 [0.64, 0.80] at layer 14. By the registered rule (CI above 0.15) the loss is beyond wording in all four cells; the lower bound for Spanish at layer 14 is 0.153, so that cell is only just above the line. Spanish transfer to held-out Spanish phrasings is lower again (median 0.42 at the traced layer, 0.62 at layer 14).

| language | edit layer | n | English reworded | language, main prompt | D [95% CI] |
|---|---|---|---|---|---|
| Spanish | traced | 80 | 0.94 | 0.59 | 0.39 [0.29, 0.49] |
| Spanish | 14 | 83 | 1.00 | 0.70 | 0.26 [0.15, 0.36] |
| Hindi | traced | 80 | 0.94 | 0.002 | 0.79 [0.67, 0.89] |
| Hindi | 14 | 83 | 1.00 | 0.12 | 0.73 [0.64, 0.80] |

*Key similarity (A26 test, A28 scale).* The median cosine between the English key and the Spanish key is 0.61 at the traced layer and 0.68 at layer 14; for Hindi it is 0.27 and 0.46. Across facts, keys that are closer transfer better at the traced layer (Table 6, Figure 7): Spearman rho 0.41 [0.20, 0.59] for Spanish (n = 80) and 0.42 [0.19, 0.62] for Hindi, Holm p 0.0012 and 0.0008, so both cells meet the registered rule. At layer 14 there is no association (0.06 [-0.17, 0.28]; 0.02 [-0.20, 0.24]). The traced layer differs by fact (layers 0, 1, 2, 3, 4, 5, 6 and 16 for 13, 1, 24, 6, 17, 12, 5 and 2 of the 80 facts), and both key similarity and transfer can depend on the layer, so the registered check adds the edit layer to the controls: the partial rho is 0.26 [0.00, 0.49] for Spanish and 0.22 [0.01, 0.44] for Hindi. Both are at or above the registered 0.20 with a CI above 0, by lower bounds of 0.004 and 0.011, so the result holds narrowly. A first pass on 28 facts of an earlier run (registered as a test, not as a claim) gave 0.55 and 0.52 at the traced layer, and with the edit layer controlled the Hindi value was 0.06, which is why A28 was registered; the larger run did not reproduce that drop for Hindi, and the first-pass values are not used as results.

| language | edit layer | n | median key cosine | rho [95% CI] | Holm p | partial rho with edit layer [95% CI] |
|---|---|---|---|---|---|---|
| Spanish | traced | 80 | 0.61 | 0.41 [0.20, 0.59] | 0.0012 | 0.26 [0.00, 0.49] |
| Spanish | 14 | 83 | 0.68 | 0.06 [-0.17, 0.28] | 0.60 | - |
| Hindi | traced | 80 | 0.27 | 0.42 [0.19, 0.62] | 0.0008 | 0.22 [0.01, 0.44] |
| Hindi | 14 | 83 | 0.46 | 0.02 [-0.20, 0.24] | 0.60 | - |

*Layer map.* To separate the layer from the fact, we edited the first 40 valid facts at layers 2, 6, 10, 18 and 22 (36-38 usable facts per layer; Table 7, Figure 8). Median Spanish transfer to held-out phrasings is 0.39, 0.42, 0.32, 0.17 and 0.007 at these layers (0.62 at layer 14 in the main run), so an English edit reaches Spanish from early and middle layers and not from the last ones; Hindi transfer is at most 0.008 at all five (0.04 at layer 14). Within a layer, the association between key cosine and transfer is supported for Spanish (mean of the five per-layer rho 0.29 [0.12, 0.44], Holm p 0.0024) and not for Hindi (0.17 [-0.02, 0.36], Holm p 0.047; below 0.20 with a CI that includes 0). Per layer the Spanish rho is 0.59, 0.29, -0.13, 0.21 and 0.46 and the Hindi rho 0.43, 0.70, -0.02, -0.04 and -0.22, so the within-layer pattern is not uniform. Where median transfer is close to 0 (Hindi at every layer, Spanish at layer 22) the rho ranks very small values and is weak evidence.

| edit layer | n | Spanish: median transfer | Spanish: rho | Hindi: median transfer | Hindi: rho |
|---|---|---|---|---|---|
| 2 | 36 | 0.387 | 0.59 | 0.008 | 0.43 |
| 6 | 36 | 0.419 | 0.29 | 0.004 | 0.70 |
| 10 | 38 | 0.319 | -0.13 | 0.005 | -0.02 |
| 18 | 38 | 0.169 | 0.21 | 0.004 | -0.04 |
| 22 | 37 | 0.007 | 0.46 | 0.0001 | -0.22 |
| pooled | 40 facts | | 0.29 [0.12, 0.44] | | 0.17 [-0.02, 0.36] |

*Shared-key edit (A24 rule).* If the English key were the problem, writing the update at a key shared by the three languages should raise transfer (Table 8). It does not. The median paired gain in held-out transfer over the English-key edit is at most 0.044 in any of the eight cells (two modes, two layer conditions, two languages), against the registered 0.15; the largest is the rank-1 mean-key edit for Spanish at the traced layer (0.044 [0.008, 0.076]), and at layer 14 all gains are within 0.007 of zero. Group medians look larger (for example Spanish at the traced layer 0.49 for the English-key edit against 0.66 for the mean-key edit on the 73 facts used) because the gains sit in a few facts; the registered quantity is the paired median. Specificity loss rises by a factor of 1.05 to 1.38 and English success changes by at most 0.04 in either direction; perplexity ratios stay at or below 1.001. The edit therefore does not fix the gap, and the training-free fix suggested by the key difference is not supported. Writing the update at the shared key also requires the translated subject, which an English-only edit does not.

| mode | edit layer | language | n | English-key median | paired gain [95% CI] |
|---|---|---|---|---|---|
| mean key (rank 1) | traced | Spanish | 73 | 0.490 | 0.044 [0.008, 0.076] |
| mean key (rank 1) | traced | Hindi | 73 | 0.003 | 0.019 [0.001, 0.044] |
| mean key (rank 1) | 14 | Spanish | 80 | 0.656 | 0.005 [-0.015, 0.025] |
| mean key (rank 1) | 14 | Hindi | 80 | 0.045 | 0.007 [-0.001, 0.028] |
| joint keys (rank 3) | traced | Spanish | 78 | 0.446 | 0.024 [0.008, 0.050] |
| joint keys (rank 3) | traced | Hindi | 78 | 0.003 | 0.006 [0.001, 0.032] |
| joint keys (rank 3) | 14 | Spanish | 83 | 0.621 | 0.004 [-0.009, 0.020] |
| joint keys (rank 3) | 14 | Hindi | 83 | 0.038 | 0.005 [-0.000, 0.026] |

*Not testable.* The non-geography relations (A23) remain untestable: the template filter keeps too few relations and source entities to meet the registered floor, so we make no claim about relations outside geography and events.

*What this section supports.* Per fact, the closeness of the English and the Spanish or Hindi key at the edit layer is associated with how well an English edit carries over, for Spanish both across layers (traced-layer partial rho 0.26) and within layers (pooled 0.29), for Hindi across layers narrowly (0.22) and not within a layer. The association is a correlation computed on facts where the English edit worked: the intervention that follows from it, writing the edit at a shared key, gave no gain, so key similarity may be a marker of how language-specific a fact's subject representation is rather than the reason the edit fails. Spanish and Hindi transfer remain below transfer to unseen English wording in all four cells. One model, instruct only, mostly capital and event relations, 36-83 facts per cell; the same analysis on Gemma-2-2B is registered (A29) and has not been run.

### 5. Discussion

*What the evidence supports.* In both models the English and Spanish causal profiles of the same fact are aligned beyond a shuffled-fact null, and the alignment survives a relation-stratified null, rewording of the question (H8), and, in Llama, restriction to answers that are not copyable across languages. This is consistent with partly shared lookup for these two languages, and the lexical control (Section 4.6) shows that in Llama the alignment is stronger for facts whose English and Spanish subject names share subword pieces (0.40 against 0.30) but still present for those that do not. Hindi is weaker: its overlap with English is above the pooled null in Llama (z = 2.90 on 126 facts) but not significant after the stratified null with Bonferroni correction (z = 2.67, p = 0.012), and depends on how many tokens the Hindi subject takes.

*Interchange (Section 4.7).* Swapping the subject-token state between English and Spanish prompts moves the answer nearly as much as a same-language swap (ratio 0.90-1.17 in both models; pre-registered verdict SHARED for English-Spanish on four relations), which supports shared retrieval of the fact at the subject token beyond profile similarity. Patching the whole Hindi subject span gives ratios of 0.81-0.93 in both directions and both instruct models, and the Llama base model reproduces all four cells within 0.15 (English to Hindi, 0.72, is the weakest and PARTIAL), so the result does not depend on instruction tuning for Llama. Two readings agree with earlier work that script matters (Ifergan et al., 2024; Beniwal et al., 2024): the single-token Hindi estimate was lower, and Hindi edit transfer is near zero. The per-fact profile increments do not predict per-fact interchange success in Llama, so the two measurements agree in direction, not in detail.

*What it does not support.* Profile similarity at one token and one component is not identity of mechanism, and we found no evidence of a single language-agnostic space. Edits made in English reach Spanish for about half of the edits and Hindi rarely; transfer depends on the layer and on the model (H7 supported in Gemma, contradicted for Spanish in Llama). The in-context baseline shows that the model can use an English fact in Spanish and Hindi when it is in the prompt (Llama), so the low Hindi transfer of weight edits is not simply a lack of cross-lingual access; the edit site and the way Hindi is tokenised are alternative explanations that we have not separated.

*Shared at the subject token, yet edits carry over only partly.* The interchange ratio for Spanish and Hindi is near 1, but the weight edit carries over to Spanish only in about half of the cases and to Hindi rarely. Taken together, these results say that a shared subject representation is not sufficient for a rank-1 edit of one language to reach another; where the edit is written matters (layer 14 against the traced layer; Hase et al., 2023). The paraphrase control (A21) now exists (Section 4.8): on the Kaggle run, transfer to Spanish and Hindi is lower than transfer to unseen English wording in all four cells (difference 0.26-0.39 for Spanish, 0.73-0.79 for Hindi), so the shortfall is not only a lack of generalisation to new wording, with the Spanish layer-14 cell only just above the registered line.

*Which facts carry over.* Section 4.8 adds a per-fact result: the closer the English key and the Spanish or Hindi key of a fact at the edit layer, the better the English edit carries over (rho 0.41 and 0.42 at the traced layer; 0.26 and 0.22 with the edit layer controlled; none at layer 14). Within fixed layers the association holds for Spanish and not for Hindi. We read this as an association, not a mechanism, for two reasons: the intervention that follows from it (a shared-key edit) did not help (paired gain at most 0.044), and Hindi transfer is near zero at every layer, so Hindi ranks are computed among tiny values. A shared subject representation (Section 4.7) with language-specific keys is therefore one description that fits the Spanish data; the Hindi data cannot separate it from an edit-layer effect with the sample we have.

*Cost of mid-layer edits.* Pooled over 40 Llama facts (Section 4.5, A20), edits at layer 14 have higher median Spanish transfer than edits at the traced layer (0.75 [0.64, 0.91] against 0.58 [0.11, 0.88]; the intervals overlap) and succeed more often in English (0.93 against 0.80), but have a higher median specificity loss (0.148 against 0.038); any transfer gain is therefore not free.

*Scope.* Two model families (plus one base-model replicate; Section 4.8 is Llama only), hand-written facts, interchange on four relations (English-Spanish) and 3-4 relations (English-Hindi, both directions, span-wise), 10 to 40 edit facts per condition and a small Gemma non-copyable sample limit generality. The BLOOM-1b7 and Llama-3.2-1B pilots (A10), which test whether the weak Hindi result is a tokenisation effect, had not run when this draft was written.

### 6. Deviations from the pre-registration

H6 (EN-ES z > 2 and EN-ES > EN-HI) holds in both models. H8 (phrasing robustness) is supported in both models. H9 (EN-ES z > 2 in at least two new or extended relations) is not supported: only event_year was testable among the pre-registered relations (z = 0.91); the later city_country relation reaches z = 1.96, just below the threshold. The F4 target of 50 non-copyable facts is met for Llama (104) and not for Gemma (29). The F7 follow-up (30 edit facts) finished for Llama only (20 new facts); the Gemma run was interrupted and its partial output is kept unused. The expected layer 8-19 peak is not met for Llama. Amendments A1-A10 (docs/HYPOTHESIS.md; A10, the BLOOM-1b7 pilot, is pre-registered but not yet run): critical-site definition; competitor from the corrupted run (v1 only); statistics definition; edit design; v2 design; matching rule; template selection after a poor first filter; Gemma pool extension; analysis rules (stratified null as primary for new data, non-copyable set, in-context baseline). Later amendments A11-A22 are in `docs/HYPOTHESIS.md`, each registered before its data or computation; A22 (the split by name overlap on independent-noise profiles) was run on stored data after A11 had shown the first split, which is disclosed in Section 4.6. The stratified null and the Hindi span split were added after seeing results and are labelled post hoc. v1 first-token results (66 facts, Llama only) are kept in `paper/paper.md` as a pilot; in v1 the Hindi pairs were at the null, which the v2 analysis shows to be sample- and metric-dependent.

*Permutation count (A14).* The pre-registered overlap tests use 1000 permutations, so the smallest possible p is 0.001. We reran the same traces with the same seed and 10,000 permutations (CPU, no new model run). Observed medians are identical. For the MLP-profile cosine, z changes by at most 0.21 on the main-table sets (EN-ES: Llama 4.22 to 4.01, Gemma 4.81 to 4.82; p = 0.0001 in both) and no verdict changes; the stratified Gemma EN-HI p is 0.0502 (0.050 before), still not significant. The top-5 IoU z is unstable across draw counts (Llama 14.9 to 10.9 on the 126-fact set), so we do not rely on it. Two secondary tests at p of about 0.05 cross 0.05 (Gemma no-copyable EN-ES Spearman 0.051 to 0.043; Gemma event_year EN-HI 0.051 to 0.050). The tables in this paper keep the 1000-draw values; the 10,000-draw values are in `results/manifest.json` (keys `pk_*`).

*Kaggle run (A21, A24, A26-A28).* The first pass of the key-similarity test (A26, 28 facts per cell, earlier Kaggle run) gave rho 0.55 (Spanish) and 0.52 (Hindi) at the traced layer; its predictions that rho is positive in all four cells and that Hindi is weaker than Spanish were not met (Spanish at layer 14 was negative; Hindi was not weaker). A28 repeated it on 80-83 facts with the same rules and gave 0.41 and 0.42; the first pass is reported only as the reason for A28. Of the eight sub-predictions of A28, seven were met; the miss is the pooled within-layer support for Hindi (0.17 against the line of 0.20). The shared-key edit (A24) met three of its six registered predictions clearly and failed two; its decision rule gave no improvement in any of the eight cells. The A21 sanity check (re-fitted edits against the stored laptop edits) failed its registered limit (Section 4.8) and is reported as such. The shared-key gain in the A28 verdict text of `docs/HYPOTHESIS.md` (largest 0.047) was read from the main-prompt column; the registered held-out column gives 0.044 and the verdict is the same.

### 7. Limitations

Two models, one of them a base and one an instruct model; hand-written facts; small per-relation samples; event_year answers are digit strings that can be copied across languages and make up half of Gemma's set; Hindi is hard to trace because subjects span many tokens; invention_inventor keeps 0 facts in both models. Layer-0 edit results were not reproducible between runs (v1). The EN/ES/HI facts were written by the authors' automated assistant and have not been checked by a native speaker; this is the main validity limitation of the data. Five relations, mostly geography; three languages; models of 2-3B parameters; no 7B-class model. The edit comparison of Section 4.5 is descriptive (30-40 Llama facts, one model, instruct only). Section 4.8 adds the paraphrase control and a per-fact key-similarity test, but it is one model (instruct only), mostly capital and event relations, 36-83 facts per cell, edits fitted on a different machine and number format (Kaggle T4, fp16) from Sections 4.1-4.7 and not reproducible per edit against them, the key-similarity result is a correlation among facts where the English edit worked, the Hindi layer-controlled result passes its line by a lower bound of 0.011, and Hindi transfer is near zero at every layer. Relations outside geography and events are untestable with our template filter (A23). The replication on Gemma-2-2B (A29) is registered and has not been run. Multiple testing: with Bonferroni over the six primary overlap tests (alpha 0.0083) Gemma's EN-HI (p = 0.016) is not significant. The tracing increments do not predict interchange success per fact in Llama (Spearman -0.04), so the two measurements are not shown to measure the same quantity. There is no external (OSF/Zenodo) pre-registration timestamp; the order of pre-registration and data is evidenced by git history only.

### 8. Reproducibility

Code, configs, write-once run directories, seeds, logs and a claim manifest (`results/manifest.json`, `./reproduce.sh verify`) are in the repository. End-to-end rerun status: see `docs/LOG.md`. Section 4.8 is produced by one self-contained Kaggle notebook (code id bundle-f8371a452538), archived with its raw output in the folder `analysis/kaggle_output_20261031`; the four analysis files are the folders named `run_20261031_*_A28` under `analysis`.

### References

- Meng, K., Bau, D., Andonian, A., Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. arXiv:2202.05262.
- Grattafiori, A. et al. (2024). The Llama 3 Herd of Models. arXiv:2407.21783.
- Merity, S., Xiong, C., Bradbury, J., Socher, R. (2016). Pointer Sentinel Mixture Models. arXiv:1609.07843.
- Geva et al. (2023). Dissecting Recall of Factual Associations in Auto-Regressive Language Models. arXiv:2304.14767.
- Meng et al. (2022). Mass-Editing Memory in a Transformer. arXiv:2210.07229.
- Hase et al. (2023). Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models. arXiv:2301.04213.
- Gupta et al. (2024). Rebuilding ROME: Resolving Model Collapse during Sequential Model Editing. arXiv:2403.07175.
- Yang et al. (2024). Understanding the Collapse of LLMs in Model Editing. arXiv:2406.11263.
- Wendler et al. (2024). Do Llamas Work in English? On the Latent Language of Multilingual Transformers. arXiv:2402.10588.
- Fierro et al. (2024). How Do Multilingual Language Models Remember Facts? arXiv:2410.14387.
- Zhang et al. (2024). Multilingual Knowledge Editing with Language-Agnostic Factual Neurons. arXiv:2406.16416.
- Wang et al. (2025). Lost in Multilinguality: Dissecting Cross-lingual Factual Inconsistency in Transformer Language Models. arXiv:2504.04264.
- Lu et al. (2025). Paths Not Taken: Understanding and Mending the Multilingual Factual Recall Pipeline. arXiv:2505.20546.
- Schut et al. (2025). Do Multilingual LLMs Think In English? arXiv:2502.15603.
- Jiang et al. (2020). X-FACTR. arXiv:2010.06189.
- Kassner et al. (2021). Multilingual LAMA. arXiv:2102.00894.
- Wang et al. (2023). Cross-Lingual Knowledge Editing in Large Language Models. arXiv:2309.08952.
- Khandelwal et al. (2024). Cross-Lingual Multi-Hop Knowledge Editing. arXiv:2407.10275.
- Nie et al. (2024). BMIKE-53: Investigating Cross-Lingual Knowledge Editing with In-Context Learning. arXiv:2406.17764.
- Qi et al. (2023). Cross-Lingual Consistency of Factual Knowledge in Multilingual Language Models. arXiv:2310.10378.
- Banerjee et al. (2024). Breaking Boundaries: Investigating the Effects of Model Editing on Cross-linguistic Performance. arXiv:2406.11139.
- Beniwal et al. (2024). Cross-lingual Editing in Multilingual Language Models. arXiv:2401.10521.
- Durrani et al. (2025). Editing Across Languages: A Survey of Multilingual Knowledge Editing. arXiv:2505.14393.
- Mousi, B., Durrani, N., Dalvi, F. (2025). An Exploration of Knowledge Editing for Arabic. arXiv:2507.09629.
- Ifergan, M., Choshen, L., Aharoni, R., Szpektor, I., Abend, O. (2024). Beneath the Surface of Consistency: Exploring Cross-lingual Knowledge Representation Sharing in LLMs. arXiv:2408.10646.
- Dumas et al. (2024). Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers. arXiv:2411.08745.
- Zhang, F., Nanda, N. (2023). Towards Best Practices of Activation Patching in Language Models: Metrics and Methods. arXiv:2309.16042.
- Heimersheim, S., Nanda, N. (2024). How to use and interpret activation patching. arXiv:2404.15255.
- Hu, Y., Zhou, W., Xiao, J. (2026). CLM-Bench: Benchmarking and Analyzing Cross-lingual Misalignment of LLMs in Knowledge Editing. arXiv:2601.17397.

## Figures of the paper

The figure files are in `paper/figures/` (PNG and PDF). Captions are the ones used in the LaTeX build (`paper/build_tex.py`).

![Figure 1. Median normalised MLP indirect effect per layer at each language's last subject token (band: interquartile range), for Llama-3.2-3B (left) and Gemma-2-2B (right).](paper/figures/fig1_profiles.png)

![Figure 2. MLP-profile cosine between languages: observed median with 95% bootstrap CI (dots) and mean of the shuffled-fact permutation null (bars).](paper/figures/fig2_overlap.png)

![Figure 3. Canonical-phrasing accuracy by answer length in tokens, per language and model.](paper/figures/fig3_hindi.png)

![Figure 4. Distribution of the critical MLP layer (argmax of the profile) per language. Exploratory: Gemma shows modes at layers 4, 6 and 14.](paper/figures/fig4_layer_hist.png)

![Figure 5. Llama-3.2-3B edit sweep (10 facts, every layer): English success, transfer ratio, specificity and perplexity ratio by layer.](paper/figures/fig5_sweep_llama.png)

![Figure 6. Gemma-2-2B edit sweep, same layout as Figure 5.](paper/figures/fig6_sweep_gemma2.png)

![Figure 7. Spearman rho between key cosine (English key against the Spanish or Hindi key) and held-out transfer of the English edit, at the fact's traced layer and at layer 14 (Llama-3.2-3B-Instruct, Kaggle run; points with 95% bootstrap CI over facts; squares: partial rho with the edit layer among the controls; dashed line: registered 0.30).](paper/figures/fig7_key_similarity.png)

![Figure 8. Layer map (40 facts, five fixed edit layers): median held-out transfer (top, with 95% CI) and within-layer rho between key cosine and transfer (bottom), Spanish (left) and Hindi (right).](paper/figures/fig8_layer_map.png)

## Appendix: full overlap tables

*Copied word for word from `paper/tables/overlap_full.md` when this record was built. To change it, edit that file and rebuild.*

### Overlap tables (generated by code/make_tables.py from the run artifacts)

#### Llama-3.2-3B: pooled shuffled-fact null (n=92)

| statistic | EN-ES | EN-HI | ES-HI |
|---|---|---|---|
| MLP cosine | 0.627 [0.557, 0.702] (null 0.519, z=4.22, p=0.001) | 0.509 [0.335, 0.594] (null 0.429, z=2.45, p=0.003) | 0.423 [0.313, 0.529] (null 0.328, z=2.47, p=0.006) |
| MLP Spearman | 0.461 [0.428, 0.531] (null 0.334, z=5.15, p=0.001) | 0.323 [0.233, 0.407] (null 0.248, z=2.61, p=0.007) | 0.283 [0.213, 0.317] (null 0.191, z=3.08, p=0.001) |
| residual cosine | 0.984 [0.976, 0.988] (null 0.957, z=5.62, p=0.001) | 0.903 [0.860, 0.926] (null 0.891, z=1.47, p=0.062) | 0.921 [0.900, 0.942] (null 0.882, z=3.57, p=0.001) |
| residual Spearman | 0.888 [0.852, 0.901] (null 0.820, z=4.95, p=0.001) | 0.784 [0.752, 0.815] (null 0.762, z=1.53, p=0.062) | 0.793 [0.762, 0.834] (null 0.763, z=1.73, p=0.039) |
| IoU top-5 | 0.250 [0.111, 0.250] (null 0.117, z=5.25, p=0.031) | 0.111 [0.111, 0.250] (null 0.167, z=-0.87, p=1.000) | 0.250 [0.181, 0.250] (null 0.226, z=0.49, p=0.792) |
| IoU top-10 | 0.429 [0.333, 0.538] (null 0.338, z=3.26, p=0.059) | 0.381 [0.333, 0.538] (null 0.375, z=0.13, p=0.508) | 0.538 [0.429, 0.667] (null 0.518, z=0.52, p=0.771) |
| IoU top-20 | 0.600 [0.538, 0.667] (null 0.450, z=5.10, p=0.001) | 0.481 [0.429, 0.538] (null 0.455, z=0.95, p=0.444) | 0.600 [0.538, 0.633] (null 0.526, z=3.01, p=0.013) |

#### Llama-3.2-3B: relation-stratified null (post hoc for these two models) (n=92)

| statistic | EN-ES | EN-HI | ES-HI |
|---|---|---|---|
| MLP cosine | 0.627 [0.561, 0.702] (null 0.547, z=3.29, p=0.001) | 0.509 [0.335, 0.597] (null 0.452, z=1.79, p=0.050) | 0.423 [0.325, 0.529] (null 0.345, z=1.98, p=0.020) |
| MLP Spearman | 0.461 [0.428, 0.531] (null 0.363, z=3.82, p=0.001) | 0.323 [0.233, 0.407] (null 0.269, z=2.22, p=0.018) | 0.283 [0.211, 0.317] (null 0.210, z=2.38, p=0.012) |
| residual cosine | 0.984 [0.976, 0.987] (null 0.972, z=4.12, p=0.001) | 0.903 [0.862, 0.926] (null 0.898, z=0.76, p=0.219) | 0.921 [0.900, 0.942] (null 0.903, z=2.22, p=0.015) |
| residual Spearman | 0.888 [0.852, 0.902] (null 0.849, z=3.46, p=0.001) | 0.784 [0.748, 0.815] (null 0.768, z=1.48, p=0.067) | 0.793 [0.762, 0.834] (null 0.779, z=1.15, p=0.125) |
| IoU top-5 | 0.250 [0.111, 0.250] (null 0.157, z=1.49, p=0.293) | 0.111 [0.111, 0.250] (null 0.177, z=-1.00, p=1.000) | 0.250 [0.181, 0.252] (null 0.229, z=0.45, p=0.818) |
| IoU top-10 | 0.429 [0.333, 0.538] (null 0.410, z=0.50, p=0.759) | 0.381 [0.333, 0.538] (null 0.399, z=-0.44, p=0.759) | 0.538 [0.429, 0.667] (null 0.536, z=0.18, p=0.963) |
| IoU top-20 | 0.600 [0.538, 0.667] (null 0.529, z=2.37, p=0.055) | 0.481 [0.429, 0.538] (null 0.471, z=0.47, p=0.729) | 0.600 [0.538, 0.633] (null 0.549, z=2.20, p=0.154) |

#### Llama-3.2-3B: pooled null, without event_year (n=70)

| statistic | EN-ES | EN-HI | ES-HI |
|---|---|---|---|
| MLP cosine | 0.649 [0.551, 0.733] (null 0.537, z=3.85, p=0.001) | 0.553 [0.434, 0.639] (null 0.481, z=2.13, p=0.015) | 0.529 [0.364, 0.609] (null 0.379, z=3.51, p=0.001) |
| MLP Spearman | 0.450 [0.386, 0.532] (null 0.329, z=4.20, p=0.001) | 0.354 [0.259, 0.504] (null 0.264, z=2.96, p=0.002) | 0.294 [0.226, 0.351] (null 0.201, z=2.91, p=0.005) |
| residual cosine | 0.986 [0.981, 0.989] (null 0.974, z=3.57, p=0.001) | 0.927 [0.902, 0.947] (null 0.921, z=1.04, p=0.144) | 0.932 [0.906, 0.959] (null 0.914, z=1.91, p=0.017) |
| residual Spearman | 0.885 [0.843, 0.901] (null 0.829, z=3.65, p=0.001) | 0.808 [0.774, 0.841] (null 0.794, z=1.11, p=0.135) | 0.798 [0.762, 0.852] (null 0.783, z=0.97, p=0.163) |
| IoU top-5 | 0.111 [0.111, 0.250] (null 0.112, z=-0.09, p=0.996) | 0.111 [0.111, 0.250] (null 0.138, z=-0.54, p=1.000) | 0.250 [0.111, 0.250] (null 0.165, z=1.35, p=0.325) |
| IoU top-10 | 0.333 [0.250, 0.429] (null 0.276, z=1.47, p=0.277) | 0.333 [0.250, 0.381] (null 0.325, z=0.29, p=0.826) | 0.484 [0.333, 0.538] (null 0.448, z=0.83, p=0.250) |
| IoU top-20 | 0.481 [0.455, 0.600] (null 0.424, z=2.09, p=0.084) | 0.429 [0.379, 0.538] (null 0.412, z=0.55, p=0.552) | 0.600 [0.538, 0.600] (null 0.506, z=3.42, p=0.004) |

#### Gemma-2-2B: pooled shuffled-fact null (n=58)

| statistic | EN-ES | EN-HI | ES-HI |
|---|---|---|---|
| MLP cosine | 0.464 [0.329, 0.542] (null 0.292, z=4.81, p=0.001) | 0.226 [0.168, 0.283] (null 0.136, z=2.19, p=0.010) | 0.137 [0.014, 0.267] (null 0.134, z=0.08, p=0.494) |
| MLP Spearman | 0.332 [0.190, 0.465] (null 0.162, z=4.55, p=0.001) | 0.134 [0.029, 0.199] (null 0.082, z=1.35, p=0.087) | 0.054 [-0.024, 0.119] (null 0.072, z=-0.48, p=0.699) |
| residual cosine | 0.974 [0.961, 0.985] (null 0.925, z=5.18, p=0.001) | 0.872 [0.848, 0.899] (null 0.872, z=0.06, p=0.506) | 0.894 [0.862, 0.914] (null 0.868, z=2.09, p=0.013) |
| residual Spearman | 0.930 [0.883, 0.951] (null 0.813, z=5.18, p=0.001) | 0.706 [0.594, 0.788] (null 0.731, z=-0.92, p=0.822) | 0.754 [0.685, 0.791] (null 0.738, z=0.70, p=0.245) |
| IoU top-5 | 0.429 [0.250, 0.429] (null 0.223, z=3.72, p=0.006) | 0.250 [0.179, 0.339] (null 0.261, z=-0.27, p=0.994) | 0.250 [0.250, 0.429] (null 0.275, z=-0.44, p=1.000) |
| IoU top-10 | 0.667 [0.538, 0.742] (null 0.505, z=3.03, p=0.011) | 0.484 [0.333, 0.538] (null 0.528, z=-1.23, p=0.929) | 0.538 [0.429, 0.667] (null 0.545, z=-0.23, p=0.986) |
| IoU top-20 | 0.779 [0.739, 0.818] (null 0.643, z=3.39, p=0.001) | 0.538 [0.481, 0.600] (null 0.538, z=0.04, p=0.901) | 0.600 [0.538, 0.667] (null 0.564, z=1.26, p=0.348) |

#### Gemma-2-2B: relation-stratified null (post hoc for these two models) (n=58)

| statistic | EN-ES | EN-HI | ES-HI |
|---|---|---|---|
| MLP cosine | 0.464 [0.329, 0.541] (null 0.330, z=3.85, p=0.001) | 0.226 [0.168, 0.283] (null 0.165, z=1.68, p=0.050) | 0.137 [0.014, 0.268] (null 0.156, z=-0.45, p=0.693) |
| MLP Spearman | 0.332 [0.190, 0.465] (null 0.190, z=4.38, p=0.001) | 0.134 [0.032, 0.199] (null 0.107, z=0.72, p=0.237) | 0.054 [-0.024, 0.120] (null 0.073, z=-0.55, p=0.706) |
| residual cosine | 0.974 [0.962, 0.984] (null 0.967, z=1.66, p=0.033) | 0.872 [0.849, 0.899] (null 0.881, z=-1.32, p=0.880) | 0.894 [0.862, 0.916] (null 0.887, z=0.74, p=0.250) |
| residual Spearman | 0.930 [0.889, 0.950] (null 0.879, z=3.93, p=0.001) | 0.706 [0.594, 0.788] (null 0.726, z=-0.96, p=0.816) | 0.754 [0.688, 0.791] (null 0.744, z=0.52, p=0.329) |
| IoU top-5 | 0.429 [0.250, 0.429] (null 0.370, z=0.75, p=0.614) | 0.250 [0.181, 0.339] (null 0.265, z=-0.34, p=0.998) | 0.250 [0.250, 0.429] (null 0.299, z=-0.68, p=1.000) |
| IoU top-10 | 0.667 [0.538, 0.744] (null 0.598, z=1.16, p=0.394) | 0.484 [0.333, 0.603] (null 0.512, z=-0.66, p=0.827) | 0.538 [0.429, 0.667] (null 0.540, z=-0.07, p=0.967) |
| IoU top-20 | 0.779 [0.739, 0.818] (null 0.730, z=1.86, p=0.038) | 0.538 [0.481, 0.600] (null 0.540, z=-0.14, p=0.993) | 0.600 [0.538, 0.667] (null 0.570, z=1.05, p=0.433) |

#### Gemma-2-2B: pooled null, without event_year (n=28)

| statistic | EN-ES | EN-HI | ES-HI |
|---|---|---|---|
| MLP cosine | 0.510 [0.360, 0.586] (null 0.408, z=2.51, p=0.004) | 0.334 [0.239, 0.415] (null 0.268, z=1.48, p=0.059) | 0.282 [0.214, 0.426] (null 0.271, z=0.22, p=0.417) |
| MLP Spearman | 0.336 [0.203, 0.459] (null 0.233, z=2.51, p=0.010) | 0.208 [0.104, 0.266] (null 0.158, z=1.04, p=0.149) | 0.096 [-0.001, 0.205] (null 0.134, z=-0.76, p=0.794) |
| residual cosine | 0.973 [0.961, 0.993] (null 0.973, z=-0.18, p=0.581) | 0.873 [0.846, 0.904] (null 0.881, z=-0.94, p=0.832) | 0.909 [0.864, 0.930] (null 0.898, z=1.07, p=0.124) |
| residual Spearman | 0.926 [0.814, 0.956] (null 0.839, z=2.30, p=0.005) | 0.584 [0.484, 0.688] (null 0.635, z=-1.41, p=0.921) | 0.720 [0.623, 0.848] (null 0.743, z=-0.67, p=0.732) |
| IoU top-5 | 0.250 [0.111, 0.429] (null 0.141, z=2.00, p=0.149) | 0.111 [0.000, 0.250] (null 0.062, z=1.02, p=0.426) | 0.250 [0.111, 0.429] (null 0.219, z=0.60, p=0.677) |
| IoU top-10 | 0.538 [0.176, 0.538] (null 0.292, z=3.14, p=0.004) | 0.176 [0.082, 0.429] (null 0.182, z=-0.14, p=0.785) | 0.429 [0.176, 0.603] (null 0.418, z=0.25, p=0.724) |
| IoU top-20 | 0.703 [0.541, 0.818] (null 0.551, z=2.47, p=0.023) | 0.600 [0.538, 0.667] (null 0.527, z=2.15, p=0.067) | 0.600 [0.510, 0.739] (null 0.592, z=0.23, p=0.685) |


# Part 6. Results at a glance, and what decides the paper

Source: `GOAL.md`, `docs/HYPOTHESIS.md` (verdicts), the paper chapter (Part 5). One page; the paper chapter is the authority.

## 6.1 Llama-3.2-3B-Instruct (laptop, October)

| Question | Result |
|---|---|
| Is the fact read from a shared subject state? (swap test, 1 = fully shared) | EN to ES 0.90 to 1.21 (shared). EN to HI 0.86 and 0.87 (Llama), 0.93 and 0.81 (Gemma), shared. Base Llama 0.86 and 0.72 (the second is partial). |
| Does tracing predict which facts are shared? | No (rho -0.04). |
| Does an English rank-1 edit carry over? (traced layer / layer 14 / layer 21; 40 facts) | English success 0.80 / 0.93 / 0.72. Spanish 0.58 / 0.75 / 0.00. Hindi 0.00 / 0.10 / 0.00. |

## 6.2 Llama-3.2-3B-Instruct (Kaggle, 86 facts, final run)

| Question | Spanish | Hindi |
|---|---|---|
| Key similarity vs transfer, traced layer (Spearman) | **0.41** [0.20, 0.59] | **0.42** [0.19, 0.62] |
| Same, layer 14 | 0.06 | 0.02 |
| Edit layer controlled | 0.26 [0.004, 0.49] | 0.22 [0.011, 0.44] |
| Layer map pooled (within layer) | 0.29 [0.12, 0.44], supported | 0.17 [-0.02, 0.36], not supported |
| Loss beyond rewording (traced / layer 14) | 0.39 / 0.26 | 0.79 / 0.73 |
| Shared-key edit, best gain | 0.044 (null) | 0.044 (null) |

Pass bars (registered): Spearman at least 0.30 with the interval above 0, Holm-corrected over four cells; layer-controlled at least 0.20; wording gap above 0.15; shared-key gain at least 0.15.

## 6.3 Gemma-2-2B (Kaggle)

Not read yet. Attempt 4 is running (code `bundle-e186cf5d5e1b`). What will be read first when the Output arrives: whether the Spanish key-similarity result replicates (bar 0.30 with the interval above 0), then Hindi, the layer map, the wording control and the shared-key gain.

## 6.4 What decides the paper

| Outcome | Paper |
|---|---|
| Spanish and Hindi both pass the layer-controlled key-similarity test, on a second model too | Key similarity predicts transfer: the contribution (Q1 candidate). |
| Only Spanish passes | Partial claim (Q2). |
| Neither | The diagnostic paper: layer map, wording gap, null shared-key result (Q2 to Q3). |

## 6.5 Honest limits as of now

- One model with strong numbers; a second (Gemma) is not read yet.
- The Hindi layer-controlled pass is narrow, and the pooled layer map misses the bar.
- Geography facts only (non-geography untestable so far).
- Cross-lingual edit failure and orthogonal edit vectors are published; our novelty is the per-fact link, and it is worded "not found in prior work (abstracts and passages)".
- No baselines (ROME, MEMIT, in-context) on a standard benchmark yet; the shared-key fix did not work, so there is no better editor to present.
- No native-speaker check of the Hindi and Spanish facts.


# Part 7. Open items and next steps

## 7.1 When the Gemma Output zip arrives

1. Save it in `C:\Users\Som Praneeth\Desktop\reasearch\BABEL_EFFECT_ORGANIZED\08_waiting_for_gemma_output\` (and unpack a copy into `analysis\kaggle_output_<date>\` in the repo, write-once).
2. Read in this order: job list and failures in `driver.log`; fp16 decision; tracing; English-key edits; wording control; key similarity; key-mode comparison; layer map; graphs.
3. Check whether the Spanish key-similarity result replicates (bar 0.30, interval above 0), then Hindi, then partial rho with the edit layer controlled.
4. If a CPU analysis job failed (key similarity, key modes, graphs were never run on Gemma data), rerun it locally from the Output.
5. Register the verdicts in `docs/HYPOTHESIS.md` as an appendix (the registered text is not edited), add a Gemma section to the paper, rebuild the PDF, regenerate the manifest, rebuild this record.

## 7.2 Next models

Each new model needs: its own entry in `kaggle/steps.py`, a `driver.MODELS` entry, a config pack, a fingerprint, a registration in `docs/HYPOTHESIS.md`, a question-style smoke prompt checked on the CPU, and a dry run of every job type on the laptop GPU before the notebook goes to Kaggle.

| Order | Model | Why |
|---|---|---|
| 1 | Llama-3.2-3B base | Same family as the primary; tests whether instruction tuning matters for key similarity. |
| 2 | BLOOM-560M | Used by XME (2401.10521); small and comparable to published work. |
| 3 | mGPT-1.3B | Used by MEMLA (2406.11566). |
| 4 | BLOOM-1b7 | First pilot kept only 34 facts; needs a larger fact pool. |
| 5 | Qwen2.5-3B | Qwen appears in six published papers. |
| 6 | One 7 to 8B model | Only after a T4 test (both T4s or 4-bit; 4-bit changes the edited weights). |

## 7.3 Science that would raise the paper

- **Baselines on the same facts:** ROME, MEMIT, in-context; a Bi-ZsRE or MzsRE slice (no Hindi in those; check languages).
- **A causal test beyond the null shared-key edit:** for example editing a fact with a key made closer to the English key, pre-registered before it runs.
- **More relations** than geography (non-geography was untestable), registered first.
- **More languages** (two or three, one non-Latin non-Indic script).
- **Full reads** of the five closest papers: 2310.10378, 2601.17397, 2406.16416, 2504.09378, 2510.10280; then final related-work wording.
- **Tracing split across both GPUs** (GPU 1 idles about 44 to 73 minutes during tracing).

## 7.4 Housekeeping

| Item | Who |
|---|---|
| Hindi and Spanish native-speaker check of `data/audit/native_speaker_audit.csv` (362 facts) | You: a Hindi checker and a Spanish checker |
| OSF or Zenodo timestamp for the pre-registration | You: an account |
| Revoke the Hugging Face token | You |
| Confirm the Kaggle weekly GPU quota (about 30 hours is a guess) | You |
| Optional: rerun the English-key base edits on the laptop GPU (about 25 min, close other programs) to test the three analysis jobs on Gemma data | Needs your "restart" |
| Keep the one-folder copy in step with the repo | Assistant, on request |

## 7.5 How to revise this record

1. Edit the hand-written chapter in `paper/master/` (or the source document it copies).
2. Run `.venv/Scripts/python.exe paper/build_master.py`.
3. Check the PDF opens and the contents page matches.
4. Commit when you say so.


# Part 8. Pre-registrations

*Copied word for word from `docs/HYPOTHESIS.md` when this record was built. To change it, edit that file and rebuild.*

## HYPOTHESIS (pre-registration)

Written and committed before any causal tracing or circuit-overlap computation. At the time of writing, `code/causal_trace.py` and `code/circuit_overlap.py` do not exist and no attribution has been computed on any fact. Only P0 (smoke test) and P1 (fact filtering by greedy accuracy) have been run. Later edits go in the "Amendments" section at the bottom, with a date and a reason. Nothing above that line is edited after the P3 computation.

### Setup (fixed)
- Model: Llama-3.2-3B-Instruct (28 layers, d=3072), bf16, TransformerLens 3.9.0, `from_pretrained_no_processing`.
- Facts: `data/facts.json`, n=52 (26 capital_country, 22 capital_of, 4 landmark_city), correct in EN, ES and HI under greedy decoding.
- Tracing: Gaussian noise of 3x the subject-embedding std, per-fact seed; attribution = logit(object first token) minus logit(top competitor), patched minus corrupted; residual stream and MLP-output sites.
- Critical site per language = argmax attribution at that language's last subject token.
- Primary overlap statistic: per-layer attribution profile at each language's own critical token, compared by cosine and Spearman. Secondary: span-pooled top-k grid IoU, k in {5, 10, 20}.
- Null: shuffle fact IDs across languages (1000 permutations) so the EN profile of fact i is paired with the ES or HI profile of a different fact; z-score and one-sided p from the permutation distribution. Bootstrap 95% CIs over facts.
- Analysis set: all n=52 facts. Robustness set: the 48 facts excluding landmark_city (n=4 is too small to analyse alone).

### Predictions (directions, stated before computing)
Priors come from the causal-tracing literature (Meng et al. 2022, arXiv:2202.05262, for English GPT-style models), not from any measurement here.

- **H1 (shared locus).** Per-layer profile cosine and Spearman for EN-ES and EN-HI are above the permutation null (z > 2, p < 0.05, bootstrap CI above the null mean). Reason: factual recall in English models concentrates in mid-layer MLPs at the last subject token, and this layer-wise structure is likely shared.
- **H2 (distance ordering).** EN-ES overlap > EN-HI overlap, on both primary and secondary statistics. Reason: Spanish shares script and subword vocabulary with English; Hindi is a different script, uses multi-token subjects, and has lower accuracy (0.18-0.58 vs 0.65-0.78).
- **H3 (peak location).** The English critical layer lies in the middle third of the network (layers 8-19 of 0-27). Spanish peak within 2 layers of English (median over facts). Hindi peak the same or later than English, with lower peak magnitude.
- **H4 (editing).** Rank-1 edit at the EN traced layer succeeds in >= 80% of EN test facts. Transfer ratio satisfies 0 < HI < ES < 1 (median over facts). Edit at the random-layer control transfers less than the traced and mid-layer edits.
- **H5 (side effects).** Specificity (max |dP| over 20 control facts) below 0.05 and damage (perplexity ratio on 100 Wikipedia sentences) below 1.05 for the traced-layer edit.

### Interpretation rules (fixed in advance)
- Language-agnostic account supported: H1 holds for both pairs and transfer ratio for both languages has a bootstrap CI above 0.5.
- Language-specific account supported: overlap for both pairs is not distinguishable from the null (CI includes the null mean) and transfer ratio CI includes 0.
- Partial/mixed: anything else. In particular, a shared layer-wise profile with low edit transfer is reported as "shared locus, language-specific content", not as agreement.
- A result against these predictions is reported as such. Thresholds above are not changed after seeing data.

### Known limitations (declared now)
- n=52 and a skewed relation mix; CIs will be wide.
- Selection bias: kept facts are those the model already answers correctly in all three languages.
- The exact-match filter rejects some spelling variants (e.g. "Hanoi" vs "Hanói").
- The model weights come from an unsloth mirror, not byte-verified against Meta's repo.
- One model only; claims are limited to Llama-3.2-3B-Instruct.

### Amendments
- **A1 (2026-10-12, before any tracing computation; clarifications, not changes of direction).**
  - Critical site: the argmax over layers of the MLP-output attribution at the last subject token (the site type the ROME edit modifies). The residual-stream argmax is stored as a secondary quantity.
  - Noise scale: sigma = 3 x the std of subject-token embeddings, pooled over all 52 facts per language (as in ROME), computed once and independent of how many facts a run traces.
  - Object token and competitor: object = the model's clean top-1 token at the last prompt position (the filter guarantees it begins the object string); competitor = the clean runner-up token. Attribution = (logit_obj - logit_comp) patched minus corrupted, with the final dot product computed in fp32.
  - Collapse check: a fact-language pair counts as collapsed if P(object) under corruption is below 0.5 x its clean value. Facts that do not collapse in all three languages are excluded from the primary P3 analysis. If that leaves n < 50, the primary analysis uses all 52 facts and the exclusion becomes a sensitivity analysis.
- **A2 (2026-10-12, after the first P2 trace on 10 facts, before any P3 computation; a method fix, not a change of prediction).** In run `analysis/run_20261012_p2trace`, with the competitor defined as the clean runner-up token (A1), 5 of 30 fact-language pairs had a negative total effect (clean score below corrupted score), so their attribution grids are not interpretable: capital_country_002 hi, capital_of_000 hi, landmark_city_003 en, landmark_city_009 es, landmark_city_009 hi. Fix: the competitor is the highest-logit non-object token in the corrupted run (the token that wins when the subject is destroyed). The score is still logit(object) - logit(competitor), patched minus corrupted. The superseded run stays on disk unchanged. All P2/P3 numbers use the new definition. Observed in that first run and noted here for honesty: MLP peaks at the last subject token fall mostly in layers 0-10, not in layers 8-19 as H3 predicted; H3 is evaluated against the new run.
- **A3 (2026-10-12, after P2 on 10 facts, before any P3 overlap computation).**
  - *P2 result recorded against H3.* In `analysis/run_20261012_p2trace_v3` (10 facts x 3 languages, all 30 pairs collapsed and had a positive total effect), the critical MLP layer at the last subject token was 0-5 in every pair: EN median 2.5 (IQR 0.5-3.0), ES 2 (2-2.75), HI 2 (0.25-2); 0 of 10 per language fall in layers 8-19. The prediction in H3 that the English peak lies in layers 8-19 is **not met** on this sample; the peak is in the first third of the 28 layers. The residual stream at the last subject token carries the effect up to about layer 20, then it moves to the last token. Hindi effect at the subject token is about half the English/Spanish size (residual ~0.4 vs ~0.8 of the total effect), consistent with the "lower magnitude" part of H3. The H3 "Spanish within 2 layers of English" and "Hindi same or later" parts are evaluated on the full set in P3. Layer 0 is the argmax in several pairs; this may partly reflect undoing the embedding corruption rather than fact lookup, and is reported as is.
  - *Definition of the P3 statistics (fixed now).*
    - Primary profile: the MLP-output attribution at the last subject token, one value per layer (28), per fact and language, raw (not normalised). Per-fact similarity between a language pair = cosine and Spearman of the two profiles. Test statistic = median over facts. Secondary profile: the same for the residual stream.
    - Secondary statistic: top-k IoU, k in {5, 10, 20}, on a grid of cells (kind in {resid, mlp}) x (layer) x (region in {subject span summed over its tokens, last prompt token}), 112 cells; the top-k cells by attribution per language are compared.
    - Null: for each of 1000 permutations, pair fact i's EN profile with the ES (or HI) profile of fact pi(i), recompute the statistic; z = (observed - null mean)/null sd; one-sided p = (1 + #{null >= observed}) / 1001. Seeds are logged. 95% CIs by bootstrap over facts (2000 resamples).
    - Sets: primary = facts valid under A1/A2 in all three languages; sensitivity = all kept facts, and without landmark_city (only 2 facts kept).
  - *What the null can and cannot show.* The shuffled-fact null measures similarity beyond what any two facts' profiles share, i.e. fact-specific alignment. If all facts share one generic layer profile, observed similarity can be high while z stays near 0; this is reported as "shared generic locus, no fact-specific alignment", not as support for or against a language-agnostic store.
- **A4 (2026-10-12, after P3, before any edit is run; fixes the P4/P5 design).**
  - Edit set: the 10 P2 facts (4 capital_country, 4 capital_of, 2 landmark_city). New object for fact f = the object of another valid fact of the same relation, chosen with an RNG seeded by f's seed (never the same English object). The new object is applied in the language of each prompt (EN/ES/HI forms of that other fact).
  - Conditions (per fact): (a) the fact's own EN critical MLP layer from the P3 trace, (b) layer 14 (L//2), (c) a random layer in 0-27 excluding (a) and (b), RNG seeded by the fact seed. 30 edits in total.
  - Edit method: ROME-style rank-1 update of the MLP down-projection (`W_out`) at the chosen layer, applied once, edit language = EN only. Key k* = mean MLP hidden activation at the last subject token over 5 EN contexts (prompt with prefixes "", "Fact: ", "Note: ", "Q: ", "Did you know? "). Covariance C = second moment of those activations over 30,000 WikiText-2 train tokens (128-token windows), ridge 1e-3 x mean diagonal. Value v* = current output + delta, delta fitted by 25 Adam steps (lr 0.5) on cross-entropy of the new object's first token at the last prompt position (mean over contexts) + 0.0625 x KL on the prompt "<subject> is" + weight decay 0.5 x |delta|^2/|v0|^2, with |delta| clamped to 4 x |v0|. Update dW = (C^-1 k*) outer (v* - W k*) / ((C^-1 k*) . k*). Weights restored after every edit.
  - Metrics: dP = P(new object first token) after minus before, per language, at the last prompt position. EN edit success = after the edit the EN top-1 token is the new object's first token (gate: >= 80% in condition (a)). Transfer ratio = dP_L2 / dP_EN, median over facts with bootstrap 95% CI. Specificity = max |dP| of the original first-token probability over 20 control facts (EN prompts, other valid facts, sampled with the fact seed). Damage = perplexity after / before over 100 sentences from the WikiText-2 test split (8-40 words, seeded). Same per-condition reporting for all three conditions.
  - Caveat declared now: the object metric is the first subword, so for ES/HI "transfer" means the first subword of the new object moves.
- **A5 (2026-10-14, before any v2 data or run; Study v2 design).** The v1 study (Llama-3.2-3B, 3 capital-type relations, first-token scoring, 10 edit facts) stays on record as the pilot. Study v2 is the confirmatory study for a full-paper submission. Everything below is fixed before any v2 candidate fact is filtered or any v2 run is made. Observations from v1 that motivated v2 are stated as such; the v2 tests of them are confirmatory only for the predictions listed in H6-H9.
  - **Models.** Llama-3.2-3B-Instruct (primary) and Gemma-2-2B (secondary, base model), both bf16 (Gemma estimate ~6.0 GiB weights; 4-bit is not used because it is untested in TransformerLens and alters the weights being edited). Further models (Llama-3.2-1B, Qwen3-1.7B-Base, BLOOM-1b7) are added only if a forward-only pilot keeps n >= 50 facts correct in all three languages. A model is never dropped after seeing its overlap or edit results.
  - **Relations.** capital_of, capital_country, landmark_city (more candidates), invention_inventor ("X was invented by" -> inventor), event_year ("X took place in the year" -> year). EN/ES/HI forms written by hand; every relation analysed separately only if it keeps >= 10 facts; the pooled analysis uses all kept facts.
  - **Phrasings.** 3 phrasings per fact and language. A fact-language pair is correct if the greedy continuation (leading spaces ignored, a leading newline fails) begins with the exact full object string for the canonical phrasing AND at least one other phrasing. A fact is kept per model if correct in EN, ES and HI. The canonical phrasing is used for tracing and editing; the other phrasings are used for the filter and for robustness checks. Target n >= 50 kept facts per model.
  - **Metrics (changed from v1).** Primary attribution = indirect effect on the full-answer log-probability: sum over the object's tokens of log P(token | prompt + preceding object tokens) under teacher forcing, patched minus corrupted. The v1 first-token logit difference is kept as a secondary metric. Collapse = full-answer probability under corruption below half its clean value. Edit success = the greedy full-answer continuation after the edit equals the new object string; dP = change in full-answer probability of the new object. Transfer ratio, specificity and damage are defined as in A4, with full-answer probabilities (specificity uses the original full-answer probability of each control fact).
  - **Layer sweep.** For each model, 10 edit facts (stratified over relations, chosen by sorted id) x every layer 0..L-1, English prompt only. The "traced", "mid" and "random" conditions of A4 are read from this sweep (traced = the fact's own EN critical layer, mid = L//2, random = a seeded layer), so all conditions use identical optimisation. Reported: EN success, ES and HI transfer, specificity and damage against layer, with bootstrap 95% CIs over facts.
  - **Statistics.** Same as A3 (median over facts, shuffled-fact permutation null with 1000 permutations, 2000 bootstrap resamples, one-sided p). Primary tests per model: EN-ES, EN-HI, ES-HI MLP-profile cosine and Spearman (6 tests); Bonferroni alpha 0.05/6 within a model. Cross-model statements compare z-scores and CIs descriptively; no test of difference is claimed.
  - **Predictions (directions).**
    - H6 (replication): EN-ES MLP-profile overlap is above the null (z > 2) in both models; EN-ES overlap exceeds EN-HI overlap in both models. No prediction that Hindi overlap is at the null in Gemma.
    - H7 (layer sweep; motivated by the v1 observation that a layer-14 edit transferred more to HI than the traced early-layer edit): median ES and HI transfer ratio at layers within 3 of L//2 exceed the median transfer ratio at layers 0-5, in both models; specificity loss is larger at the mid layers than at layers 0-5.
    - H8 (phrasing robustness): within a language, per-layer MLP profile cosine between phrasings of the same fact exceeds the cross-language cosine for the canonical phrasings, in both models.
    - H9 (relation generality): EN-ES overlap above the null (z > 2) in at least 2 of the 3 new or extended relations in the primary model, with the null recomputed within relation.
  - **Interpretation rules.** As in the original rules: shared-locus support needs the overlap and transfer criteria to hold together; a result against any of H6-H9 is reported as such and the text of this amendment is not edited afterwards.
  - **Declared limitations.** Gemma is a base model and Llama an Instruct model (a model and training-stage confound); n per relation may fall below 10; exact-match scoring still rejects spelling variants; year and name objects can be copied across languages, which may inflate transfer for event_year and invention_inventor. These two relations are therefore also analysed without them.
- **A6 (2026-10-14, before the v2 filter is run; clarification of A5 matching).** A continuation matches if, after dropping leading spaces, it begins with the object string or one of its listed alternative surface forms, compared case-insensitively (so "the Netherlands" matches "The Netherlands"). A leading newline still fails. The primary object string used for scoring in tracing and editing is the longest matched form actually generated by the model for the canonical phrasing (the model's own correct answer). Generation is batched with left padding (greedy, 24 new tokens); borderline differences from unbatched decoding are possible and are checked in tracing, where the clean top-1 token must equal the first token of the matched string. Candidate pool: 408 facts (125 capital_of, 125 capital_country, 81 landmark_city, 40 invention_inventor, 37 event_year), written before any model was run on them.
- **A7 (2026-10-14, after the first v2 filter run on Llama kept only 38 of 408 facts, before any v2 tracing, overlap or editing).** Inspection of the raw generations (analysis/run_20261014_p1v2_llama) showed that most failures were prompt-format and matching problems, not missing knowledge: Hindi fragments such as "<invention> का आविष्कार करने वाले व्यक्ति का नाम" continue with a question ("क्या है?"); question-answer prompts restate the subject before the answer; Hindi spellings differ only by a nukta. Changes, all independent of the overlap and edit outcomes (H6-H9):
  - **Template selection.** For each relation and language a pool of 6 answer-first templates is written in code/select_templates.py. The 30 lowest-id candidate facts of that relation (the tuning set) are decoded with every template using the strict rule below; the 3 templates with the highest accuracy become phrasings 0-2 (best = canonical, ties broken by pool order). Template accuracies on the tuning set are logged. Question-answer templates are dropped.
  - **Normalisation for matching.** NFC, lower-case, nukta (U+093C) and zero-width characters removed, hyphens treated as spaces, whitespace collapsed; applied to both generation and object forms.
  - **Strictness.** Phrasing 0 (canonical, the one traced and edited) must begin with the object (strict prefix, as in A6). Phrasings 1 and 2 are correct if the object occurs within the first line and the first 60 characters of the generation (so restating the subject is tolerated there); a fact-language pair is kept if phrasing 0 and at least one other phrasing are correct, as in A5.
  - **Consequence for reporting.** The tuning facts are part of the dataset; templates were chosen on correctness only. The v1 files and the first v2 filter run stay on disk as the record of the first attempt.
- **A8 (2026-10-15, after Gemma-2-2B kept 45 of 408 facts with its own selected templates, before any further Gemma run).** The Gemma-2-2B candidate pool is extended by 39 historical events (code/build_facts_v2.py, `--with-events2`; years distinct from every earlier event), because the n >= 50 requirement of A5 is otherwise not met for this model and events are the relation Gemma answers in Hindi. Gemma uses its own selected templates (data/v2/templates_gemma2.json; selection tuning set unchanged because the new events are appended after the first 30). Llama's fact set (92 valid facts) is not re-run. Cross-model statements are made on each model's own fact set and, additionally, on the facts kept by both. No model is dropped after seeing results; if Gemma still falls below 50 it is reported with that caveat.
- **A9 (2026-10-15, after the Llama and Gemma v2 overlap runs, before any further run; analysis rules for everything that follows).**
  - **Primary null.** For all new data and models, the primary overlap test uses a relation-stratified shuffled-fact null (permutations swap only facts of the same relation, 1000 permutations); the pooled null is reported as secondary. The pooled-versus-stratified distinction was noticed after the first v2 results (pooled significance can come from relation-level similarity), so for the Llama and Gemma results already obtained, the stratified result is labelled post hoc. Six primary tests per model (3 pairs x cosine and Spearman), Bonferroni alpha 0.05/6 = 0.0083; a relation is analysed separately only with >= 10 valid facts.
  - **Non-copyable analysis.** Two relations are added whose answers differ across languages: landmark_country (landmark -> country) and city_country (non-capital city -> country), candidates written before they are run. A "non-copyable" set excludes event_year and invention_inventor. Target: >= 50 valid non-copyable facts per model; if not reached it is reported as not reached. Main claims are stated on the non-copyable set as well as on all facts.
  - **In-context baseline (descriptive, no prediction).** For each edit fact, the new target is placed in an English statement, "Fact: <canonical English prompt> <new target>.", followed by a newline and the canonical prompt in each language; dP and greedy success are measured as for the weight edit. This puts the weight-edit transfer in context; no hypothesis is tested.
  - **H8 subset.** Phrasings 1 and 2 are traced for the first 30 valid facts by sorted id per model (all three languages).
  - **Third model.** Llama-3.2-1B-Instruct is piloted first (same tokenizer and code path); the full pipeline runs only if its filter keeps >= 50 facts under the A7 template-selection procedure, using the same pool (including the A8 events). Qwen3-1.7B-Base and BLOOM-1b7 follow only with their own pilots.
  - **More edit facts.** After the sweep, a second edit run uses 30 facts per model at layers chosen from the sweep by an aggregate rule fixed before that run: the layer with the highest median EN success among layers where median specificity loss is below the sweep median, plus the traced layer of each fact and L//2.
- **A10 (2026-10-19, before any BLOOM-1b7 data exists; before the Llama-3.2-1B pilot result is known).** A third architecture family is added because Llama and Gemma tokenise Hindi poorly (median Hindi subject span 5 and 8 tokens against 2 for English) and the Hindi results depend on span length (F2). BLOOM-1b7 (bigscience/bloom-1b7, base model, 24 layers, ALiBi, GELU MLP, trained on Hindi; measured 1.43 tokens per Hindi word in research_notes) tests whether the weak Hindi overlap is a tokenisation/training-data effect.
  - **Procedure.** Same pipeline and code as the other models (A6/A7 template selection with the same pool and A8 events, strict matching, full-answer tracing, relation-stratified null as primary per A9, 1000 permutations, Bonferroni 0.0083 over six tests). The edit step is limited to the three A5 conditions (the fact's traced layer, L//2 = 12, and one random layer fixed now as 18 by `random.Random(20261019).randrange(24)`) for 10 facts, run only if the filter keeps >= 50 facts and tracing is valid for >= 50. Weights are verified against the official SHA-256 before use. BLOOM's tokenizer adds no BOS token; prompts are used as tokenised (declared difference).
  - **Predictions (directions, stated before computing).** B1: the filter keeps >= 50 facts (otherwise BLOOM is reported as a pilot only and no further stage runs). B2: EN-ES MLP-cosine z > 2 under the stratified null (consistent with H6). B3 (tokenisation hypothesis): median Hindi subject span <= 3 tokens, and EN-HI stratified z > 2 with EN-HI cosine not lower than ES-HI minus its CI. If B3's EN-HI condition fails while the span is <= 3, Hindi weakness is not explained by tokenisation alone and the paper says so; if it holds, the Llama and Gemma Hindi weakness is attributed (as a hypothesis, not a proven cause) to tokenisation/training exposure. No model is dropped after seeing results.
  - **Limits declared now.** One more model is a small sample of architectures; BLOOM is a base model; its factual accuracy at 1.7B is lower than Llama's, so the kept set may be small and dominated by one or two relations.

### Verdicts (appended 2026-10-20, after the runs; the amendment texts above are not edited)
- **H6** (EN-ES z > 2 and EN-ES > EN-HI in both models): supported. Pooled null: Llama z 4.22 (n=92), 5.06 (n=126); Gemma z 4.81 (n=58), 3.86 (n=59). Relation-stratified null: Llama 4.24, Gemma 3.18 (n=126/59).
- **H7** (mid-layer edits transfer more than layers 0-5, with more specificity loss), 10 facts per model: supported in Gemma, contradicted for Spanish in Llama; no pooled claim. The 20-fact Llama follow-up (traced, L//2, random layers only) does not re-test it.
- **H8** (within-language phrasing cosine > cross-language cosine): supported in both models (Llama 0.754 vs 0.609, d = 0.139 [0.101, 0.228]; Gemma 0.741 vs 0.338, d = 0.341 [0.287, 0.463]).
- **H9** (EN-ES z > 2 in at least two new or extended relations, Llama): not supported among the pre-registered relations (event_year z = 0.91; the others had < 10 facts). city_country (A9, later) z = 1.96.
- **F4** (>= 50 non-copyable valid facts): met for Llama (104), not met for Gemma (29).
- **F3, F7**: descriptive. Gemma F7 did not finish (interrupted); B1-B3 (BLOOM-1b7) and the Llama-3.2-1B pilot have not run.

- **A11 (2026-10-21, after an external review of draft v2 and before any of the following is computed; no new GPU run).** The review argued that the EN-ES overlap may reflect generic profile shape and English/Spanish subject-name overlap rather than fact-specific shared lookup. Controls on the cached traces (Llama: 126 valid facts, main + extra runs; Gemma: 59), code/controls_overlap.py:
  - **C0 (descriptive).** Mean cosine of the last-subject-token MLP profile for: same fact EN-ES; different fact, same relation, EN-ES; EN-EN and ES-ES different fact, same relation. The fact-specific increment is same-fact minus different-fact EN-ES.
  - **C1 (centred statistic).** Profiles are centred by subtracting the per-language mean profile over facts; EN-ES median cosine with the relation-stratified null at 10,000 permutations. Prediction: z > 2 in both models.
  - **C2 (lexical control).** Lexical similarity of the EN and ES subject strings = difflib ratio after accent stripping and lower-casing. Facts split at the median similarity; EN-ES centred and uncentred stratified z within each half; Spearman correlation of similarity with the per-fact EN-ES cosine. Prediction (shared-lookup reading): z > 2 in the LOW-similarity half in both models. If it fails there but holds in the high-similarity half, the lexical explanation is supported and the shared-lookup claim is withdrawn for these data.
  - **C3 (entity dependence).** Facts sharing an English subject or answer string are clustered (union-find); 95% CI of the median EN-ES cosine by cluster bootstrap (2000 resamples).
  - Reported whatever the result. This amendment is not edited afterwards.

- **A12 (2026-10-21, after the A11 results and a second external review; before any A12 statistic is computed; no GPU).** Confound-adjusted analysis of the cached traces (same trace sets as A11), code/confound_overlap.py. Seen before writing this: the low-name-similarity half is dominated by event_year (Gemma: 20 of 30 facts; Llama: 22 of 64, and all 22 event_year facts), i.e. by digit-string answers. This is a known confound of the A11 "low-similarity" result, not a finding.
  - **Profiles.** Relation-centred: subtract the mean profile of the fact's relation, per language (relations with < 10 valid facts merged into "other").
  - **S1 increment.** s_ij = cosine(centred EN profile i, centred ES profile j); D_i = s_ii - mean over same-relation j != i of s_ij; report mean D with a 95% entity-cluster bootstrap CI (2000 resamples) and an empirical p from 10,000 within-relation permutations of the ES labels, p = (1 + #{null >= obs}) / (B + 1). Computed for all valid facts, for non-event_year facts only, and for event_year only.
  - **S2 retrieval.** Top-1 accuracy and MRR of the matching ES profile among same-relation ES profiles, against the same permutation null.
  - **S3 depth bands.** S1 within three fixed layer bands (thirds of the layer count: early, middle, late), fixed now.
  - **S4 covariate model.** D_i regressed on: subject-name similarity (difflib ratio, accent-stripped), |EN span - ES span| in tokens, answer-string-identical flag, log of the smaller total effect of the two languages, relation (fixed effects); OLS with entity-cluster robust SEs. The intercept is the name-free, span-matched, non-copyable alignment.
  - **S5 identity source.** Mean s_ij for different-fact pairs sharing a subject string or an answer string versus pairs sharing neither (exploratory if < 20 pairs).
  - **Smallest effect of interest.** One third of Gemma's mean D in S1 (all facts), in the same statistic. "No name-free alignment" is claimed for a model only if the upper 95% bound of the S4 intercept is below this value; a non-significant result alone is not evidence of absence.
  - **Predictions (directions).** Gemma: S1 D > 0 with CI excluding 0 on non-event_year facts and on all facts; S3 effect in the middle band; S4 intercept CI excludes 0. Llama: S4 intercept small, CI includes 0. If Gemma's S1 fails on non-event_year facts (n = 29) it is reported as unresolved at that n, not as absence.
  - **Not covered.** Cross-language interchange patching and name-dissimilar new facts need GPU runs and get their own amendments (A13+) before any data exist. Reported whatever the result; this amendment is not edited afterwards.
- **A12a (2026-10-23, after an external review of A12 and before any A12 or A12a statistic is computed; CPU only, cached traces, no GPU). A12 is not edited and is run exactly as written; A12a adds analyses and corrects how two of them are read.**
  - **Why.** Code inspection and numbers already in the manifest show four problems in A12 as written. None of them comes from an A12 result.
    - (1) The corruption noise is shared across languages. `causal_trace_v2.py` draws the subject noise from the per-fact seed (sha256 of the fact id) for every language and every phrasing. Subject token k therefore gets the same noise row in EN, ES and HI. With subject-first templates and identical names, the EN and ES token prefixes up to the last subject token are identical, so the measured site is computed identically in both languages, apart from sigma.
    - (2) A12 centres unnormalised profiles. A centred profile then points along the mean profile when the fact's total effect is above average. A correlation of total effects across languages therefore shows up as alignment.
    - (3) The S4 intercept is evaluated at the reference relation, at log total effect = 0 and at name similarity = 0. This is an extrapolation, and it makes the A12 absence rule unusable.
    - (4) The A12 smallest effect of interest is defined from Gemma's own D.
    - Also: the A11 Spearman broke ties by fact order (46 Llama similarities equal 1.0), and the A11 and A12 permutations allow fixed points.
  - **Seen before writing this.** The A11 results, the per-relation z-values in docs/LOG.md, the code, and the external review's predictions (Q1-Q5 below). No A12 or A12a statistic and no prefix-class count has been computed.
  - **Sets.** As A12: Llama 126 and Gemma 59 valid facts (trace runs as in `code/controls_overlap.py` SETS), phrasing 0. Relations with < 10 valid facts are merged into "other" (as in A12). These merged labels are the strata for centring, for the different-fact baselines and for the permutations.
  - **Normalised-centred (NC) profiles.**
    - p = the MLP-output indirect effect per layer at the last subject token (as in A3/A5).
    - u = p / ||p||. The NC profile is u minus the mean u of the fact's stratum, per language.
    - Secondary variant: u − (u·m̂)m̂, where m̂ is the unit stratum mean.
    - Facts with ||p|| < 1e-8 are excluded and listed.
  - **Prefix classes (EN vs ES, from the token ids stored in the cached grids).**
    - IP: identical token ids from BOS up to and including the last subject token.
    - EQ: not IP, and the two subject spans have the same number of tokens, so the last subject token gets the same noise row.
    - OT: all other facts.
    - **Rule fixed now, whatever the counts:** IP facts are uninformative about cross-language sharing by construction. They are excluded from every cross-language claim from now on and reported only descriptively.
  - **Token-level covariates.** Computed with each model's own tokenizer files; no weights are loaded.
    - J: Jaccard of the sets of EN and ES subject token pieces (decoded, whitespace stripped, lower-cased, NFKD accents removed, empty pieces dropped).
    - LT: last subject token id identical.
    - NR: equal span length, i.e. a shared last-token noise row.
    - SD: |EN span − ES span| in tokens.
    - AI: answer strings identical after `matching.norm`.
    - TE_en, TE_es: log total effect (score_clean − score_corr) in each language, centred at the model mean.
    - The A12 difflib similarity is kept as a secondary covariate.
  - **S1a (NC increment).**
    - D_i = s_ii − mean over same-stratum j ≠ i of s_ij, on NC profiles (the A12 S1 definition).
    - Report the mean D with a 95% entity-cluster bootstrap CI (A11 clusters, 2000 resamples).
    - Empirical p from 10,000 within-stratum derangements of the ES profiles (no fixed points): p = (1 + #{null ≥ obs}) / (B + 1).
    - Sets: (i) **informative = not IP and not event_year (primary)**; (ii) all valid facts; (iii) event_year only; (iv) each prefix class, overall and within each stratum that has ≥ 5 facts in that class.
  - **S1b (magnitude diagnostic).**
    - Spearman (average ranks) of TE_en vs TE_es within stratum: ranks demeaned per stratum, with a stratified permutation p.
    - Mean D on unnormalised relation-centred profiles (A12 S1), reported next to S1a on the same sets.
  - **S4a (covariate model).** It replaces the coding of A12 S4; A12 S4 is still reported as written.
    - On informative facts, D_i (NC) is regressed by OLS on J, LT, AI, NR, SD, TE_en, TE_es and stratum (sum-to-zero coding), with entity-cluster robust SEs (CR1).
    - Target quantity: the predicted D at J = 0, LT = 0, AI = 0, NR = 0, SD = 1, TE at the mean, strata averaged with equal weight. This is "name-free, no shared noise row, non-copyable" alignment.
    - Report it with a 95% CI from the robust SE, plus a cluster-bootstrap CI as robustness.
    - Report how many facts have J = 0 and how many have NR = 0. If fewer than 10 facts have J = 0, the target is labelled an extrapolation.
  - **Corrections of A11 (labelled as such).**
    - C2 Spearman is recomputed with average ranks and a within-stratum permutation, on raw cosine and on NC D.
    - C1 is recomputed with derangements.
    - Both the old and the corrected versions are reported.
  - **Fixed smallest effect of interest.** δ = 0.05 cosine units of NC D, for both models. It is about half the smallest A11 same-minus-different increment (0.090, Llama) and was chosen without any A12 or A12a number. A12's own SESOI is also reported.
  - **Decision rules.** Per model, on the two primary quantities (S1a on informative facts and the S4a target), using 97.5% CIs (Bonferroni over 2):
    - Present: lower bound > 0.
    - Absent: upper bound < δ.
    - Unresolved: anything else.
    - A12a cannot remove shared-noise effects from EQ and OT facts. A "present" verdict is therefore provisional until the independent-seed re-trace (A13a), and it is worded "not explained by prefix identity, profile magnitude or token overlap".
  - **Predictions (directions, from the review, stated before computing).**
    - Q1: within the same stratum, mean D is higher for IP than for OT facts in Llama.
    - Q2: on all valid facts, Gemma's D on NC profiles is smaller than its D on unnormalised centred profiles (A12 S1).
    - Q3: Gemma S1a on informative facts has a CI that includes 0.
    - Q4: the Llama S4a target has a CI that includes 0.
    - Q5: TE_en and TE_es are positively correlated within stratum in both models.
    - Results against these predictions are reported as such.
  - **Implementation.** `code/confound_overlap.py` with the flag `--a12a`. Seed 20261023 for the bootstrap and the permutations. Results go to a new write-once run directory, and every reported number enters the manifest. Results are reported whatever they are, and this amendment is not edited afterwards.
  - **Not covered.** Independent noise seeds per language with averaged noise draws (A13a) and cross-language interchange patching (A13) need GPU runs. Each gets its own amendment before any data exist.

- **A13a (2026-10-24, after the A12/A12a CPU results and before any independent-noise trace exists; needs the GPU).** Does the EN-ES profile alignment survive when each language gets its own corruption noise? Seen before writing this: A12a results (analysis/run_20261024_confound_A12a). In them, prefix-identical facts are rare (Llama 1, Gemma 4), but facts whose EN and ES subjects have the same token count (and so share the last-token noise row) show much larger NC increments than the rest: Llama EQ 0.436 vs OT 0.145; Gemma EQ 0.492 vs OT 0.002. This is the motivation, not evidence of an artefact.
  - **Code.** `code/trace_seeds.py` (restores only `hook_mlp_out` at the last subject token, all layers in one batch; same scorer, sigma, templates and facts as the original tracer; phrasing 0; EN and ES only).
  - **Sets.** The valid facts of A11/A12a (Llama 126, Gemma 59), same strata and prefix classes.
  - **Draw sets per fact and language** (noise = N(0, sigma^2) per subject token as in the original tracer, sigma from the original run configs):
    - IND: K = 10 draws, seed = sha256("id|lang|k") mod 2^31 (independent between languages). Replicates IND-A = draws 0-4, IND-B = draws 5-9.
    - SH: K = 5 draws, seed = sha256("id|shared|k") mod 2^31, identical for EN and ES (shared noise, drawn from the same code path).
    - ORIG: 1 draw with the original per-fact seed (regression check).
    - Profile of a draw set = mean over its draws of the restoration effect (lp restored - lp corrupted) per layer.
  - **Regression test (P4).** ORIG profiles must reproduce the cached profiles: median absolute difference < 0.05 nats and correlation > 0.99 across facts; otherwise the new tracer is not used for any claim.
  - **Statistic.** NC increment D exactly as A12a S1a (unit-normalise, stratum-centre, D_i against same-stratum different facts, mean D, entity-cluster bootstrap 97.5% CI, 10,000 within-stratum derangements). Computed for: (a) D_ind = EN IND vs ES IND (10 draws each); (b) D_sh = EN SH vs ES SH; (c) D_orig; (d) 5-draw versions D_ind5 (A vs A and B vs B, averaged) for a fair comparison with SH; (e) within-language ceilings D_EN-EN' (EN IND-A vs IND-B) and D_ES-ES'; reliability index R = D_ind5 / sqrt(D_EN-EN' x D_ES-ES') when both ceilings are positive.
  - **Sets analysed.** All valid facts; informative (not prefix-identical, not event_year); EQ; OT; event_year only; non-event_year.
  - **Paired noise effect.** Delta_i = D_i(SH) - D_i(IND5) per fact; mean with entity-cluster bootstrap CI, overall and within EQ and OT.
  - **Gate G1 (per model, on informative facts, 97.5% CI of D_ind):** SURVIVES if the lower bound > 0; ARTEFACT if the interval includes 0 and the upper bound < 0.05; UNRESOLVED otherwise. If either ceiling D_EN-EN' or D_ES-ES' is < 0.10, the verdict is UNRESOLVED (profiles too noisy at K = 10 to read).
  - **Predictions (directions).** P1 Gemma: D_ind on informative facts has a CI that includes 0, and mean Delta > 0 with CI excluding 0 (shared noise inflated the alignment). P2 Llama: D_ind on informative facts > 0, and smaller than the A12a value 0.239. P3: D_sh >= D_ind in both models on EQ facts. P4: regression test passes.
  - **Reported whatever the result; this amendment is not edited afterwards.** Cross-language interchange patching (E2) is A13 and gets its own amendment before any data exist.

- **A13b (2026-10-24, after the A13a result, before any covariate model is fitted on the A13a profiles; CPU only).** A13a showed the EN-ES NC alignment survives independent noise (Llama informative D 0.346, Gemma 0.257). The question left is whether it is explained by token overlap, answer identity, span or magnitude. The A12a S4a covariate model is applied unchanged to the per-fact increments D_i computed from the A13a IND profiles (10 draws per language): same facts, strata, covariates (J, LT, AI, NR, SD, TE_en, TE_es), sum-to-zero stratum coding, CR1 entity-cluster robust SEs, target = predicted D at J = 0, LT = 0, AI = 0, NR = 0, SD = 1, TE at the mean. Informative facts only (not prefix-identical, not event_year). Fixed delta = 0.05 and the A12a decision rules (present: lower 97.5% bound > 0; absent: upper bound < 0.05; else unresolved) are unchanged. Also reported: the same model on all non-prefix-identical facts with a relation x AI term dropped (event_year included), and the unit-profile (not centred) variant is not repeated. Prediction (direction, stated now): the target is > 0 in Llama (CI excluding 0); in Gemma the CI is wide (n = 25) and may be unresolved. Reported whatever the result; not edited afterwards. Script: `code/seeds_covariates.py`.

- **A13 (2026-10-24, before any E2 data or E2 template/filter run exists; GPU).** Cross-language subject interchange: is the subject representation computed in English interchangeable, for the downstream recall computation, with the one computed in Spanish? Seen before writing this: A12a, A13a and A13b results (alignment of tracing profiles survives independent noise and the covariate model in Llama; unresolved at n=25 in Gemma). The tracing profile cannot show shared retrieval; this test can.
  - **Models and facts.** Llama-3.2-3B-Instruct and Gemma-2-2B. Candidate pool: data/v2 candidates for capital_of, capital_country, landmark_city, landmark_country, city_country (event_year and invention_inventor excluded: identical answers across languages). A candidate is KEPT for a model if the model answers correctly in BOTH EN and ES (strict prefix match, as A6) with the E2 prefix templates; Hindi is not required (conditioning on Hindi selects famous entities).
  - **Prefix templates (fixed now; at least 2 language-specific tokens before the subject).**
    - capital_of: EN "The capital of {s} is" / ES "La capital de {s} es la ciudad de".
    - capital_country: EN "The country whose capital is {s} is" / ES "El país cuya capital es {s} es".
    - landmark_city: EN "The city where {s} is located is" / ES "La ciudad donde está {s} es".
    - landmark_country and city_country: EN "The country where {s} is located is" / ES "El país donde está {s} es".
    - The first 30 candidates per relation by id (tuning facts) are used only to confirm that the template gives >= 50% accuracy in each language; if a relation fails, it is dropped. Tuning facts are excluded from evaluation.
  - **Pairs.** Ordered (target t, source s), same relation, different first answer token in the target language, t != s. Two sources drawn per target by a fixed seed (20261024). The CONFIRMATORY subset requires the source's EN and ES subject token sets to share no piece (J = 0 under the model tokenizer, A12a definition); targets are unrestricted. Primary direction EN->ES (source computed in English, patched into the Spanish target run); ES->EN is secondary.
  - **Intervention.** `blocks.{l}.hook_resid_post` at the last subject token of the target prompt <- the source's value at its last subject token, one layer per batch row, l = 0..L-1. Site B (secondary): same at the final prompt token. Conditions per pair, layer and direction: `same` (source = s in the target language; ceiling), `cross` (source = s in the other language; the test), `self` (source = t in the other language; retention), `ctrl` (source = s in the other language at the token before the subject; disruption floor).
  - **Readout.** Teacher-forced full-answer log-probability (SeqScorer). Delta(l) = change of [lp(answer of s) - lp(answer of t)] in the target language relative to the unpatched target run. At site B also lp of the source answer in the source language (to separate concept from language).
  - **Band and statistic.** Band B = layers l where mean_pairs Delta_same(l) >= 0.5 x max_l mean_pairs Delta_same(l), computed from `same` only, before looking at `cross`. Primary statistic I = sum_{l in B} mean_p Delta_cross(l) / sum_{l in B} mean_p Delta_same(l), EN->ES, confirmatory subset; 95% CI by cluster bootstrap over source entities (2000 resamples). Secondary: I(l) per layer; ES->EN; all pairs; greedy-decode flip rate at l* = argmax mean Delta_same; covariate model of per-pair ratio on source token Jaccard, span difference and relation (predicted ratio at Jaccard = 0).
  - **Gates and decision rules (per model).** P1 (gate): `same` flip rate >= 60% at l* and mean Delta_same(l*) >= 2 nats; if it fails, the fallback patches the last min(|s|,|t|) subject positions aligned from the end; if that fails too, E2 is uninformative for that model (a method failure, no claim). P2 SHARED: I lower 95% bound >= 0.7. P3 LANGUAGE-SPECIFIC: 90% upper bound <= 0.3. P4 PARTIAL: anything else (report I and I(l)). P5 LEXICAL: the predicted ratio at Jaccard = 0 is lower than at Jaccard = 1 by more than 0.2 => interchange is partly lexical. P6 METRIC VALIDITY: Spearman between per-fact interchange score and the per-fact A13a NC increment (facts in both sets) > 0.2; otherwise tracing-profile similarity has no validity as a sharing measure (reportable).
  - **Size.** Target >= 120 pairs from >= 40 distinct token-disjoint source entities per model; floor 25 source entities. Below the floor the result is reported as underpowered and no SHARED/LANGUAGE-SPECIFIC claim is made for that model.
  - **Sanity tests before the real run (unit tests).** Patching the target with its own clean activation gives Delta = 0 exactly; a same-language swap of single-token subjects at layer 0 gives a large Delta; a 10-pair debugging pilot is run, excluded from the analysis, and no thresholds are changed after it.
  - **Direction prior (stated now).** The literature (Dumas et al.; Fierro et al.; Lindsey et al.) suggests P2 or P4 for EN-ES at middle layers; no prediction is made for Gemma versus Llama beyond that. Hindi (EN->HI with an HI->HI ceiling, facts correct in EN and HI) is exploratory only. Reported whatever the result; not edited afterwards. Code: `code/interchange.py` (to be written; the unit tests are part of it).

- **A13c (2026-10-24, after the E2 fact filter with the A13 templates and before any E2 interchange data exists).** The A13 templates passed the >= 50% tuning check for every relation in Llama (198 facts kept) but failed it for three of five relations in Gemma-2-2B (a base model that continues with descriptive text: capital_country ES 0.17, capital_of EN 0.13, landmark_country ES 0.40; only 35 facts kept). The check is a template-quality gate, not a result; for Gemma only, templates are now selected per relation and language from the fixed candidate lists below by accuracy on the 30 tuning facts (ties go to the earlier candidate; the A13 template is the first candidate). Llama keeps the A13 templates. A relation still below 50% in a language after selection is dropped, as in A13.
  - capital_of EN: "The capital of {s} is" | "The capital of {s} is the city of" | "The capital city of {s} is called"; ES: "La capital de {s} es la ciudad de" | "La capital de {s} se llama".
  - capital_country EN: "The country whose capital is {s} is" | "The country whose capital is {s} is called" | "The country with the capital city {s} is called"; ES: "El país cuya capital es {s} es" | "El país cuya capital es {s} se llama".
  - landmark_city EN: "The city where {s} is located is" | "The city where {s} is located is called" | "You can visit {s} in the city of"; ES: "La ciudad donde está {s} es" | "Se puede visitar {s} en la ciudad de" | "La ciudad donde está {s} se llama".
  - landmark_country and city_country EN: "The country where {s} is located is" | "The country where {s} is located is called" | "You can visit {s} in the country of"; ES: "El país donde está {s} es" | "El país donde está {s} se llama" | "Se puede visitar {s} en el país de".
  - The Gemma facts file produced with the A13 templates is kept as data/v2/e2_facts_gemma2_A13templates.json and the run directory analysis/run_20261024_e2filter_gemma2 stays on disk. Not edited afterwards.

- **A14 (2026-10-25, before the rerun; CPU only, no new model run).** The primary overlap tables (A3, 1000 permutations) have a p-value floor of 1/1001 = 0.001. We rerun `code/circuit_overlap.py --perm 10000` on the same trace runs, seed (20261012) and fact sets as the "all relations" overlap runs (Llama: analysis/run_20261019_p3overlap_v2all_10191057; Gemma: analysis/run_20261020_p3overlap_v2all_10200339), tags `p3overlap_v2all_perm10k_{llama,gemma2}`. Nothing else changes (bootstrap 2000). Expectation: observed medians and CIs identical (same data), z and p within permutation noise (|dz| < 0.3), no EN-ES verdict changes. If any pre-registered verdict changes, the 10,000-draw value is reported and the change is flagged in the paper. The 1000-draw results stay on disk and in the manifest; the paper table switches to 10,000 draws.

- **A14a (2026-10-25, after the A14 rerun).** A14 said the paper table would switch to 10,000 draws. Because the paper's numbers come from several runs and are quoted in many places, the paper keeps the 1000-draw tables (as pre-registered in A3) and reports the 10,000-draw rerun as a sensitivity check (section 6, manifest keys `pk_*`). Outcome against the A14 expectation: observed medians identical; |dz| <= 0.21 for the MLP-profile cosine, larger for IoU top-5 (up to 3.9, unstable null SD); no EN-ES verdict changed; two secondary tests near p = 0.05 changed side (Gemma no-copyable EN-ES Spearman, Gemma event_year EN-HI). Runs: analysis/run_20261025_p3overlap_{v2,v2all}_perm10k_{llama,gemma2}.

- **A15 (2026-10-25, after A13 results and before any Hindi E2 template, filter or interchange data exists; GPU).** Confirmatory English-Hindi (EN-HI) subject interchange. A13 left Hindi as exploratory and tested only EN-ES. Seen before writing this: the A13 EN-ES results (Llama and Gemma both P2 SHARED, site A) and A13 itself; no Hindi interchange number has been computed. Same question, same method, same decision rules as A13, with Hindi in place of Spanish. Hindi is written in a different script, so subject token overlap is zero by construction (every pair is confirmatory; the Jaccard covariate and P5 do not apply).
  - **Models and facts.** Llama-3.2-3B-Instruct and Gemma-2-2B. Candidate pools as A13 (capital_of, capital_country, landmark_city, landmark_country, city_country; the same `data/v2` candidate files, which already contain Hindi subjects and objects). The first 30 candidates per relation by id are tuning facts and are excluded from evaluation, as in A13. A candidate is KEPT for a model if the model answers correctly in BOTH EN and HI (strict prefix match as A6) with the prefix templates; Spanish is not required.
  - **Templates.** EN: the template each model used in A13 (the `templates` entry of `data/v2/e2_facts_<model>.json`, English side). HI: for each relation the first candidate below is the default; per relation the candidate with the highest accuracy on the 30 tuning facts is used (ties go to the earlier one), and a relation whose best accuracy is below 50% in either language is dropped (the A13c rule, applied to both models).
    - capital_of: "यह सर्वविदित है कि {s} की राजधानी है" / "{s} देश की राजधानी का नाम है".
    - capital_country: "वह देश जिसकी राजधानी {s} है, उसका नाम है" / "जिस देश की राजधानी {s} है, वह देश है".
    - landmark_city: "वह शहर जहाँ {s} स्थित है, उसका नाम है" / "{s} जिस शहर में स्थित है, वह शहर है".
    - landmark_country and city_country: "वह देश जहाँ {s} स्थित है, उसका नाम है" / "{s} जिस देश में स्थित है, वह देश है".
  - **Pairs, intervention, readout, band, statistic.** Exactly as A13 with ES replaced by HI: ordered (target t, source s), same relation, different first answer token in the target language, two sources per target by seed 20261024; `blocks.{l}.hook_resid_post` at the last subject token (site A, primary) or the final prompt token (site B); conditions same / cross / self / ctrl; teacher-forced full-answer log-probability; band from `same` only; I = sum over the band of mean cross effect divided by the same over the same band. Primary direction EN->HI (source computed in English, patched into the Hindi target run), with HI->HI as the ceiling (`same`); HI->EN is secondary. 95% CI by cluster bootstrap over source entities (2000). To bound run time, the pair list is capped at the first 200 pairs of the seeded list per model; the first 10 distinct pairs are the debugging pilot and are excluded, as in A13.
  - **Gates and decision rules (per model).** Identical to A13: P1 gate (`same` flip >= 60% at l* and mean Delta_same(l*) >= 2 nats; if it fails the model is uninformative for Hindi, no claim); P2 SHARED if the lower 95% bound of I >= 0.7; P3 LANGUAGE-SPECIFIC if the 90% upper bound <= 0.3; otherwise P4 PARTIAL. Floor: at least 25 distinct source entities, otherwise underpowered and no claim. P5 (lexical) and P6 (metric validity against the A13a increments) are not applicable to Hindi (no token overlap; A13a has no Hindi independent-noise traces) and are reported as such.
  - **Predictions (stated now, uncertain).** Llama: I (EN->HI, site A) between 0.4 and 1.0, i.e. P4 PARTIAL or P2, below its EN->ES value (1.17), because Hindi tracing alignment was weaker in every earlier analysis. Gemma-2-2B: likely few kept facts or a failed gate (a small base model with weak Hindi); if it passes the gate, P4. If the Hindi `cross` effect were near zero while EN->ES stayed near 1, that would support a language-specific route for Hindi (P3).
  - **Code.** `code/e2_filter_hi.py` (filter), `code/interchange.py --langs en,hi`, `code/e2_analysis.py --langs en,hi`. Outputs: `data/v2/e2h_facts_<model>.json`, run dirs tagged `e2hfilter`, `e2hfull`, `e2h_analysis`. Reported whatever the result; not edited afterwards.

- **A16 (2026-10-26, after the A15 Hindi results, before any A16 filter or interchange data exists; GPU).** Confirmatory re-test of Hindi-to-English (HI->EN) subject interchange with a larger Hindi fact set. A15 was underpowered (23 Llama and 12 Gemma source entities against a floor of 25) and made no claim.
  - **Seen before writing this (disclosure).** The A15 filter tables and generations, and the A15 interchange results including the HI->EN site-A ratios (Llama 0.74 [0.67, 0.81], Gemma 0.74 [0.69, 0.82], both PARTIAL, descriptive only). So the A16 predictions below are informed by A15, and A16 is a replication with more facts, not an independent first test. The reported A16 result will also be given on the subset of facts that are NOT in the A15 kept set, as a clean check.
  - **Why the design changes (from what A15 showed).**
    - The A15 Hindi templates ended in "... है" or "... का नाम है", and Llama often continued with a question mark and a line break (e.g. "?\nउत्तर: ईरान"), which the strict prefix rule (A6) counts as wrong although the answer was right. This, not missing knowledge, removed many facts. A16 adds two question/answer-format Hindi templates per relation ("प्रश्न: ... ?\nउत्तर:") in which the answer follows directly.
    - The A15 per-relation threshold (template accuracy >= 50% in both languages on the 30 tuning facts) kept one relation per model. It was a template-validity guard, but each kept fact is already checked individually (correct in EN and in HI), so A16 drops the threshold and instead requires >= 8 kept evaluation facts for a relation to be included.
    - EN->HI (target prompt in Hindi) is dropped as a primary direction: Hindi subjects span a median of 5 (Llama) and 8 (Gemma) tokens (4.3), and patching only the last subject token cannot transfer an entity into a Hindi prompt (A15 same-language flip 0.12 Llama, 0.36 Gemma, gate P1 failed). It is a limit of the single-token patch, not evidence about sharing, and is not tested again here (a span-wise patch would need its own pre-registration).
  - **Models and facts.** Llama-3.2-3B-Instruct and Gemma-2-2B, same candidate pools as A13/A15 (`data/v2` pools; first 30 candidates per relation by id are tuning facts, excluded). English templates: the A13 ones (`e2_facts_<model>.json`, English side). Hindi template per relation: the candidate with the highest accuracy on the 30 tuning facts among 4 fixed candidates (the two A15 ones and the two Q/A ones in `code/e2_filter_hi.py`, `HI_CANDS16`; ties go to the earlier one). A fact is kept if the model's greedy 12-token continuation passes the strict prefix match (A6) in BOTH English and Hindi. A relation is included if it has >= 8 kept evaluation facts. Output `data/v2/e2h2_facts_<model>.json`.
  - **Pairs, intervention, readout.** As A13 with direction HI->EN only (source prompt in Hindi, target prompt in English; `same` = English source into English target is the ceiling). Pairs: for each target fact, 4 sources of the same relation whose first English answer token differs from the target's (the source list is shuffled with seed 20261024, the full pair list is then shuffled with the same seed and cut at 300 pairs, so relations are mixed); `blocks.{l}.hook_resid_post` at the last subject token (site A, primary) or the final prompt token (site B, secondary); conditions same / cross / self / ctrl; teacher-forced full-answer log-probability. First 10 distinct pairs are the pilot and excluded. Confirmatory pairs: source Jaccard 0 (zero by construction for Hindi).
  - **Statistic, gates, decision rules (per model, site A).** Exactly as A13/A15: band from `same` only (layers with mean Delta_same >= 0.5 of its maximum), l* its argmax; **P1 gate**: same-language flip >= 60% at l* and mean Delta_same(l*) >= 2 nats, otherwise the model is uninformative and no claim; I = sum over the band of mean Delta_cross divided by the same sum for `same`; 95% CI and 90% CI by cluster bootstrap over source entities (2000 draws). **P2 SHARED** if the lower 95% bound of I >= 0.7; **P3 LANGUAGE-SPECIFIC** if the upper 90% bound <= 0.3; otherwise **P4 PARTIAL**. **Floor:** at least 25 distinct source entities in the confirmatory set, otherwise underpowered and no claim. Site B is secondary: its ratio is reported as interpretable only if the cross-language flip rate at its l* is >= 0.2 (in A15 the cross flip was 0.00 and the ratios were not meaningful).
  - **Descriptive extras (no verdicts).** Ratio per relation (>= 5 confirmatory pairs); the HI->EN site-A ratio next to the ES->EN ratio from A13 (Llama 0.90 [0.86, 0.94], Gemma 0.94 [0.91, 0.97]) by their intervals only; the result on the facts that are not in the A15 kept set; flip rates at l*.
  - **Predictions (stated now).** (1) Kept evaluation facts: Llama >= 80 and Gemma >= 60 across >= 2 relations each; if a model has fewer than 25 source entities, it is underpowered and reported as such. (2) Site A, both models: P4 PARTIAL with I between 0.55 and 0.90 (A15: 0.74). P2 SHARED (lower bound >= 0.7) with probability about 20% per model; P3 below 3%. (3) HI->EN ratio lower than ES->EN in both models (cross flip below the same-language flip). (4) The result on the facts outside the A15 set is within 0.15 of the full-set ratio.
  - **If the floor is missed again** the Hindi question is reported as open with the numbers; no third round is planned without new facts or a model with better Hindi recall.
  - **Code.** `code/e2_filter_hi.py --a16`, `code/interchange.py --langs en,hi --directions hi2en --k-src 4 --shuffle-pairs --max-pairs 300`, `code/e2_analysis.py --langs en,hi` (now also writes `floor_25_sources_ok`, `ratio_interpretable_cross_flip_ge_0.2` and `by_relation_descriptive`), driven by `scripts_night_a16.sh`; run dirs `*_e2h2filter_*`, `*_e2h2full_A_*`, `*_e2h2_analysis_A16`. Expected run time about 2 hours on one 8 GB GPU. Reported whatever the result; not edited afterwards.

- **A17 (registered 2026-10-26, before any leakage-excluded number was computed): answer-in-subject leakage re-analysis (CPU only, stored artifacts).**
  - **Why.** An external audit of the release pack noted that for some facts the answer string is contained in the subject (Mexico City -> Mexico, Panama City -> Panama, Tokyo Skytree -> Tokyo). For these the answer is partly derivable from the subject inside each language, which none of the A13b covariates (name overlap J, answer identity AI, span, magnitude) controls. The audit reported 16/93 (Llama) and 25/58 (Gemma) such facts in the overlap sets; our rule below finds fewer (the audit's exact rule is not known to us), and we pre-register ours, not theirs.
  - **Rule (fixed, `code/leakage.py`).** A fact is leaky if, in any of its languages, the accent- and case-folded object (or `target`, or an alternative answer) of at least 3 characters is a substring of the folded subject. A fact is dropped from an analysis if it is leaky in any language. No other rule is tried.
  - **Analyses (each is the existing, unchanged analysis with leaky facts removed; flag `--exclude-leak`).** (a) A13a G1 and the S sets (`code/seeds_analysis.py`, 10,000 permutations, same seed). (b) A13b target prediction D, informative and all-non-IP (`code/seeds_covariates.py`). (c) A14 overlap, primary_valid and stratified sets (`code/circuit_overlap.py`, 10,000 permutations). (d) A13 interchange EN<->ES and (e) A16 interchange HI->EN: pairs where the target or the source fact is leaky are dropped (`code/e2_analysis.py`). Also reported for (d) and (e): whether the 95% lower bound is >= 0.60, 0.65, 0.70, 0.75 (threshold sensitivity; the verdict threshold stays 0.7).
  - **Decision rule.** A headline claim is *robust to leakage* if its verdict category (present/unresolved/absent for A13a/A13b; P2/P3/P4 for A13/A16; z > 3 and p < 1e-3 for the EN-ES overlap) is unchanged. If the category changes, the paper reports the leakage-excluded value as the primary one and the original as a sensitivity. No facts, thresholds or covariates are tuned. Counts of dropped facts are printed per set.
  - **Predictions (stated now).** (1) G1 SURVIVES in both models. (2) A13b Llama target stays > 0 with the lower bound > 0; Gemma stays unresolved or present. (3) EN-ES overlap stays above the null in both models (z > 3, mlp_cosine). (4) A13 EN-ES stays P2 SHARED in all four cells. (5) A16 Llama stays P4 PARTIAL. (6) A16 Gemma: no prediction on the direction; we expect a 50% chance that SHARED (lower bound 0.71) survives, because dropping 6 of 55 facts moves the interval by roughly its own width.
  - **Not covered.** The A11 median-split versus A13b disagreement is a text reconciliation (no new test). Native-speaker audit of facts and EN->HI span-wise patching (A18 if run) are separate.
  - **Code.** `code/leakage.py`, `--exclude-leak` in the four scripts above, driven by `scripts_a17.sh`; run directories `analysis/run_<date>_*_A17`.

### A17b: bootstrap-clustering sensitivity of the interchange verdicts (CPU, registered 2026-10-26 before running; motivated by an external audit that recomputed the A16 CIs under other clusterings)
- **Why.** `code/e2_analysis.py` resamples *source entities* only, but the same target fact appears in several pairs (k=2 in A13, k=4 in A16), and a fact can be a target in one pair and a source in another. The CI may therefore be too narrow or too wide depending on the clustering, and the P2/P4 label sits within about 0.01-0.03 of the 0.7 line in A16. This is a sensitivity analysis on stored effects; no new model run.
- **Analyses (site A, confirmatory = token-disjoint pairs, same band and l* as A13/A16, leak-excluded and not).** Four 95% CIs for I, 5,000 draws, seed 20261026: (S) source fact (the registered one), (T) target fact, (C) connected components of the pair graph (target and source fact ids as nodes), (W) two-way: fact ids resampled independently as targets and as sources, each pair weighted by count(target) x count(source).
- **Decision rule.** A verdict is *clustering-robust* if the label (lower 95% bound >= 0.7 -> SHARED, else PARTIAL; upper 90% bound <= 0.3 -> LANGUAGE-SPECIFIC) is the same under S, T, C and W. If not robust, the paper reports the estimate with the four CIs and does NOT use the label. No registered verdict is changed retroactively; this adds a robustness column.
- **Predictions (stated now).** (1) EN<->ES SHARED in A13 stays SHARED under all four (lower bounds were >= 0.855). (2) The A16 point estimates do not move (0.74-0.76; 0.80 leak-excluded Gemma). (3) At least one of the four A16 cells (Llama, Gemma; plain, leak-excluded) changes label under some clustering, because their lower bounds sit within 0.03 of 0.7.
- **Code.** `code/cluster_sensitivity.py`; output `analysis/run_<date>_cluster_A17b/results.json`.

### A18: English->Hindi span-wise subject interchange (registered 2026-10-26 before any span-wise data; GPU, about 1 h)
- **Gap.** A15 could not test EN->HI: the same-language Hindi ceiling failed P1 (same-language flip 0.12 Llama, 0.36 Gemma) because a single-last-token swap does not carry a Hindi subject that spans 5 (Llama) to 8 (Gemma) tokens. A16 therefore reported HI->EN only (I 0.74 Llama, 0.76-0.80 Gemma), so the Hindi "about three-quarters shared" estimate is one-directional. A18 patches the whole subject span so that both directions have a working ceiling.
- **Design (unchanged from A13/A16 except the patch).** Same facts (`data/v2/e2h2_facts_{llama,gemma2}.json`), seed 20261024, `--k-src 4 --shuffle-pairs --max-pairs 200`, site A only. Variants (`code/interchange.py --span`): **end** = the last k = min(n_target, n_source) subject positions of the target are replaced by the last k subject-position states of the source (end-aligned), earlier target positions are left alone; **bcast** = every target subject position receives the source's last-subject-token state. The four conditions keep their meaning: same (source in target language), cross (source in the other language), self (own fact in the other language), ctrl (other-language source at the token before its subject, broadcast over the same target positions). Readout, band, l*, ratio I, source-entity cluster bootstrap, confirmatory = Jaccard-0 source pairs, floor of 25 source entities, P1/P2/P3/P4 rules: exactly as A13/A16. `end` is run in both directions (en2hi and hi2en, so the comparison with HI->EN is like for like); `bcast` in en2hi only.
- **Primary variant and selection.** `end` is primary. `bcast` becomes primary for a given model only if `end` fails the P1 gate for en2hi in that model and `bcast` passes it. The choice depends only on the same-language ceiling, never on the cross ratio. Both variants are reported either way. Leak-excluded companion (`--exclude-leak`) is run as in A17.
- **Interpretation rules.** (i) EN->HI is *testable* only if the gate passes and the floor of 25 source entities holds. (ii) I is reported as an estimate with its 95% CI and the A17b clustering sensitivity; no SHARED/PARTIAL label is used as a headline unless it is the same under the four clusterings (A17b). (iii) The Hindi estimate is called *direction-symmetric* if |I(en2hi) - I(hi2en)| <= 0.15 (both from the same `end` variant), *direction-inflated in HI->EN* if I(en2hi) < I(hi2en) - 0.15, and *deflated* if I(en2hi) > I(hi2en) + 0.15. (iv) Span-wise HI->EN is compared with the A16 last-token estimate to show whether the patch choice moves the number.
- **Predictions (stated now).** (1) Gate: `end` passes P1 for en2hi in Llama (60%) and in Gemma (55%); if it fails, `bcast` passes (50%). (2) If testable, I(en2hi) lies in 0.5-0.9 in both models. (3) Symmetry (rule iii) holds in about 60% of the model cells. (4) Span-wise HI->EN stays within 0.15 of A16 (0.74 Llama, 0.76 Gemma). (5) Site B is not run.
- **Caveats stated in advance.** End alignment is arbitrary when the two prompts have different span lengths; `bcast` is a cruder carrier. A span-wise patch writes more of the context, so a high same-language ceiling is a necessary check, not a guarantee that cross is a pure entity swap; ctrl and self guard against that.
- **Code.** `code/interchange.py --span {last,end,bcast}`, `code/e2_analysis.py --sites A`, `scripts_a18.sh`; run directories `analysis/run_<date>_e2h3{end,bcast}_A_<model>`.

### A18b: clustering sensitivity of the span-wise EN<->HI verdicts (CPU, registered 2026-10-28 before running; stored effects only)
- Same four clusterings, 5,000 draws, seed 20261026, and the same label rule as A17b (`code/cluster_sensitivity.py --set a18b`), on the `end` runs `e2h3end_A_{llama,gemma2}`, both directions, plain and leak-excluded (8 cells).
- Rule: a label is used as a headline only if it is the same under S, T, C and W; otherwise the estimate and the four CIs are reported without a label. No A18 label is changed retroactively.
- Predictions (stated now): (1) point estimates are those of A18 (0.81-0.93). (2) At least 6 of the 8 cells are SHARED under all four clusterings. (3) The cells most likely to fail are Gemma HI->EN (lower bound 0.76 under S) and Llama EN->HI with the 0.80 lower bound.

### A19: base-model replication (Llama-3.2-3B base vs Instruct; GPU, registered 2026-10-28 before any base-model data; roadmap item A1 / E4)
- **Gap.** Everything so far is on Llama-3.2-3B-*Instruct* and Gemma-2-2B (a base model). The Llama result could be an effect of instruction tuning. A19 repeats the interchange on `meta-llama/Llama-3.2-3B` (base, same architecture and tokenizer, local dir `models/llama-3.2-3b-base`, model key `llama_base`).
- **Pipeline (unchanged from A13/A16/A18 except the pair cap).** (1) Fact filter on the base model with the same candidate pools, templates and tuning rule: `code/e2_filter.py --model llama_base` (EN/ES, template accuracy >= 50% on the 30 tuning facts per relation, facts correct in EN and ES) and `code/e2_filter_hi.py --model llama_base --a16` (Hindi, A16 rules). (2) EN<->ES interchange, `code/interchange.py --model llama_base --langs en,es --k-src 4 --shuffle-pairs --max-pairs 200 --tag e2b_A` (A13 used k=2 and no cap; k=4, shuffling and the 200-pair cap follow A16/A18 and are used here so the run fits in about 1 h; this is a stated deviation). (3) EN<->HI interchange with the span-wise `end` patch (A18 primary), `--langs en,hi --facts data/v2/e2h2_facts_llama_base.json --span end --directions en2hi,hi2en --k-src 4 --shuffle-pairs --max-pairs 200 --tag e2hb_A`. Seed 20261024. Site A only is the primary endpoint; the EN-ES run also stores site B (descriptive). Analysis: `code/e2_analysis.py --llama-label llama_base --models llama_base` (plain and `--exclude-leak`), then `code/cluster_sensitivity.py --set a19b`.
- **Statistic, gates, floors.** Exactly as A13/A18: interchange ratio I at site A, source-entity cluster bootstrap, confirmatory = Jaccard-0 source pairs, P1 gate (same-language flip >= 0.6 and Delta >= 2 nats), SHARED if lower 95% >= 0.7, LANGUAGE-SPECIFIC if upper 90% <= 0.3, else PARTIAL, floor 25 source entities. P5 (lexical covariate) and P6 (metric validity) are not evaluated for the base model (no A13a seeds there).
- **Question and decision rule.** *Does the instruct result depend on instruction tuning?* The base model *replicates* the instruct result in a cell if the cell passes the gate and |I_base - I_instruct| <= 0.15, with the instruct values from A13 (EN->ES 1.173, ES->EN 0.905, leak-free values in A17) and A18 `end` (EN->HI 0.860, HI->EN 0.868). *Different* if the gate passes and the difference exceeds 0.15. *Untestable* if the gate or the floor fails (then no claim for that cell). Labels are reported with the A19b clustering sensitivity and used as headlines only if clustering-robust.
- **Gate G-A (from `docs/ROADMAP_Q1.md`).** "EN-ES shared holds on base Llama": met iff both EN<->ES cells are gate-passing and SHARED under the registered clustering. This is a necessary condition for continuing to Phase B; its failure would change the paper to a "when is it shared" claim.
- **Predictions (stated now).** (1) The base model keeps at least 40 EN/ES facts and at least 25 Hindi facts; the Hindi count is the uncertain one (60% to reach 25). (2) EN<->ES passes the gate and is SHARED in both directions (80%). (3) EN<->HI passes the gate in both directions (60%); if testable, I in 0.6-1.0 and replication within 0.15 in at least one direction (65%). (4) Across all testable cells at least 3 of 4 replicate (60%). (5) Base I is not systematically higher than instruct I (no direction predicted).
- **Caveats stated in advance.** Different model weights can change which facts survive the filter, so the fact sets overlap but are not identical to the instruct sets; I comparisons are between cells, not paired by fact. The base model has no chat tuning, so the plain prefix templates used in all earlier runs are, if anything, a better fit for it. The 200-pair cap lowers power vs A13.
- **Code.** `code/common.py` (key `llama_base`), `code/e2_filter.py`, `code/e2_filter_hi.py`, `code/interchange.py`, `code/e2_analysis.py --llama-label`, `code/cluster_sensitivity.py --set`; driver `scripts_a19.sh`; run dirs `analysis/run_<date>_{e2filter,e2hfilter,e2b_A,e2hb_A,e2b_analysis_A19,...}_llama_base`. Reported whatever the result; not edited afterwards.

### A18b + A19: clustering sensitivity of A18, and the base-model replication E4 (registered 2026-10-28 before any base-model data; A18b is CPU on stored effects, A19 is GPU, about 2.5 h)
- **A18b.** `code/cluster_sensitivity.py --set a18b`: the A17b rule (four clusterings S/T/C/W, 5,000 draws, seed 20261026, confirmatory pairs, plain and leak-free) applied to the A18 `end` runs (en2hi, hi2en, both models). The label (SHARED if lower 95% >= 0.7) is a headline only if identical under all four. Prediction: Llama and Gemma EN->HI and HI->EN stay SHARED under all four in >= 6 of the 8 plain/leak-free cells (A18 lower bounds were 0.76-0.90).
- **A19 question.** Does the EN/ES and EN/HI result depend on instruction tuning? Model: `meta-llama/Llama-3.2-3B` (base, key `llama_base`), same architecture and tokenizer as the instruct model used so far.
- **A19 design.** (1) EN/ES: `code/e2_filter.py --model llama_base` (A13 fixed templates, same candidate pools, same tuning/eval split rule, same >= 50% template threshold) -> `data/v2/e2_facts_llama_base.json`; `code/interchange.py --langs en,es --span last` with `--k-src 2 --shuffle-pairs --max-pairs 200` (a cap, not the 396 pairs of A13, to fit the time budget; shuffled so relations mix) -> tag `e2b_A`. (2) EN/HI: `code/e2_filter_hi.py --a16 --model llama_base` -> `data/v2/e2h2_facts_llama_base.json`; `code/interchange.py --langs en,hi --span end --directions en2hi,hi2en --k-src 4 --shuffle-pairs --max-pairs 200` -> tag `e2hb_A`. (3) Analysis: `code/e2_analysis.py --llama-label llama_base --models llama_base --sites A` (P5/P6 skipped for the base model: no A13a seeds exist for it), plus the leak-free companion and `cluster_sensitivity.py --set a19b`. Site A only. Everything else (conditions, band, l*, ratio I, source-entity cluster bootstrap, Jaccard-0 confirmatory pairs, floor of 25 source entities, P1 gate, P2/P3/P4 labels) exactly as A13/A16/A18.
- **Decision rules.** (i) A cell is testable only if the P1 gate passes and the floor of 25 source entities holds; otherwise no claim. (ii) *Instruct/base-invariant* for a cell if |I_base - I_instruct| <= 0.15 with I_instruct from the matching earlier analysis (EN->ES 1.17, ES->EN 0.90 from A13; EN->HI 0.86 and HI->EN 0.87 from A18 `end`); the fact sets differ because the base model keeps its own correctly-answered facts, so this is a comparison of estimates, not a paired test. (iii) SHARED is a headline label only if clustering-robust (A18b/A17b rule). (iv) If the base model fails the gate for Hindi (weaker multilingual ability is possible), Hindi is reported as untestable and the EN/ES result alone is claimed.
- **Predictions (stated now).** (1) Base keeps >= 100 EN/ES facts (80%) and >= 40 Hindi facts (60%). (2) EN/ES gate passes in both directions (90%) and the label is SHARED (80%). (3) Hindi gate passes in both directions (60%); if so I lies in 0.6-1.0. (4) Invariance (rule ii) holds in >= 3 of the 4 cells that are testable (65%).
- **Caveats stated in advance.** The 200-pair cap makes the EN/ES CI wider than A13's. One base model, one scale. No unit test beyond A13/A18's; the changed code paths are only the model key and `--llama-label`.
- **Code.** `code/common.py` (model key `llama_base`, weights in `models/llama-3.2-3b-base`), `code/e2_filter.py`, `code/e2_filter_hi.py`, `code/interchange.py`, `code/e2_analysis.py --llama-label`, `code/cluster_sensitivity.py --set`, `scripts_a19.sh`. Reported whatever the result.

### A19: Llama-3.2-3B *base* replicates the instruct interchange result? (registered 2026-10-28 before any base-model data; GPU, about 2.5-3 h)
- **Gap.** Every result so far is on Llama-3.2-3B-**Instruct** and Gemma-2-2B (a base model). A reviewer can say the shared state is a product of instruction tuning. Llama-3.2-3B (base, `meta-llama/Llama-3.2-3B`, same architecture and tokenizer, weights in `models/llama-3.2-3b-base`) is the paired control. Model key `llama_base` (`code/common.py`).
- **Design.** Same recipe as the instruct model, with a pair cap to fit the time: (a) fact filter `code/e2_filter.py --model llama_base` (same candidate pools and EN/ES templates as A13, strict match, 30 tuning facts per relation, template OK if >= 50%), and Hindi filter `code/e2_filter_hi.py --model llama_base --a16` (A16 rules). (b) EN<->ES: `code/interchange.py --model llama_base --langs en,es --k-src 2 --shuffle-pairs --max-pairs 200 --tag e2b_A` (A13 last-token patch, site A and B recorded). (c) EN<->HI: `--langs en,hi --facts data/v2/e2h2_facts_llama_base.json --span end --directions en2hi,hi2en --k-src 4 --shuffle-pairs --max-pairs 200 --tag e2hb_A` (the A18 `end` design). Analysis: `code/e2_analysis.py --models llama_base --llama-label llama_base`, plus `--exclude-leak` and `code/cluster_sensitivity.py --set a19b` (A17b clusterings). Gates, bands, l*, I, source-entity bootstrap, confirmatory = Jaccard 0, floor of 25 source entities, P1-P4 rules: exactly as A13/A16/A18. P5 and P6 are not run for the base model (no matched A13a seeds).
- **Deviation stated.** The pair cap (200, shuffled) differs from the instruct EN-ES run (all ~396 pairs); the instruct EN-ES numbers are not recomputed. Comparison uses the point estimate and CI, not equal n.
- **Decision rules.** (i) A cell is testable only if P1 passes and >= 25 source entities. (ii) *Replicates* (per language pair): base I within 0.15 of the instruct I in both directions and the same P1/P2 label, labelled only if it is the same under all four clusterings. (iii) *Differs* if |I_base - I_instruct| > 0.15 in any direction. (iv) If a gate fails the cell is reported untestable, no claim.
- **Predictions (stated now).** (1) Base passes P1 for EN-ES in both directions (80%). (2) EN-ES I >= 0.7 lower bound (P2 SHARED) in both directions (70%); |dI| <= 0.15 versus instruct (A13: en2es 1.17, es2en 0.90) in both (55%). (3) Base passes the span-wise Hindi gate in both directions (55%); if testable, I in 0.6-0.95 and within 0.15 of the instruct span-wise values (0.86/0.87) (50%). (4) Kept facts: >= 100 EN-ES facts (75%), >= 40 Hindi (50%).
- **Caveats.** One extra model, same family: it tests the instruct-vs-base confound only, not model-family generality. The base model may need a different filter pass rate, so the fact set differs from instruct's. A 200-pair cap lowers precision.
- **Code.** `code/common.py` (`llama_base`), flags above; driver `scripts_a19.sh`; run dirs `analysis/run_<date>_{e2bfilter,e2hbfilter,e2b_A,e2hb_A}_llama_base`, analyses tagged `e2b_analysis_A19`, `e2hb_analysis_A19` (and `_leakfree`). Reported whatever the result.

### A20: more Llama edit facts for the traced-layer vs mid-layer edit comparison (registered 2026-10-29 before any of these edits; GPU, up to about 1.7 h, queued behind A19)
- **Gap.** The Llama edit comparison rests on 10 sweep facts plus 20 F7 facts (30 total); both external reviews called it too small to separate layer conditions. A20 adds 20 further Llama-3.2-3B-Instruct facts, in the deterministic stratified order of `select_edit_facts` (positions 31-50), with the same F7 design: ROME-style rank-1 edit of the down-projection in English only, at (a) the fact's English traced layer, (b) L//2 = layer 14, (c) the A9-selected layer 21 (called "random" in earlier tables; it is a fixed selected layer, see EXPERIMENT_LOG 2026-10-25). Command: `knowledge_edit_v2.py --model llama --trace-run analysis/run_20261015_p3trace_v2_llama --n-edit 50 --skip-first 30 --layers 21 --add-traced --tag p4a20`. Endpoints and metrics as F7: English success, Spanish and Hindi transfer ratio (median), specificity (max |dP| over 20 control facts), perplexity ratio.
- **Analysis.** Pool F7 and A20 (50 facts at most) and report per-condition medians with bootstrap-over-facts 95% CIs; descriptive, no new hypothesis test. Existing F7 numbers are not changed.
- **Predictions (stated now).** (1) English success: L//2 >= traced in the pooled set (as in F7: 0.95 vs 0.80). (2) Median Spanish transfer: traced and L//2 within 0.15 of each other; layer 21 near 0 (as in F7). (3) Median Hindi transfer < 0.15 in all three conditions. (4) Specificity loss at L//2 exceeds that at the traced layer. (5) Perplexity ratio < 1.01 in all conditions.
- **Time cap.** The run is cut off at 06:30 on the day it starts (`timeout`); an unfinished run is resumed later with `--resume` and reported only when complete or with its actual n.
- **Caveats.** One model; facts are the same hand-written geography/event set; the Llama-vs-Gemma edit-layer contrast stays unsupported (instruct vs base, different n).

### A21: English-paraphrase control for the edit transfer result (registered 2026-10-29 before any of these edits; GPU, about 3 h)
- **Gap.** Spanish and Hindi transfer of a rank-1 English edit is read as a cross-language effect. But the edit is fitted on one English phrasing (with random prefixes), so low transfer could just mean the edit does not generalise to any new wording. Without an English-paraphrase control, "does not transfer to Spanish/Hindi" cannot be separated from "does not transfer to anything but the edited sentence". Both external reviews asked for this control.
- **Design.** The same Llama-3.2-3B-Instruct edit procedure as F7/A20 (ROME-style rank-1 update of the down-projection, fitted on English `prompts[0]` with prefixes, same seeds and replacement answers), on the same 30 facts as F7 and A20 (edit-fact positions 11-40 of `select_edit_facts`; flags `--n-edit 40 --skip-first 10`). Conditions: the fact's traced layer and layer 14 (`--layers 14 --add-traced`); layer 21 is dropped because its Spanish transfer is 0. New flag `--para` also scores the replacement answer under the two other hand-written phrasings (`prompts[1:]`) of each fact, in English, Spanish and Hindi. Command in `scripts_a21.sh`. Analysis `code/para_analysis.py`: transfer = dP / dP(English `prompts[0]`); facts with English dP <= 0.1 are dropped (as in A20); median with bootstrap-over-facts 95% CI; unseen phrasings are averaged within a language.
- **Sanity check.** The re-fitted edit must reproduce the stored F7/A20 numbers: the median absolute difference of English dP and of Spanish dP per (fact, layer) against the earlier records must be < 0.02. If not, the control is reported but flagged as non-reproducing.
- **Primary quantity.** D = English-paraphrase transfer minus Spanish (`prompts[0]`) transfer, per fact; median and CI, per layer condition. Same for Hindi.
- **Interpretation rules.** CI of D above 0.15: Spanish transfer is lower than generalisation to new English wording, i.e. a cross-language loss beyond wording. D within +-0.15: Spanish transfer is about as good as generalisation to new English wording, so the limit of the edit is wording-generality, not language. D below -0.15: Spanish exceeds English paraphrase transfer (unexpected, reported as such).
- **Predictions (stated now).** (1) Layer 14: English paraphrase transfer median in 0.5-0.95 and D within +-0.15 (60%). (2) Traced layer: D > 0.15 (55%), because that edit sits early, is tied to the subject token whatever the wording, and its Spanish transfer is bimodal across facts. (3) Hindi (`prompts[0]`) transfer is below English paraphrase transfer by more than 0.3 in both conditions (75%). (4) The sanity check passes (85%). (5) Unseen Spanish phrasings transfer within 0.15 of Spanish `prompts[0]` (70%).
- **Time.** About 2 records per fact x 30 facts at about 3 min each = about 3 h. The driver stops at 06:30; an unfinished run resumes with `--resume` (rerun `scripts_a21.sh`) and is reported with its actual n.
- **Caveats.** One model, 30 facts; the paraphrases are mild (the pipeline's three templates share syntax); descriptive, no new hypothesis test.
- **Code.** `code/knowledge_edit_v2.py --para`, `code/para_analysis.py`, `scripts_a21.sh`.

### A22: does the Llama EN-ES increment survive in facts with no shared subject tokens? (registered 2026-10-29 before the computation; CPU, stored traces, minutes)
- **Gap.** Section 4.6 (A11) found that Llama's EN-ES alignment drops below the registered threshold in the low-name-similarity half (centred z = 1.44) using the original single-seed profiles. Section 4.7 (A13b) later fitted a covariate model on independent-noise profiles and found an increment of 0.319 [0.210, 0.428] at zero token overlap (n = 103). The two use different profiles and different methods; the draft never reconciled them. A22 repeats the A11-style split on the A13b profiles so the two are comparable.
- **Design.** `code/split_indnoise.py`: per-fact increment D_i (strata-centred unit profiles, same-relation baseline; the A13a independent-noise profiles, mean over 10 draws) for the informative set of A13b (non-identical-prefix facts, event_year excluded; Llama n = 103, Gemma n = 25), plus the all-non-identical-prefix set. Split by subject-token Jaccard J: J = 0 (no shared subword piece in English and Spanish subject spans) versus J > 0. Per half: mean D and a cluster bootstrap 97.5% interval (clusters as in A13b, 2000 draws, seed 20261023). Plain and leak-free (A17 exclusion list).
- **Decision rule.** The increment is *present* in a half if the lower 97.5% bound is > 0 (same Z as A13b), *absent* if the upper bound is < 0.05, otherwise unresolved. The A11 conclusion "Llama's alignment is largely subject-name overlap" is retained only if the J = 0 half is not present; if it is present the paper replaces that sentence with the A13b statement and treats A11's low-half drop as a noise artefact of the single-seed profiles.
- **Predictions (stated now).** (1) Llama J = 0 half (n about 54): present, mean D within 0.15 of the J > 0 half (65%). (2) Llama J > 0 half: present (90%). (3) Gemma halves are too small (n = 25 informative) and mostly unresolved (70%). (4) Leak-free results keep the same labels (85%).
- **Caveats.** Post hoc relative to A11 (the split was run there first); it uses the same facts, so no new data. Gemma is power-limited. Reported whatever the result; the paper text is changed to match.
- **Code.** `code/split_indnoise.py`; output `analysis/run_<date>_split_indnoise_A22/`.
- **Result (recorded 2026-10-29).** Llama informative set: J = 0 half present (mean D 0.303 [0.192, 0.423], n = 54), J > 0 half present (0.397 [0.291, 0.505], n = 49); leak-free J = 0: 0.312 [0.202, 0.441]. Gemma J = 0 present (0.264 [0.139, 0.402], n = 21); Gemma J > 0 has 4 facts (unresolved). Predictions (1) and (2) held; (3) did not (Gemma J = 0 half is present); (4) held for Llama, not for the tiny Gemma halves. Run dirs `analysis/run_20261029_split_indnoise_A22[_leakfree]`; the paper's Section 4.6 is changed to match.

### A23: does the shared-subject-state result hold outside geography? Non-geography relations (registered 2026-10-29 before any model sees these facts; GPU, about 3-4 h, needs the user's laptop window)
- **Gap.** Every fact so far is geography or events (capital, landmark, city, inventor, event year). The commonest reviewer objection is "geography only". A23 adds four relations with different answer types: official_language (country to language), currency (country to currency), profession (famous person to profession), book_author (book to author). Answers are written differently in EN/ES/HI (no copyable answers). Candidates: `data/v2/facts_candidates_a23.json`, built by `code/build_facts_a23.py` (150: 52, 31, 38, 29). The facts were written by the AI assistant and are **unaudited**; kept facts go to the native-speaker sheet (`code/make_audit_sheet.py`), and corrections would trigger a rerun of the affected cells.
- **Design.** (a) `code/e2_filter.py --model llama --set a23` (fixed first template per relation, 8 tuning facts per relation, template OK if >= 50% on tuning in both languages) writes `data/v2/e2a23_facts_llama.json`; `code/e2_filter_hi.py --model llama --set a23` (three question/answer-format Hindi candidates per relation chosen on the tuning facts, relation kept if >= 8 kept facts) writes `data/v2/e2ha23_facts_llama.json`. (b) EN<->ES interchange, site A, last subject token, with the A19 settings (`--k-src 2 --shuffle-pairs --max-pairs 200`), tag `e2a23_A`; EN<->HI span-wise (`--span end --k-src 4 --max-pairs 200`), tag `e2ha23_A`; Llama-3.2-3B-Instruct only. (c) Analysis with `code/e2_analysis.py` using the A13/A18 gates and labels, leak-free and plain, and clustering sensitivity (`cluster_sensitivity.py`; the set is added when the runs exist). Driver `scripts_a23.sh`.
- **Decision rules (as A13/A16/A18).** A cell is testable only if the P1 gate passes and there are >= 25 source entities. Label SHARED if the lower 95% bound is >= 0.7, LANGUAGE-SPECIFIC if the upper 90% bound is <= 0.3, else PARTIAL; a headline only if identical under the four clusterings. *Generalises* if the pooled non-geography EN->ES and ES->EN cells are both SHARED. *Relation-dependent* if any single relation (>= 10 source entities) has a point estimate outside 0.6-1.3 or a different label from the pooled cell.
- **Predictions (stated now).** (1) Llama keeps >= 50 EN/ES facts over the four relations (60%) and >= 100 (20%); per relation: official_language >= 20 (65%), currency >= 10 (50%), profession >= 15 (50%), book_author >= 10 (40%). (2) Hindi keeps >= 25 facts over the four relations (35%). (3) Pooled EN->ES and ES->EN pass the gate (75%) and are SHARED (55%). (4) At least one relation is not SHARED on its own or is untestable (70%). (5) If Hindi is testable, I is within 0.6-1.0 (60%).
- **Caveats stated in advance.** One model; hand-written, unaudited facts; strict answer matching favours short, canonical answers; book titles and professions vary more in translation than geography does. If fewer than 25 source entities survive, the run is reported as untestable and no claim is made.
- **Code.** `code/build_facts_a23.py`, `code/e2_filter.py --set a23`, `code/e2_filter_hi.py --set a23`, `code/interchange.py --facts`, `scripts_a23.sh`. Reported whatever the result.

### A24: write the edit at a key shared across languages (registered 2026-10-29 before any run; GPU, about 6 h after A21; code is CPU-tested, unit tests in `tests/test_core.py`)
- **Gap and idea.** The interchange test says the subject state is shared (ratios 0.8-1.2), yet an English-key rank-1 edit reaches Spanish about half of the time and Hindi rarely (A20). The edit is written at the key k_EN (the MLP hidden state at the last English subject token). If the Spanish and Hindi keys point in other directions, the update does not fire for them. A24 writes the update at a shared key: `mean_all` (rank-1 update at the mean of the EN, ES, HI keys) and `joint_all` (rank-3 update that sets the output to old + delta for each of the three keys, `joint_update` in `code/knowledge_edit_v2.py`). Same delta (fitted on English prompts), same covariance C, same facts, layers and replacement answers as the A21 run, which supplies the English-key baseline (`--key-modes en` is the default and is unchanged).
- **What differs from an English-only edit.** The Spanish and Hindi keys come from the Spanish and Hindi **subject names**, so the method needs the subject translated; an English-only edit does not. This is stated as the price of the method.
- **Design.** Llama-3.2-3B-Instruct, the 30 facts of A21 (edit positions 11-40); layers: the fact's traced layer and layer 14; `--key-modes mean_all,joint_all --para` (`scripts_a24.sh`, run only after A21 exists). Evaluation on **held-out phrasings** (`prompts[1:]` in each language: not used to fit keys or delta) and, for reference, `prompts[0]`. Per (fact, layer): English success, transfer = dP(target) / dP(English `prompts[0]`) of the same edit (facts with English dP <= 0.1 in either condition dropped), specificity (max |dP| over 20 controls), perplexity ratio. `code/keymode_analysis.py` reports medians with bootstrap-over-facts 95% CIs and **paired differences** against the English-key baseline; the key cosines cos(k_EN, k_ES) and cos(k_EN, k_HI) are recorded.
- **Decision rules.** *Improves transfer to language L* if the paired difference in held-out transfer (mode minus English-key baseline) has a 95% CI above 0 and a median >= 0.15 at the same layer condition, with median specificity <= 2x the baseline's, perplexity ratio <= 1.01 and English success within 0.1 of the baseline. *No improvement* if the CI includes 0 or the median is < 0.15. Traced and layer-14 conditions are reported separately; no pooling.
- **Predictions (stated now).** (1) `joint_all` raises held-out Spanish transfer at the traced layer by >= 0.15 (55%); at layer 14 the baseline is already 0.75, so the gain there is below 0.15 (65%). (2) `mean_all` improves Spanish less than `joint_all` (70%). (3) Hindi held-out transfer rises by >= 0.15 under `joint_all` at the traced layer (45%) and stays below 0.3 in absolute terms under `mean_all` (60%). (4) The median cos(k_EN, k_ES) exceeds the median cos(k_EN, k_HI) (85%). (5) Specificity under `joint_all` is within 2x of the baseline (60%). (6) English success stays within 0.1 of the baseline for `joint_all` (80%) and `mean_all` (65%).
- **Caveats stated in advance.** One model, 30 facts, descriptive paired differences, no multiplicity correction across 2 modes x 2 layers x 2 languages (eight differences, all reported). A rank-3 update has more freedom than a rank-1 one, so `joint_all` may harm specificity for reasons unrelated to transfer; `mean_all` is the rank-matched comparison. A null result is reported as such and would mean the English-key gap is not a key-direction problem.
- **Code.** `code/knowledge_edit_v2.py --key-modes ... --para`, `code/keymode_analysis.py`, `scripts_a24.sh`. Reported whatever the result.

### A25: the Kaggle rerun (registered 2026-10-29, amended the same day before any Kaggle run; changes how runs execute, not what they measure)
- Scope: every GPU result is regenerated on Kaggle from scratch, one step per notebook (`kaggle/steps.py`): fact preparation, tracing, the noise and phrasing checks, the swap tests for all three models, the edit runs, then A21/A23/A24. The Kaggle dataset carries no laptop results and no laptop-derived fact files, so nothing can mix. Laptop results stay as they are; the rerun is reported as a reproduction, with every difference reported.
- Hardware: Kaggle 2x Tesla T4 (15 GB each); same torch 2.11.0+cu128 and pinned transformer-lens 3.9.0 / transformers 5.19.0.
- Sharding: long runs may be split by fact (edits) or by pair (swap tests) into one shard per GPU (`--shard i/n`, applied after the seeded selection, so shards partition exactly the unsharded set; per-fact seeds unchanged). `kaggle/merge_shards.py` joins them. Unit-tested (`tests/test_kaggle.py`).
- Number format, decided once per model before its first job (`kaggle/dtype_check.py`): 24 fixed EN/ES/HI factual prompts in bf16 and in fp16, recording the top-1 token and the gradient of its loss at the middle layer's MLP output. Use fp16 only if all values are finite AND top-1 agreement with bf16 >= 0.95 AND median gradient cosine >= 0.9 AND fp16 is >= 1.5x faster; otherwise bf16. If bf16 itself fails on the T4 and fp16 is finite, fp16. The decision and numbers are saved (`analysis/kaggle_dtype_decision.json`) and the format is written into every run config. (Amendment: replaces this morning's 2-fact edit check, which needed laptop-derived facts.)
- Caveat: the format check only detects breakage. The reproduction comparison against the laptop bf16 runs is the equivalence check; a difference outside the laptop CIs is reported as a possible format or hardware effect.
- **Status (2026-10-29, night): superseded.** The Kaggle rerun was dropped before any Kaggle result existed; all runs stay on the laptop GPU. Nothing above was measured. Hardware, sharding and number-format text is kept for the record only.

### A26: why the English-key edit fails, baselines, and a benchmark slice (registered 2026-10-29 before any of these data exist; laptop only)
- **Why now.** Reading Bi-ZsRE (2309.08952), XME (2401.10521), LU-LAFNs (2406.16416) and CLM-Bench (2601.17397) shows: parameter edits carry over poorly across languages, in-context editing carries over far better (Bi-ZsRE, ROME English->Chinese generality 23-30 F1 against 92-97 for in-context), LU-LAFNs gains only about 2 F1 over ROME when edited in English and tested in Chinese (Llama2-7B, their Appendix B), and edit vectors for the two languages are nearly orthogonal (CLM-Bench). None of them ties this to a shared read-out measured per fact. Bi-ZsRE has English and Chinese only, so Spanish and Hindi stay on our controlled facts.
- **Test 1, key similarity (CPU, no new model run; needs the A21 and A24 outputs).** For language L in {ES, HI} and each layer condition (traced, layer 14): Spearman rho between cos(k_EN, k_L) at the edit layer and the held-out transfer to L of the English-key edit (A21 run); facts with English dP <= 0.1 dropped as in A24. Prediction: rho > 0 in all four cells (70%); Hindi cells weaker than Spanish cells (65%). Supported if rho >= 0.30 with a bootstrap-over-facts 95% CI above 0, Holm-corrected over the four cells. Confound check: partial Spearman controlling for English edit strength (dP_EN), the language's subject span length and shared-subword Jaccard; "beyond confounds" only if partial rho >= 0.20 with CI above 0. Caveat: about 30 facts per cell, and Hindi keys may vary little (low power); a null means key similarity does not by itself explain which facts transfer, and the paper then drops the mechanism wording.
- **Test 2, baselines on the controlled facts (Llama-3.2-3B-Instruct, the A21 facts, held-out phrasings, ES and HI).** (i) English-key rank-1 edit (A21, exists). (ii) MEMIT-style multi-layer edit written by us from Meng et al. 2022 (arXiv:2210.07229), covariance from WikiText-2, hyperparameters fixed in code and logged before the first run, no tuning on the test facts. (iii) In-context edit (existing code `code/incontext_baseline.py`, extended to held-out phrasings). (iv) A24 shared-key edits. Same metrics as A24; paired differences against (i) with bootstrap over facts.
- **Test 3, benchmark slice (Bi-ZsRE English-Chinese, authors' public data; Qwen2.5-3B-Instruct, which reads Chinese).** 200 test samples drawn with seed 0 (fixed now). Edit in English, test in Chinese, and the reverse; reliability, generality and locality as in the benchmark's `evaluate.py` (F1 and exact match). Methods: English-key ROME, shared-key edit (keys from the English and Chinese subject), in-context. A published multilingual editor (LU-LAFNs, code github.com/XZhang00/LU-LAFNs, commit 6841e99, which has a Qwen script) is run on the same slice if its 30 paraphrases per question can be made locally (the authors used a 72B API; $0 rule), recorded as a deviation; otherwise its published numbers are quoted with the model-size difference stated. If Qwen2.5-3B fails the smoke test, Qwen2.5-1.5B-Instruct is used and logged.
- **Decision rule for Test 3.** The shared-key edit "improves cross-lingual transfer" if English-to-Chinese generality rises by >= 5 F1 points over the English-key edit (bootstrap over samples, CI above 0) with locality down by <= 2 points. The 5-point bar is set above the roughly 2-point gain reported for LU-LAFNs. Prediction: met on the translated-subject edit (50%).
- **Wording rule.** The claim will be "best training-free weight edit among those compared", never "best method": in-context editing is expected to score higher on generality (92-97 F1 in the original paper) and is reported alongside.
- **Not run before:** MEMIT code, Bi-ZsRE loader, the Qwen entry in `code/common.py` and the benchmark scorer do not exist yet; they are written and unit-tested on CPU first. Each GPU run gets a VRAM and runtime estimate and waits for the user's go.

### A27: Llama-3.2-3B-Instruct runs on Kaggle (registered 2026-10-29 before any Kaggle result; changes where the pending runs execute, not what they measure)
- Scope: one Kaggle notebook (`kaggle/steps.py`, step `babel_llama`) runs the pending Llama jobs: the English-paraphrase control (A21), the shared-key edit (A24) and the non-geography facts (A23), plus a tokenizer comparison. The design, decision rules and predictions of A21, A23 and A24 are unchanged. The saved laptop tracing run (`analysis/run_20261015_p3trace_v2_llama`) is the input of the edit jobs, exactly as it would be on the laptop, so facts and traced layers are the laptop's. This replaces the "regenerate everything on Kaggle" scope of A25; hardware, number-format rule and sharding text of A25 apply to these jobs.
- Model: weights from Meta's own Kaggle Model (`metaresearch/llama-3.2`, Transformers, `3b-instruct`, version 1). Before any job starts the runner checks every one of the 254 weight tensors byte for byte against the laptop copy (`code/model_manifest.py`, fingerprint in `data/model_manifests/`); a single difference stops the run. Config and tokenizer files are the laptop copy's (`data/model_files/`).
- Finding that prompted this: the laptop's config, generation config and tokenizer files come from the Unsloth mirror (`unsloth_fixed: true`, pad token set, tokenizer.json re-serialised; 17.2 MB against 9.1 MB in Meta's release). They differ in bytes from Meta's, and a raw-tokenizer test on six EN/ES/HI strings gave different token ids. Weights are identical (all 254 tensors). Every laptop result used the mirror's tokenizer. The notebook compares both tokenizers over every string in the fact, template and candidate files and reports how many differ (`analysis/tokenizer_compare.json`). Outcome decides whether the paper must state the tokenizer source: if no id differs on our prompts, one sentence; if some do, those facts are listed and the affected results are rechecked.
- Number format: decided by the A25 rule (bf16 or fp16 equivalence check, saved in `analysis/kaggle_dtype_decision.json`, written into every run config). A difference from a laptop bf16 run is reported as a possible format or hardware effect, not as a finding.
- GPU use: one job per T4 (about 7.5 GiB each of 15 GiB for the 3B model), two GPUs busy at once. Time budget 10.5 h; unfinished jobs resume when the notebook is run again with its own output attached.
- Untested before the first Kaggle run: every GPU step on a T4, fp16, and the two-GPU scheduling on Kaggle (the scheduler and the model preparation are unit-tested on CPU; a local dry run of the check steps is logged in docs/LOG.md).

### Verdicts for A21, A23, A24 and A27 (appended 2026-10-30 after the second full Kaggle run, code id bundle-f122bd5c1df0, 2x T4, fp16; the registered texts above are not edited)
Data: `analysis/run_20261030_*` (edit runs `p4a21r023249a1_llama`, `p4a24r023334a1_llama`; analyses `para_analysis_A21`, `keymode_analysis_A24`, `e2a23_analysis*`, `e2ha23_analysis*`), `analysis/tokenizer_compare_kaggle_llama.json`. The first Kaggle run (bundle-05074174f20e) gave pixel-identical graphs; its two Spanish swap-test analyses failed for missing inputs (fixed, not a result).
- **A21 sanity check (prediction 4): FAILED.** Median absolute difference against the stored laptop edits (60 fact-layer pairs): English dP 0.101, Spanish dP 0.114 (limit 0.02). No systematic shift: signed median differences 0.014 (EN), 0.006 (ES), 0.000 (HI); group medians EN 0.918 vs 0.871, ES 0.569 vs 0.584, HI 0.012 vs 0.009; English success 0.933 vs 0.850; but single edits differ by up to 0.98. Per the registered rule the control is flagged **non-reproducing** at the level of single edits. Possible causes, not separated: fp16 on T4 instead of bf16 on the laptop GPU, the 25-step fitting. Median-level statements below are usable; per-fact statements are not comparable across the two machines.
- **A21 primary quantity D (English-paraphrase transfer minus language transfer, median [95% CI]).** Spanish, traced layer 0.373 [0.158, 0.588] (n=28): above 0.15, so Spanish transfer is lower than transfer to unseen English wording (loss beyond wording; lower bound only just above 0.15). Spanish, layer 14: 0.131 [0.053, 0.309] (n=29): median inside +-0.15 but the interval reaches 0.31, so neither rule is cleanly met (inconclusive). Hindi: 0.830 [0.666, 0.950] traced, 0.656 [0.484, 0.800] layer 14: a large loss beyond wording in both.
- **A21 predictions:** (1) layer 14 English paraphrase transfer in 0.5-0.95 and D within +-0.15: not met (English paraphrase 0.982). (2) traced D > 0.15: met. (3) Hindi more than 0.3 below English paraphrase in both conditions: met. (4) sanity check: not met. (5) unseen Spanish phrasings within 0.15 of Spanish prompts[0]: not met (0.327 vs 0.503 traced, 0.676 vs 0.851 layer 14). 2 of 5 met.
- **A24 decision rule (paired difference in held-out transfer, mode minus English key; CI above 0 and median >= 0.15): no improvement in any of the eight cells.** Spanish traced: joint_all 0.061 [0.009, 0.178], mean_all 0.079 [0.027, 0.184] (CI above 0 but median below 0.15); Hindi traced: joint_all 0.009 [0.000, 0.050], mean_all 0.034 [0.001, 0.199]; layer 14: all medians within +-0.03 of 0 (Spanish -0.027 to 0.003, Hindi 0.005). Group medians look larger (Spanish traced 0.327 baseline vs 0.535 joint_all, 0.687 mean_all) because the gains sit in a few facts; the registered paired median is what counts. Other measures: English success within 0.1 of baseline in every cell (mean_all layer 14 exactly -0.10); specificity joint_all 1.4x (traced) and 1.98x (layer 14) the baseline; perplexity ratio <= 1.002.
- **A24 predictions:** (1) joint_all raises Spanish traced transfer by >= 0.15: not met; layer 14 gain < 0.15: met. (2) mean_all improves Spanish less than joint_all: not met (0.079 vs 0.061). (3) Hindi joint_all +0.15 at the traced layer: not met; Hindi mean_all absolute < 0.3: met. (4) cos(k_EN,k_ES) > cos(k_EN,k_HI): met (median 0.559 vs 0.367). (5) joint_all specificity within 2x: met (1.98x at layer 14, borderline). (6) English success within 0.1: met for joint_all; mean_all met at the traced layer, on the limit at layer 14. 3 of 6 clearly met, 2 not, 1 split.
- **A24 reading (pre-registered interpretation of a null):** keys differ across languages (cosine 0.56 for Spanish, 0.37 for Hindi) yet writing the update at a shared key does not close the gap, so the English-key gap is not mainly a key-direction problem under these two constructions; the training-free fix claimed in the plan is not supported. Plan B (diagnostic paper) stands.
- **A23: untestable, no claim.** Kept facts: EN/ES 53 (official_language 40, book_author 13; currency and profession dropped by the template filter, English accuracy on tuning facts 0/8 and 2/8), EN/HI 13 (official_language only). Testability rule (P1 gate and >= 25 source entities) fails in all four directions: source entities 15 (EN->ES, ES->EN) and 13 (EN->HI, HI->EN); the EN->ES P1 gate also fails. Predictions: >= 50 EN/ES facts met (53); official_language >= 20 met (40); book_author >= 10 met (13); currency >= 10 and profession >= 15 not met (0 each); Hindi >= 25 facts not met (13); pooled SHARED not assessable. Point estimates, descriptive only: EN->ES 1.12 [0.96, 1.21], ES->EN 0.95 [0.84, 1.07], EN->HI 0.69 [0.58, 0.84], HI->EN 0.86 [0.76, 0.96]. The automatic labels in results.json (P2 SHARED, P4 PARTIAL) must not be quoted as verdicts.
- **A27 tokenizer comparison (12,689 strings, 35 files):** token ids of the text are identical in the mirror and in Meta's release for every string (0 differ without special tokens); with special tokens all differ because the mirror's tokenizer prepends the beginning-of-text token (128000) by default and Meta's release does not; pad token differs (128004 vs none); vocabulary 128,256 in both. In code I read, the covariance windows add the beginning token explicitly; I did not trace every path. The paper states the tokenizer source in one sentence. Kaggle and laptop runs used the same tokenizer files, so they are comparable to each other.
- **A27 number format:** fp16 chosen on the T4s (1.85x faster, top-1 agreement 1.00, median gradient cosine 0.9997). Run times: paraphrase control 41 min, shared-key edit 58.5 min, Spanish swap run 18 min, Hindi swap run 9 min, whole run 73.9 min.

### A26 Test 1, first pass on the Kaggle data (appended 2026-10-30; the registered text above is not edited)
Run: `code/key_similarity.py` on the Kaggle English-key edits (`analysis/run_20261030_p4a21r023249a1_llama`) and the key cosines stored in the Kaggle shared-key run (`analysis/run_20261030_p4a24r023334a1_llama`; identical in both key modes, checked), output `analysis/run_20261030_key_similarity_A26`. 28-29 facts per cell.
- **Traced layer: supported by the registered rule.** Spanish rho 0.550 [0.200, 0.762], Holm-adjusted p 0.006; Hindi rho 0.521 [0.127, 0.815], Holm-adjusted p 0.010. Partial rho with the registered controls (English edit strength, subject token count, subject-piece Jaccard): 0.520 [0.089, 0.786] and 0.531 [0.185, 0.816], so "beyond confounds" by the registered rule. Leave-one-fact-out rho 0.50-0.62 (Spanish) and 0.47-0.62 (Hindi).
- **Layer 14: no relation.** Spanish rho -0.061 [-0.436, 0.323], Hindi 0.020 [-0.393, 0.448]; not supported.
- **Predictions:** (1) rho > 0 in all four cells: NOT met (Spanish layer 14 is -0.061). (2) Hindi weaker than Spanish in both conditions: NOT met (Hindi 0.521 vs Spanish 0.550 at the traced layer is nearly equal; at layer 14 Hindi is higher).
- **Extra check, not registered, reported because it changes the reading.** The traced layer differs by fact (layers 0, 1, 2, 4, 5, 6 with 8, 1, 7, 7, 3, 2 facts), and both key similarity and transfer can depend on the layer. With the edit layer added as a fourth control the partial rho is 0.450 [-0.013, 0.752] for Spanish and 0.061 [-0.323, 0.574] for Hindi. So with 28 facts the Hindi traced-layer relation cannot be told apart from an edit-layer effect, and the Spanish one is borderline. The mechanism wording is NOT supported yet; A28 below is the registered follow-up.

### A28: scale-up and layer map for the key-similarity result (registered 2026-10-30 before any of these data exist; Kaggle only, Llama-3.2-3B-Instruct)
- **Why.** The first pass above had 28 facts per cell and one edit layer per fact, so it cannot separate "close keys transfer better" from "early layers transfer differently". A28 asks again with about three times the facts and several fixed edit layers. Everything in the run is produced on Kaggle: the notebook traces the facts itself and the edit runs use that tracing (no saved laptop run is an input of the edit jobs; the fact file `data/v2/facts_llama.json` is a fixed input).
- **Design.** (1) Tracing of all kept facts (`code/causal_trace_v2.py --subset all`, phrasing 0): the source of each fact's traced layer. (2) Base run: all valid facts, English-key edit at the fact's traced layer and at layer 14, `--para`. (3) Shared-key run: same facts and layers, `--key-modes mean_all,joint_all --para`. (4) Layer map: the first 40 valid facts, layers 2, 6, 10, 18 and 22, `--key-modes en,mean_all --para` (the `en` rows are the English-key edit, written with the key cosines). Number format from the A25 rule; same fitting code, steps and answers as A21/A24.
- **Analyses (CPU).** `code/para_analysis.py` (A21 rules) and `code/keymode_analysis.py` (A24 rules) on runs (2) and (3); `code/key_similarity.py` (A26 Test 1) on runs (2) and (3); `code/layer_map.py` on run (4).
- **Decision rules.** (a) A26 Test 1 rule unchanged (rho >= 0.30, bootstrap CI above 0, Holm over the four cells). (b) *Key similarity predicts transfer beyond the edit layer* in the traced condition if the partial rho with dP_EN, subject token count, subject-piece Jaccard AND the edit layer as controls (`extra_partial_rho_with_layer`) is >= 0.20 with the CI above 0, per language. (c) Layer map: per layer and language, the Spearman rho between cos(k_EN, k_L) and held-out transfer over facts with English dP > 0.1; the pooled statistic is the mean of the per-layer rho over layers with >= 20 usable facts, bootstrap over facts with all layers resampled together; supported if pooled rho >= 0.20, CI above 0 and the Holm-adjusted one-sided p < 0.05 over the two languages. (d) The A21 and A24 rules are repeated on the larger runs unchanged. (e) The per-layer medians of held-out transfer are descriptive (no threshold).
- **Predictions (stated now).** (1) Spanish traced-layer rho >= 0.30 with the CI above 0 again (60%). (2) Hindi traced-layer partial rho with the layer control >= 0.20 with the CI above 0 (30%). (3) Layer-14 rho within +-0.2 of zero for both languages (65%). (4) Pooled within-layer rho supported for Spanish (50%) and for Hindi (30%). (5) Median Spanish held-out transfer is below 0.5 at layers 2 and 6 and at least 0.5 by layer 14 (65%). (6) The shared-key gain stays below 0.15 in all eight cells (75%). (7) The English-paraphrase gap D stays above 0.15 for Hindi in both layer conditions (90%).
- **Caveats stated in advance.** One model; mostly capital relations; held-out transfer uses the same fitted edit (key cosines are measured, not manipulated, so a correlation is not a mechanism: the shared-key edit of A24 is the intervention and did not help); the fp16 T4 numbers cannot be compared to any earlier machine and are not; facts with English dP <= 0.1 are dropped, so n differs by cell; subject translations are needed for the language keys.
- **Run-time estimate (from the last Kaggle run: 41 min for 60 fact-layer pairs without shared keys, 58.5 min for 60 pairs with two shared-key modes).** Tracing about 0.5 h (not measured on a T4), base run about 2.1 h, shared-key run about 3.0 h, layer map about 3.2 h; two T4s in parallel; whole step about 6 h; limit 10.5 h; unfinished jobs resume when the notebook is run again with its own output attached.
- **Untested on a T4 before the run:** `code/causal_trace_v2.py` in fp16, and the three new scales. The scheduler, the job list and the analyses are CPU-tested.
- **A28 verdicts (2026-10-31; notebook bundle-f8371a452538, 2x T4, fp16, 348 min, all 20 jobs done; 86 facts edited, layer map on the first 40; data in analysis/run_20261030_*_llama, analysis/run_20261031_layer_map_A28, copy of the Output in analysis/kaggle_output_20261031).**
  - (a) Key similarity at the traced layer: Spanish rho 0.411 [0.198, 0.586], Holm p 0.001; Hindi 0.418 [0.186, 0.615], Holm p 0.001; both supported. Layer 14: 0.059 [-0.173, 0.278] and 0.015 [-0.203, 0.239], not supported.
  - (b) Beyond the edit layer (traced, partial rho with dP_EN, token count, Jaccard and edit layer as controls): Spanish 0.26 [0.004, 0.494], Hindi 0.22 [0.011, 0.438]; both meet the rule (>= 0.20, CI above 0), by lower bounds of 0.004 and 0.011.
  - (c) Layer map (layers 2, 6, 10, 18, 22): pooled within-layer rho Spanish 0.286 [0.117, 0.437], Holm p 0.0024, supported; Hindi 0.171 [-0.024, 0.361], Holm p 0.047, NOT supported (below 0.20, CI includes 0). Per layer rho Spanish 0.59, 0.29, -0.13, 0.21, 0.46; Hindi 0.43, 0.70, -0.02, -0.04, -0.22.
  - (d) A21 repeated: D (English-rewording transfer minus language transfer) Spanish traced 0.389 [0.290, 0.494], layer 14 0.257 [0.153, 0.359] (above 0.15 by 0.003); Hindi traced 0.786 [0.674, 0.893], layer 14 0.731 [0.642, 0.800]: loss beyond wording in all four cells. A24 repeated: largest median shared-key gain over the eight cells 0.047 (traced Spanish 0.040 mean_all, 0.047 joint_all; Hindi at most 0.013); no gain of practical size, the fix is not supported.
  - (e) Median held-out transfer by edit layer (2, 6, 10, 14, 18, 22): Spanish 0.387, 0.419, 0.319, 0.621, 0.169, 0.007; Hindi 0.008, 0.004, 0.005, 0.038, 0.004, 0.000.
  - Predictions: (1) met (0.411); (2) met, barely (0.22, lower bound 0.011); (3) met (0.059, 0.015); (4) Spanish met, Hindi NOT met; (5) met (0.387, 0.419 below 0.5; 0.621 at layer 14); (6) met (max 0.047); (7) met. 7 of 8 sub-predictions met.
  - Reading: key similarity predicts which facts carry over for Spanish in both the layer-controlled and the within-layer test; for Hindi only in the layer-controlled test (narrowly) and not within a layer. It is a correlation: the shared-key edit, the intervention, did not help. Caveats: where median transfer is near 0 (Hindi at every layer, Spanish at layer 22) the within-layer rho ranks tiny values and is weak evidence; one model; mostly capital relations; 36-38 facts per layer.
  - Non-geography facts (A23) rerun: unchanged, untestable (Spanish 53 facts but 5 source entities against the floor of 25; Hindi 13 facts). Reproducibility: the paraphrase numbers of this fresh session equal those of the earlier session's checkpoint to three decimals.
  - Time on 2x T4 (measured): tracing 108 min, base edit run 112 min, shared-key run about 1.95 min per fact; the progress bars of the edit jobs count to the cap of 200, not to the 86 facts.

### A29: does the key-similarity result replicate on Gemma-2-2B? (registered 2026-10-31 before any Gemma-2-2B Kaggle data exist; Kaggle only)
- **Why.** A28 is one model (Llama-3.2-3B-Instruct). The standing model rule (user, 2026-10-31) is to use the models the published editing papers used and to show our result on them. Gemma-2-2B (base, `google/gemma-2-2b`: 26 layers, width 2304, a norm after the MLP) is the same size class and is the first replication. A model-specific result would be reported as such.
- **Design (identical to A28 except as listed).** Tracing of every kept fact; base run (English-key edit at the traced layer and at the middle layer, `--para`); shared-key run (`mean_all`, `joint_all`); layer map (first 40 facts, layers 2, 6, 10, 18, 22, `en,mean_all`); the same CPU analyses (`para_analysis.py`, `keymode_analysis.py`, `key_similarity.py`, `layer_map.py`). Fact file `data/v2/facts_gemma2.json` (58 kept: event_year 30, capital_of 11, capital_country 11, landmark_city 6); the edit runs take the 52 facts of relations with at least 10 kept facts (Llama: 87). Differences: (i) the middle layer is 13 (= 26 // 2), not 14; the layer map layers are unchanged, so layer 13 comes from the base run; (ii) no non-geography chain (A23): it was untestable with 93 Llama facts and Gemma has fewer; (iii) the facts are mostly event years (30 of 52 against 22 of 87), so the relation mix differs from A28; (iv) weights from Google's own Kaggle Model (`google/gemma-2`, Transformers, `gemma-2-2b`), checked tensor by tensor (288 tensors, 10.457 GB) against the laptop copy before any job starts (`data/model_manifests/gemma-2-2b.json`); a copy with other bytes stops the run. The Kaggle dataset `leimeng46/gemma-2-2b` is NOT used: its file sizes (config 627 B, shards 4.95 GB and 67.1 MB, index 13.5 kB) equal those of `google/gemma-2b`, the first-generation 18-layer Gemma, not of Gemma-2-2B (26 layers, 10.5 GB); the runner also refuses a folder whose config has the wrong layer count. Number format by the A25 rule (Gemma-2 can overflow in fp16; bf16 is then used, slower on a T4).
- **Decision rules.** Unchanged from A28 (a)-(e): the A26 Test 1 rule (rho >= 0.30, bootstrap CI above 0, Holm over the four cells), the layer-controlled partial rho (>= 0.20, CI above 0), the pooled within-layer rho (>= 0.20, CI above 0, Holm p < 0.05), the A21 wording rule (D above 0.15) and the A24 shared-key line (gain 0.15). *Replicates* means the Spanish traced-layer cell is supported by the A26 rule; it replicates *fully* if the Hindi traced cell is supported as well. Nothing is tuned after seeing the data.
- **Predictions (stated now, with my confidence).** (1) Spanish traced-layer rho >= 0.30 with the CI above 0 (55%; 0.41 on Llama, but about 52 facts instead of 80 gives roughly 70% power if the true value is 0.41). (2) Hindi traced-layer rho >= 0.30 with the CI above 0 (40%). (3) Middle-layer (13) rho within +-0.2 of zero for both languages (55%). (4) Median Hindi held-out transfer below 0.1 at every edit layer (75%). (5) Median Spanish held-out transfer lower at layers 18 and 22 than at layers 2 to 13 (65%). (6) The shared-key gain stays below 0.15 in all eight cells (75%). (7) The English-paraphrase gap D stays above 0.15 for Hindi in both layer conditions (90%).
- **If (1) fails.** The key-similarity result is then supported on Llama only, and the paper says so (one model supports it, one does not at this sample size); no rule is changed and no cell is dropped. The facts are fixed (no fact is added to raise n).
- **Caveats stated in advance.** 52 facts give wider intervals than 87; the relation mix differs (event years dominate); one more model is still two models, not a family; Gemma-2 on TransformerLens 3.9.0 on a T4 is untested here (the Gemma edit path ran on the laptop in the earlier p4 runs, the v2 edit code and the T4 did not); the fp16/bf16 outcome changes speed, not the registered rules.
- **Run-time estimate (not measured on a T4 for this model; scaled from the Llama run).** Tracing about 0.7 h (58 facts, 26 layers, smaller width), base run about 1.0 h, shared-key run about 1.5 h, layer map about 1.5 h (two GPUs in parallel); whole step 3 to 5 h if fp16 is accepted, longer if bf16 is used; limit 10.5 h, unfinished jobs resume when the notebook is run again with its own output attached.
- **Untested on a T4 before the run:** every GPU job on Gemma-2-2B (smoke, tracing, the three edit runs). Tested on the CPU: the job graph, the model lookup by whole folder name, the byte check, per-model packing, the progress cap (equals the 52 qualifying facts).

### Note on the A28 verdict text and on file names (appended 2026-10-31; the verdicts above are not edited)
- The "largest median shared-key gain 0.047" in verdict (d) was read from the main-prompt column (Spanish, traced layer, joint_all). The registered quantity is the paired gain in held-out transfer; its largest value over the eight cells is 0.044 (mean_all, Spanish, traced layer, CI [0.008, 0.076]). Both are below 0.15; the verdict (no improvement) is the same. The paper (Section 4.8) uses the held-out values.
- The Kaggle Output of the A28 run wrote its three analyses into folders that carry the first-pass names (`run_20261030_key_similarity_A26`, `run_20261030_para_analysis_A21`, `run_20261030_keymode_analysis_A24`) and replaced the 28-fact first-pass files of those names inside the zip. The repo copies of those three folders are still the first pass. The A28 versions are saved, byte for byte from the zip, as `analysis/run_20261031_key_similarity_A28`, `run_20261031_para_analysis_A28` and `run_20261031_keymode_analysis_A28` (the layer map was already `run_20261031_layer_map_A28`); the first-pass files are also kept as `run_20261030_key_similarity_A26_firstpass` and `run_20261030_para_analysis_A21_firstpass`. The manifest points to these.

### A29 amendment: first Kaggle attempt stopped at the smoke job (appended 2026-10-31; no Gemma-2-2B result exists, so nothing is changed after seeing data)
- What happened: the first run of `babel_gemma2` (code id bundle-18a19884bb78) loaded and byte-checked the model, passed the 61 unit tests, chose fp16 (1.74x faster, top-1 agreement 1.00, gradient cosine 0.999) and then failed the smoke job three times; the other 10 jobs were skipped. The smoke log was not available; I reproduced the check on the CPU with the local Gemma-2-2B weights: the smoke prompt "The capital of France is" gives " a" (0.20), " the" (0.10), " one" (0.07) and " Paris" only at 0.05, so the check exited with a wrong answer. The model is the right one (Gemma-2-2B is a base model; Llama-3.2-3B-Instruct answers that prompt with Paris). This is a fault of the check, not of the model or the GPU.
- Fix: the Gemma step uses the smoke prompt "Question: What is the capital of France? Answer:" (" Paris" 0.83 on the CPU in bf16); the Llama step is unchanged. On a wrong answer the smoke job now prints the top five tokens and whether the logits are finite.
- Added before any Gemma data: the number-format check (A25) now also runs each of its 24 prompts with the tracing noise on the first three tokens after BOS (3 x the embedding std, hook_embed) and counts non-finite logits as a failure of that format, so Gemma-2 in fp16 would fall back to bf16 in the first minutes instead of dropping facts silently in the tracing job. A25's other thresholds are unchanged. Decision text for Llama is unaffected (its decision was made earlier and its tracing finished in fp16).
- Not tested on a GPU: everything from the smoke job on. If the failed Output is attached to a new run, the earlier fp16 decision is reused and the new check is skipped, so the new run must start without it.

### A30 amendment: the Gemma-2-2B shared-key edit job failed; the key modes after the first may now fail without stopping the job (appended 2026-10-31; no shared-key result for Gemma exists, so nothing is changed after seeing data)
- What happened: the user run of babel_gemma2 (bundle-b7d6962e96d7, 238 min) passed the model check, 63 tests, the number-format check (fp16), the smoke job, tracing (44 min), the English-key edit run, both layer-map shards, the wording analysis and the layer map. The shared-key edit job (key modes mean_all and joint_all) failed on all three attempts, each right after its first written edit, so the key-similarity test, the key-mode analysis and the graphs were skipped. The error text was not in the console and the Output was not sent. Cause found the same day on the laptop GPU (not from the Kaggle log): the first edit fact in the job's sorted order is probably capital_country_003 (Madrid), whose subject text AND the words before it are the same in English and Spanish, so its English and Spanish keys are identical vectors. The joint update then states the same constraint twice, G = K^T C^-1 K is singular and torch.linalg.solve raises "input matrix is singular" (shown numerically with two equal columns). Checked on Gemma with capital_of_035 (Sri Lanka, same property): the English and Spanish key cosine is exactly 1.0. A first note here that identical subject names were ruled out was wrong: the property that matters is an identical key, not an identical name (the Llama facts with the same subject text have other words before it, their key cosines stayed below 0.995, so Llama never met this). I did not run the old code on this fact on the GPU; the cause rests on the numeric test plus the key cosine of 1.0.
- Change 1 (code/knowledge_edit_v2.py): the first key mode listed is required and raises as before; a key mode after it that fails (when the edit is written or when its effects are scored) is recorded with its traceback in the run folder (mode_failures.json), printed, and skipped, and the job ends with a WARNING line. The pre-registered statistics are unchanged: mean_all alone gives every key cosine the key-similarity test needs and the mean_all cells of the shared-key gain; joint_all cells are reported as not run if joint_all does not run, never as zero.
- Change 3 (code/knowledge_edit_v2.py, joint_update): columns of the key matrix that are copies of each other (relative difference below 1e-4) are reduced to one, because a copy states the same constraint (dW^T k = delta); the small m x m solve is done in float64. With distinct keys the result is unchanged (the existing tests of the update pass unchanged); with a copy the update equals the update without it. For such facts joint_all becomes a joint edit over the distinct keys only (English and Hindi when English and Spanish are identical), which is what the maths asks. Gemma 26-layer dry run on the laptop RTX 5060 (fp16): 3 facts at 2 layers and the Sri Lanka fact at 2 layers, both key modes, no failure.
- Change 2 (kaggle/driver.py): when a job fails for the last time the console shows the last 14 lines of its log, and a job that finishes with a WARNING line shows it next to DONE.
- Not tested on a GPU: the shared-key edit on Gemma-2-2B (the cause is unknown, so the rerun may still fail in the required mode; the console will then show the error).


# Part 9. Where everything is

All paths are on the laptop unless marked GitHub. Quote them in PowerShell and bash: the folder names contain spaces.

## 9.1 Main places

| What | Path |
|---|---|
| **Repository (source of truth)** | `C:\Users\Som Praneeth\Desktop\reasearch\babel-effect` |
| GitHub | https://github.com/praann07/babel-effect (branch `main`) |
| **One-folder copy of everything so far** (outside git, not on GitHub) | `C:\Users\Som Praneeth\Desktop\reasearch\BABEL_EFFECT_ORGANIZED\` (README.md, 01_STORY.md, notebooks, consoles, Llama Kaggle results, dry runs, paper, docs) |
| **This master record, PDF** | `paper\The_Babel_Effect_MASTER_RECORD.pdf` (in the repo) |
| Master record source | `paper\master\*.md` and `paper\build_master.py` |
| The short paper | `paper\paper_v2_draft.md`, PDF `paper\The_Babel_Effect_draft_v2.pdf`, LaTeX `paper\latex\main.tex` |
| Master goal file | `GOAL.md` |
| Pre-registrations | `docs\HYPOTHESIS.md` |
| Decision log / experiment log | `docs\LOG.md` / `docs\EXPERIMENT_LOG.md` |
| Next steps / Q1 roadmap | `docs\NEXT.md` / `docs\ROADMAP_Q1.md` |
| Results manifest (every number to a file) | `results\manifest.json` |
| Facts | `data\facts.json`, `data\v2\*.json`, audit sheet `data\audit\native_speaker_audit.csv` |
| Model manifests (tensor-by-tensor check against the laptop copy) | `data\model_manifests\`, small config packs `data\model_files\` |
| Local model weights (not in git) | `models\` |

## 9.2 Kaggle notebooks (self-contained; code packed inside)

| Notebook | Code id | What it needs on Kaggle |
|---|---|---|
| `kaggle\babel_gemma2.ipynb` | `bundle-e186cf5d5e1b` | GPU T4 x2, Internet on, Kaggle Model **Gemma 2 / Transformers / gemma-2-2b** (Google's own). Do not attach an old Output. |
| `kaggle\babel_llama.ipynb` | `bundle-4c26396318b4` | GPU T4 x2, Internet on, Kaggle Model **Meta Llama 3.2 / Transformers / 3b-instruct**. Do not attach an old Output. |

Both are in the repo, in the one-folder copy under `02_notebooks_to_run\`, and carry only their own model's files (Gemma 10.9 MB, Llama 8.3 MB).

## 9.3 Kaggle results and consoles

| What | Path |
|---|---|
| Llama final run (348 min, 86 facts): raw Output | `analysis\kaggle_output_20261031\` (copy: `BABEL_EFFECT_ORGANIZED\04_kaggle_results_llama\kaggle_output_20261031_A28_run`) |
| Llama analyses (key similarity, layer map, wording control, key modes) | `analysis\run_20261031_key_similarity_A28`, `..._layer_map_A28`, `..._para_analysis_A28`, `..._keymode_analysis_A28` |
| Llama raw edit runs | `analysis\run_20261030_p4a28*_llama` |
| Llama first pass (28 facts) | `analysis\run_20261030_key_similarity_A26_firstpass`, `..._para_analysis_A21_firstpass`, `analysis\run_20261030_keymode_analysis_A24` |
| Gemma failed run 2 (shared-key, `bundle-b7d6962e96d7`), console notebook you sent | `C:\Users\Som Praneeth\Downloads\notebook8a97a9a4df.ipynb` (copy and console text in `BABEL_EFFECT_ORGANIZED\03_kaggle_run_consoles\gemma2_run2_FAILED_shared_key_bundle-b7d6962e96d7\`) |
| Gemma run 4 (running, `bundle-e186cf5d5e1b`), console notebook you sent | `C:\Users\Som Praneeth\Downloads\notebook0476dd6f1a.ipynb` (copy and snapshot in `...\gemma2_run4_RUNNING_bundle-e186cf5d5e1b\`) |
| **Where the Gemma Output zip goes** | `BABEL_EFFECT_ORGANIZED\08_waiting_for_gemma_output\` (and, once read, `analysis\kaggle_output_<date>\` in the repo) |
| Laptop dry runs for Gemma | `analysis\run_20261031_p4a30dry*_gemma2`, copies and logs in `BABEL_EFFECT_ORGANIZED\05_laptop_dry_runs_gemma\` |

## 9.4 Commands (from the repo root, `.venv\Scripts\python.exe` is the Python with the libraries)

| Task | Command |
|---|---|
| Run the tests (69 pass, 1 skipped) | `.venv/Scripts/python.exe -m unittest discover -s tests -q` |
| Rebuild both notebooks after any code change | `.venv/Scripts/python.exe kaggle/make_notebook.py` |
| Check inside an unpacked notebook | unpack with the scratchpad script `unpack_nb.py`, then `PYTHONPATH=code python -m unittest discover -s tests` in the unpacked folder |
| Rebuild / verify the results manifest | `.venv/Scripts/python.exe code/make_manifest.py build` / `verify` |
| Rebuild the short paper PDF | `.venv/Scripts/python.exe paper/build_tex.py` (needs pandoc and pdflatex) |
| **Rebuild this master record** | `.venv/Scripts/python.exe paper/build_master.py` (needs pandoc and Microsoft Edge) |
| Key-similarity test on a Kaggle Output | `code/key_similarity.py` (CPU, minutes) |
| Layer map | `code/layer_map.py` (CPU) |

## 9.5 Agent side (the assistant's own notes)

| What | Path |
|---|---|
| Assistant memory (rules, state) | `C:\Users\Som Praneeth\.claude\projects\C--Users-Som-Praneeth-Desktop-reasearch\memory\` |
| Hive handoff | `C:\Users\Som Praneeth\HarnessAgents\hive\agents\rrap-research-assistant-mv52oi0t\handoff.md` |
| Full chat transcript | `C:\Users\Som Praneeth\.claude\projects\C--Users-Som-Praneeth-Desktop-reasearch\a63987bb-a70f-4518-9722-baa1fe5bdd7c.jsonl` |

## 9.6 Reading order for a new person

1. This Part 1 and Part 6 (goal and numbers).
2. Part 5, the paper.
3. Part 3, the problems, to see what to distrust.
4. Part 8, the pre-registrations, to see what was promised before each run.
5. Part 10, the logs, for day-by-day detail.

The rest of this part is generated from the repository each time the record is rebuilt: the top-level layout, the list of run folders, and the git history.

## 9.7 Repository layout (generated)

```text
analysis/       2772 files
code/             55 files
data/             49 files
docs/             12 files
kaggle/           11 files
paper/            47 files
reports/           1 files
research_notes/     4 files
results/           1 files
tests/             3 files

Files at the top level:
  GOAL.md
  README.md
  keep_awake.ps1
  reproduce.sh
  requirements.txt
  scripts_a17.sh
  scripts_a18.sh
  scripts_a19.sh
  scripts_a20.sh
  scripts_a21.sh
  scripts_a23.sh
  scripts_a24.sh
  scripts_after_e2.sh
  scripts_gpu_night2.sh
  scripts_hardstop.sh
  scripts_night_a13a.sh
  scripts_night_a16.sh
  scripts_night_e2.sh
  scripts_night_e2h.sh
  scripts_run_bloom.sh
  scripts_run_extra.sh
  scripts_run_gemma_templates.sh
  scripts_run_gemma_v2.sh
  scripts_run_gemma_v2c.sh
  scripts_run_incontext.sh
  scripts_run_llama1b.sh
  scripts_run_post.sh
  scripts_run_queue2.sh
  scripts_run_sweeps.sh
  scripts_watchdog.sh
  scripts_watchdog2.sh
  start_queue.ps1
  stop_queue.ps1
  track_night.ps1
  watch_queue.py
```

## 9.8 Run folders in `analysis/` (generated)

Write-once; one folder per run. Names carry the date, the test and the model.

```text
folder                                                         files       KB
_logs                                                            152     3856
figs_20261012_p2trace_v3                                          12     1279
figures_kaggle                                                     2      139
kaggle_output_20261031                                            38    28367
run_20261012_dbg                                                   5       10
run_20261012_dbg_edit                                              1        0
run_20261012_dbg_edit2                                             2        3
run_20261012_dbg_edit3                                             2        3
run_20261012_dbg_overlap                                           0        0
run_20261012_dbg_overlap2                                          4       78
run_20261012_p1filter                                              4      104
run_20261012_p1filter_v2                                           4      195
run_20261012_p2trace                                              32      104
run_20261012_p2trace_v2                                           32      103
run_20261012_p2trace_v3                                           32      103
run_20261012_p3overlap                                             4       98
run_20261012_p3trace                                             203      678
run_20261013_p1filter_repro                                        4      195
run_20261013_p3trace_repro                                       172      467
run_20261013_p4edit                                                5       87
run_20261014_p1v2_llama                                            4      783
run_20261014_p1v2b_llama                                           4      767
run_20261014_regress_llama                                         5       10
run_20261014_templates_llama                                       1        8
run_20261015_compare_models                                        4      162
run_20261015_dbg_llama                                             5       12
run_20261015_dbg_sweep_gemma2                                      2        2
run_20261015_dbg_sweep_llama                                       2        2
run_20261015_hindi_diag                                            2        2
run_20261015_p1v2_gemma2                                           4      820
run_20261015_p1v2b_gemma2                                          4      822
run_20261015_p1v2c_gemma2                                          4      898
run_20261015_p3overlap_v2_gemma2                                   4      110
run_20261015_p3overlap_v2_llama                                    4      113
run_20261015_p3overlap_v2b_llama                                   4      120
run_20261015_p3overlap_v2c_gemma2                                  4      136
run_20261015_p3trace_v2_gemma2                                    80      477
run_20261015_p3trace_v2_llama                                    281     1731
run_20261015_p3trace_v2c_gemma2                                  176     1021
run_20261015_p4sweep_llama                                         5      463
run_20261015_templates_gemma2                                      1        8
run_20261019_h8_10191057_gemma2                                    1        7
run_20261019_h8_10191057_llama                                     1        7
run_20261019_h8t_10191057_gemma2                                 182      988
run_20261019_h8t_10191057_llama                                  182     1161
run_20261019_incontext_10191057_llama                              0        0
run_20261019_layer_hist                                            2       53
run_20261019_p1v2extra_gemma2                                      4      321
run_20261019_p1v2extra_llama                                       4      294
run_20261019_p3extra_10191057_gemma2                               5       18
run_20261019_p3extra_10191057_llama                              104      627
run_20261019_p3overlap_v2all_10191057                              4      129
run_20261019_p3trace_v2extra_llama                                28      140
run_20261019_p4sweep_gemma2                                        5      438
run_20261019_templates_gemma2_extra                                1        3
run_20261019_templates_llama_extra                                 1        3
run_20261020_incontext_10200339_gemma2                             2       10
run_20261020_incontext_10200339_llama                              2       10
run_20261020_p3overlap_v2all_10200339                              4      136
run_20261020_p4f7_gemma2                                           4      242
run_20261020_p4f7_llama                                            4      241
run_20261021_controls                                              2        4
run_20261024_confound_A12a                                         2       25
run_20261024_confound_smoke2                                       2       25
run_20261024_e2filter_gemma2                                       2      198
run_20261024_e2filter_llama                                        2      192
run_20261024_e2filter_sel_gemma2                                   2      201
run_20261024_e2full_10240848_llama                                 0        0
run_20261024_e2pilot_10240848_gemma2                               2       18
run_20261024_e2pilot_llama                                         2       20
run_20261024_e2unit_llama                                          1        0
run_20261024_p3seeds_10240109_gemma2                             120      391
run_20261024_p3seeds_10240109_llama                              254      876
run_20261024_p3seedssmoke_10240104_llama                           6       14
run_20261024_seeds_A13a_10240109                                   2       21
run_20261024_seeds_covariates_A13b                                 2        4
run_20261025_confound_repro1                                       2       25
run_20261025_e2_analysis_A13                                       2       40
run_20261025_e2_analysis_A13_regress                               2       40
run_20261025_e2_repro1                                             2       40
run_20261025_e2full_A_gemma2                                       4     1507
run_20261025_e2full_A_llama                                        4     1486
run_20261025_incontext_diag_llama                                  1        1
run_20261025_ovg_repro1                                            4      136
run_20261025_ovl_repro1                                            4      121
run_20261025_p1v2c_bloom1b7                                        4      930
run_20261025_p1v2c_llama1b                                         4      852
run_20261025_p1v2extra_bloom1b7                                    4      325
run_20261025_p1v2extra_llama1b                                     4      298
run_20261025_p3overlap_v2_perm10k_gemma2                           4      136
run_20261025_p3overlap_v2_perm10k_llama                            4      121
run_20261025_p3overlap_v2all_perm10k_gemma2                        4      136
run_20261025_p3overlap_v2all_perm10k_llama                         4      129
run_20261025_seeds_repro1                                          2       21
run_20261025_seedscov_repro1                                       2        4
run_20261025_templates_bloom1b7                                    1        8
run_20261025_templates_bloom1b7_extra                              1        3
run_20261025_templates_llama1b                                     1        8
run_20261025_templates_llama1b_extra                               1        3
run_20261026_a18unit_llama                                         1        0
run_20261026_cluster_A17b                                          1       10
run_20261026_e2_analysis_A13_A17                                   2       50
run_20261026_e2h2_analysis_A16                                     2       23
run_20261026_e2h2_analysis_A16_A17                                 2       24
run_20261026_e2h2_outside_A15                                      1        0
run_20261026_e2h2filter_gemma2                                     2      267
run_20261026_e2h2filter_llama                                      2      241
run_20261026_e2h2full_A_gemma2                                     4      390
run_20261026_e2h2full_A_llama                                      4      493
run_20261026_e2h3bcast_A_gemma2                                    4      209
run_20261026_e2h3bcast_A_llama                                     4      220
run_20261026_e2h3bcast_analysis_A18                                2       12
run_20261026_e2h3bcast_analysis_A18_leakfree                       2       12
run_20261026_e2h3end_A_gemma2                                      5      448
run_20261026_e2h3end_A_llama                                       4      437
run_20261026_e2h3end_analysis_A18                                  2       24
run_20261026_e2h3end_analysis_A18_failed_nogemma                   0        0
run_20261026_e2h3end_analysis_A18_leakfree                         2       24
run_20261026_e2h3end_analysis_A18_leakfree_failed_nogemma          0        0
run_20261026_e2h_analysis_A15                                      2       37
run_20261026_e2h_analysis_A15_regress                              2       40
run_20261026_e2hfilter_gemma2                                      2      257
run_20261026_e2hfilter_llama                                       2      235
run_20261026_e2hfull_A_gemma2                                      4      112
run_20261026_e2hfull_A_llama                                       4      224
run_20261026_p3overlap_v2_perm10k_A17_gemma2                       4      125
run_20261026_p3overlap_v2_perm10k_A17_llama                        4      121
run_20261026_seeds_A13a_A17                                        2       21
run_20261026_seeds_covariates_A13b_A17                             2        4
run_20261028_cluster_A18b                                          1        7
run_20261029_cluster_A19b                                          1        7
run_20261029_e2b_A_llama_base                                      4      604
run_20261029_e2b_analysis_A19                                      2       22
run_20261029_e2b_analysis_A19_leakfree                             2       22
run_20261029_e2bfilter_llama_base                                  2      187
run_20261029_e2hb_A_llama_base                                     4      434
run_20261029_e2hb_analysis_A19                                     2       12
run_20261029_e2hb_analysis_A19_leakfree                            2       12
run_20261029_e2hbfilter_llama_base                                 2      241
run_20261029_editpool_A20                                          1        1
run_20261029_p4a20_llama                                           2       46
run_20261029_split_indnoise_A22                                    2        2
run_20261029_split_indnoise_A22_leakfree                           2        2
run_20261030_e2a23_analysis_leakfreer033342a1                      2       22
run_20261030_e2a23_analysis_leakfreer190223a1                      2       22
run_20261030_e2a23_analysisr033342a1                               2       22
run_20261030_e2a23_analysisr190223a1                               2       22
run_20261030_e2a23_Ar031521a1_llama                                4      239
run_20261030_e2a23_Ar184307a1_llama                                4      239
run_20261030_e2a23filterr031350a1_llama                            2       54
run_20261030_e2a23filterr184137a1_llama                            2       54
run_20261030_e2ha23_analysis_leakfreer034132a1                     2        9
run_20261030_e2ha23_analysis_leakfreer191214a1                     2        9
run_20261030_e2ha23_analysisr034132a1                              2       10
run_20261030_e2ha23_analysisr191214a1                              2       10
run_20261030_e2ha23_Ar033206a1_llama                               4      112
run_20261030_e2ha23_Ar190223a1_llama                               4      112
run_20261030_e2ha23filterr031436a1_llama                           2       68
run_20261030_e2ha23filterr184222a1_llama                           2       68
run_20261030_key_similarity_A26                                    1        3
run_20261030_key_similarity_A26_firstpass                          1        3
run_20261030_keymode_analysis_A24                                  1        7
run_20261030_p3trace_v2r184052a1_llama                           281     1808
run_20261030_p4a21r023249a1_llama                                  2       59
run_20261030_p4a24r023334a1_llama                                  2      128
run_20261030_p4a28baser201036a1_llama                              2      170
run_20261030_p4a28layers0r220635a1_llama                           2      213
run_20261030_p4a28layers1r225418a1_llama                           2      214
run_20261030_p4a28sharedr200951a1_llama                            2      369
run_20261030_para_analysis_A21                                     1        2
run_20261030_para_analysis_A21_firstpass                           1        2
run_20261031_key_similarity_A28                                    1        3
run_20261031_keymode_analysis_A28                                  1        7
run_20261031_layer_map_A28                                         1        4
run_20261031_p4a30drybase_gemma2                                   2        1
run_20261031_p4a30dryidentical_gemma2                              2        4
run_20261031_p4a30dryshared_gemma2                                 2       13
run_20261031_para_analysis_A28                                     1        2
```

## 9.9 Git history (generated)

133 commits at the time of building. Newest first.

```text
fb7f10f  2026-11-01  docs: master record (goal, story, problems faced, user's words, paths, paper, pre-registrations, logs) as one PDF, with its build script
3c065c4  2026-11-01  docs: paper Section 4.8 (Kaggle results), pre-registrations through the Gemma failure fix, goal file, manifest (2126 claims verified)
3a86f6a  2026-11-01  data: Kaggle and laptop run outputs (Llama A28 run, first pass, Gemma dry runs, console notebooks, figures)
1800532  2026-11-01  chore: self-contained Kaggle notebooks (Llama bundle-4c26396318b4, Gemma bundle-e186cf5d5e1b), model config packs, non-geography fact files
53d0f55  2026-11-01  feat: Kaggle pipeline for Llama and Gemma-2, key-similarity and layer-map analyses; joint edit survives identical keys
09ee9ad  2026-10-29  Kaggle runner: always replace code/tests/kaggle with the fetched commit (a second run in one session kept the first run's old files); test added
b8d3fc9  2026-10-29  Tests: layout test builds its own checkout so the suite also passes in the inputs-only Kaggle layout (30 pass there)
d8b1982  2026-10-29  Kaggle: script that writes git-ignored notebooks with the token pre-filled; private folder excluded from every bundle
36fd0aa  2026-10-29  Kaggle notebooks: optional pasted Hugging Face token (HF_TOKEN_HERE), used before the secret
c57e2c5  2026-10-29  Kaggle runner: try other spellings of the token secret label, no retry on a label that is not attached; tests 31
d001b22  2026-10-29  Docs: Kaggle per-model layout, contribution (mechanism + fix), next steps
c50276d  2026-10-29  Kaggle: one self-contained notebook per model (Llama-Instruct, Gemma-2); no output chaining
3a83e80  2026-10-29  Kaggle: notebooks fetch the code from GitHub at start (no dataset to upload); WikiText fetched from Hugging Face; inputs-only layout shared with the optional zip; tests 30
9c8f799  2026-10-29  Kaggle runner: fix package check (transformer_lens has no __version__), read HF_TOKEN first and export it to every job, stagger GPU starts; tests 29
7e305fe  2026-10-29  Kaggle notebook: find the runner at any depth (or the unpacked zip), print what is attached if missing
d7f3e75  2026-10-29  Kaggle from-scratch rerun: step-based runner, per-model bf16/fp16 check, inputs-only dataset, step-1 notebook (fact preparation); A25 amended; tests 26/26
e0f725c  2026-10-29  Kaggle runner: one driver, both GPUs (sharded jobs), bf16/fp16 check, retries/resume, merge+analyses+packaging; A25 pre-registered; tests 21/21
afc6f3a  2026-10-29  Kaggle notebook: install first, leave numpy/scipy/torch alone
37c5f30  2026-10-29  Kaggle setup: notebook, run scripts, bundle builder, setup guide (bundle zip not committed)
0a29dba  2026-10-29  CPU pass: A22 reconciliation, paper repairs, A23/A24 pre-registration and code, audit sheet, GOAL.md (1941 claims verified)
f36b869  2026-10-29  A21 pre-registration + code: English-paraphrase control for edit transfer (--para, para_analysis.py, scripts_a21.sh)
2d22cfb  2026-10-29  A20 pooled edit stats into manifest (1826 claims); paper: A19 base replication, A20 edits
5ce4c7d  2026-10-29  A19 base-model replication and A20 edit results, manifest
17ab48d  2026-10-29  Remove track.bat
f2febc6  2026-10-29  track.bat: live hardware + progress tracker
5e46641  2026-10-29  A20 pre-registration + driver: 20 more Llama edit facts, queued behind A19
a944bf0  2026-10-29  A19 driver: remove rest between GPU steps
83020de  2026-10-29  A19 driver: 10-min rest between GPU steps, 06:30 cutoff
1edb013  2026-10-29  docs: add status, one-paragraph result and figure to README
8d3b477  2026-10-28  Paper: fix leak-free change magnitude
7a3155a  2026-10-28  Paper: add A18 span-wise Hindi interchange, abstract interchange sentences, scope
781d4a2  2026-10-28  A18b clustering sensitivity (all SHARED); A19 pre-registration, llama_base model key, scripts_a19.sh
9888d2e  2026-10-27  A18: EN<->HI span-wise interchange results, manifest 1565, checkpoint retry fix
03e79bc  2026-10-26  A18 script: skip finished steps on resume
704818a  2026-10-26  A18: pre-register EN->HI span-wise interchange; span flag + script
ec7abfb  2026-10-26  Add Q1 roadmap
e489914  2026-10-26  A17b: clustering sensitivity results; EN-ES robust, Hindi labels not; manifest 1469
8731a6b  2026-10-26  A17b: pre-register clustering sensitivity of interchange verdicts
948f81c  2026-10-26  A17: leakage-excluded re-analysis results (no verdict changes), manifest 1289
95ac71b  2026-10-26  A17: pre-register answer-in-subject leakage re-analysis; add leakage.py and --exclude-leak flags
cee698c  2026-10-26  A16 Hindi->English interchange: Llama PARTIAL 0.74, Gemma SHARED 0.76 (floor met); manifest 1209 verified; paper 4.7 + PDF
70e3908  2026-10-26  Register A16 (EN-HI interchange, HI->EN, larger fact set) before any A16 data; code/script ready; A15 analysis regression: max diff 0
2066586  2026-10-26  A15 Hindi interchange: underpowered (23/12 source entities < 25 floor; EN->HI gate failed); manifest 1153 verified; paper 4.7 + PDF
4082b9a  2026-10-25  Paper v2: rebuild LaTeX/PDF with Gemma F7 and interchange tables; log
3684885  2026-10-25  Register A15 (EN-HI interchange) before any Hindi E2 data; generalise interchange/e2_analysis to a language pair (A13 regression: max diff 0); e2h filter + script
7d9f7f2  2026-10-25  Pilot data files (Llama-1B, BLOOM-1b7); Software Heritage snapshot id
01642bc  2026-10-25  Gemma F7 edits (n=19-20), Llama-1B/BLOOM pilots below 50 facts; manifest 1077 claims; F7 random-layer clarification
03c20d9  2026-10-25  README v2 headline, reproducibility log, Software Heritage save request
250b13c  2026-10-25  reproduce.sh: quote PY (paths with spaces)
a669a94  2026-10-25  reproduce.sh cpu mode + compare_results.py; CPU reproduction of A12a/A13a/A13b/E2/A14 matches saved results exactly
d50aba2  2026-10-25  Related work: verify 23 arXiv refs, add Dumas/Zhang-Nanda/Heimersheim-Nanda, fix Llama3 first author; GPU night2 script
0f49e61  2026-10-25  A14: overlap rerun with 10,000 permutations (CPU), manifest 1059 claims, paper section 6 note
a0b9a43  2026-10-25  Diagnose Llama in-context English 0/10 (model corrects counterfactual); update paper 4.5 and log
9b30704  2026-10-25  Paper: add section 4.7 (noise re-trace, covariates, interchange), notation and citations; update NEXT
2c11131  2026-10-25  E2 interchange results (A13): P2 SHARED in Llama and Gemma; manifest 1011 claims
b5376ef  2026-10-24  E2 analysis script
8fcafdb  2026-10-24  Tracker for E2
e808253  2026-10-24  E2 runner; Llama unit tests and pilot; Gemma selected templates
37d434c  2026-10-24  A13c: per-model template selection for Gemma E2; interchange code
5f9023d  2026-10-24  E2 filter script (EN+ES correctness with prefix templates)
2d85d76  2026-10-24  A13b results; register A13 (cross-language subject interchange)
5306ea6  2026-10-24  Register A13b (covariate model on independent-noise profiles)
1a2c32c  2026-10-24  A13a results: alignment survives independent noise in both models; manifest
4c18e7b  2026-10-24  Live tracker for the A13a night run
9ae823a  2026-10-24  A13a analysis script and night runner
0b01fdc  2026-10-24  A12/A12a results (CPU), manifest 719 claims, A13a pre-registration, independent-noise tracer
d4d717f  2026-10-24  Register A12a (confound-adjusted overlap; NC profiles, prefix classes, fixed SESOI) before any A12/A12a number is computed
9da82fd  2026-10-24  Add external review (2026-10-23) and A12a draft amendment
f446fdd  2026-10-21  Pre-register A12 (confound-adjusted overlap on cached traces); add improvement prompt
eae98dd  2026-10-21  Fix leftover overclaiming sentence in LaTeX build
aa7bd96  2026-10-21  A11 controls on cached traces, section 4.6, softened claims, manifest 589 claims
23f675b  2026-10-21  A11: controls for the EN-ES overlap (pre-registered before computing)
317d12c  2026-10-21  Future-work file and reviewer prompt
e74feb9  2026-10-21  LaTeX version of the paper (main.tex + PDF), build script, figure 2 legend fix
7e70164  2026-10-20  Results for rewording (H8), non-copyable relations, in-context baseline, Llama follow-up edits
5f8e662  2026-10-19  Section 4.4 (layer sweep), H7 verdict, queue2 restartable runner
835c277  2026-10-19  Full overlap tables; manifest covers Gemma stratified (390 claims)
0f6aed4  2026-10-19  paper v2 draft: related work and references merged
840dd04  2026-10-19  select_templates: separate run dir when --relations is used
caf3286  2026-10-19  Queue BLOOM-1b7 chain; update NEXT and LOG
72c97db  2026-10-19  Pre-register A10 (BLOOM-1b7) and register the model
bb66a18  2026-10-19  Manifest: add Hindi-diagnostic and layer-distribution claims (387)
907e6dd  2026-10-19  Critical-layer distribution analysis and Figure 4
2730724  2026-10-19  Add F7 layer rule, edit-script options, post-run chain
1d8d404  2026-10-19  Add paper figures, unit tests, reproduce.sh check, sweep watchdog; log outages
7f5ff5c  2026-10-15  Llama-3.2-1B weights verified
ab11b81  2026-10-15  select_templates: skip relations absent from the candidate file
6d1f44e  2026-10-15  F6 pilot chain queued (runs only if >= 50 facts)
a62cda9  2026-10-15  paper_v2_draft (overlap, tracing, Hindi diagnostic) and h8_analysis
01db420  2026-10-15  F3 code and queue; llama1b registered for the third-model pilot
c9b8030  2026-10-15  F4 data and pipeline (non-copyable relations), queued after the sweep with F5 phrasing traces
bdf4324  2026-10-15  F1 (A9) and F2 (Hindi tokenizer diagnostic) done
184d1ef  2026-10-15  A9: stratified null as primary, non-copyable set, in-context baseline, H8 subset, third-model rule; NEXT.md to-do list
6c44b11  2026-10-15  Manifest extended to v2 results (356 claims)
d9f1935  2026-10-15  Cross-model comparison on shared facts (n=33)
b35f0fb  2026-10-15  Layer-sweep runner; sweep debug runs (Llama, Gemma)
4675f4c  2026-10-15  Gemma-2-2B v2 results (n=58): EN-ES overlap replicates, Hindi weak; logged
6f4371f  2026-10-15  A8: extend Gemma candidate pool with 39 events
085c396  2026-10-15  LOG: correct Gemma fast-path gap (max 0.079)
5bb75ab  2026-10-15  Gemma-2-2B first v2 results (n=26, exploratory); per-model template selection support
74c0fa5  2026-10-15  P3v2: Llama full-answer tracing and overlap (n=92), stratified-null sensitivity, results logged
b729266  2026-10-15  eval_sweep: layer-sweep tables, conditions, H7 test
d154dbd  2026-10-15  Related work with verified citations; knowledge_edit_v2 (layer sweep, full-answer scoring); overlap analysis generalised
874239a  2026-10-15  Gemma-2-2B weights verified; causal_trace_v2 (untested) added
a3a2b7e  2026-10-14  P1v2 gate: template selection, normalised matching, 93 facts kept on Llama; span check
905d58b  2026-10-14  A7: template selection and matching normalisation for v2 filter
0a360af  2026-10-14  Study v2 candidates: 408 facts, 5 relations, 3 phrasings per language; A6 matching rule
ca95298  2026-10-14  A5: pre-register Study v2 (second model, new relations, 3 phrasings, full-answer scoring, layer sweep)
11a7db2  2026-10-14  Model-agnostic code (--model llama|gemma2), Gemma soft-cap and post-norm handling, model research report
8ff8b2c  2026-10-13  Stop full reproduction run; record status; keep partial run dirs
ca475ae  2026-10-13  P6: paper draft, README run order and headline results
a8e2c0c  2026-10-13  P6 tooling: manifest (240 claims), reproduce.sh, comparison script
07716fa  2026-10-13  LOG: correct clamp-hit count (10 of 30)
3476e97  2026-10-13  P4/P5 gates: edit at traced layer succeeds 9/10; transfer, specificity, damage tables
cb75129  2026-10-13  P4 edit code: rank-1 ROME edit with weight re-layout to fit 8 GB
3ffc9b8  2026-10-12  A4: fix P4 edit design before running edits
823ce5a  2026-10-12  P3 gate: overlap vs permutation null on 66 facts; EN-ES above null, Hindi pairs not
2a9f60f  2026-10-12  verify weights are byte-identical to Meta's repo
1b5c4ed  2026-10-12  P2 gate: tracing on 10 facts, H3 range not met; A3 defines P3 statistics before computation
21c3b5a  2026-10-12  P1 v2: stricter first-token rule, 289 candidates, 67 facts kept; trace obj_ok flag
c13ed56  2026-10-12  causal_trace: P2 trace code, superseded v1 run; amendment A2 competitor from corrupted run
045a17c  2026-10-12  HYPOTHESIS amendment A1: clarify critical site, noise scale, collapse rule before tracing
451d8f8  2026-10-12  pre-register hypotheses before tracing and overlap
95a3bc9  2026-10-12  NEXT: mark P1 done
6540a84  2026-10-12  P1 gate met: 52 facts correct in EN, ES and HI on Llama-3.2-3B-Instruct
739baf2  2026-10-12  P0 gate met on Llama-3.2-3B-Instruct: local loading, no_processing, logs
9d33283  2026-10-12  download tooling, log model source and provenance caveat
3b847af  2026-10-12  log: HF access status, Gemma download started
518765c  2026-10-12  P0 gate: GPT-2 smoke test passes; re-freeze requirements
52f1769  2026-10-12  P1: candidate facts EN/ES/HI (154), unfiltered
1696e34  2026-10-12  P0: README, NEXT, LOG, smoke test script
745c544  2026-10-12  env: venv deps pinned, dir skeleton
38b3cc9  2026-10-12  init: babel-effect skeleton
```


# Part 10. Logs and plans

Each chapter below is copied unedited. Run codes (A1 to A30) are explained in Part 8.


## 10.1 Master goal and plan file

*Copied word for word from `GOAL.md` when this record was built. To change it, edit that file and rebuild.*

### GOAL: The Babel Effect

Master tracking file. Updated 2026-10-30 (replaces the 2026-10-29 version). Detail lives in `docs/HYPOTHESIS.md` (pre-registrations A1-A28), `docs/EXPERIMENT_LOG.md`, `docs/LOG.md`, `docs/NEXT.md`, `docs/ROADMAP_Q1.md`. Nothing here is committed or pushed until the user says so.

#### 1. Goal
**Win:** the strongest results we can, aimed at a Q1 journal if the results allow, otherwise Q2-Q3 (venue decided later). The work comes first; the paper is written as well as possible after. Plan B if no positive method result appears: a diagnostic paper (layer map, wording gap, null shared-key result).

**Question:** are facts stored in language-specific circuits or in a shared space, and what does the answer mean for editing a fact in one language? Languages: English, Spanish, Hindi.

**Evidence rule (2026-10-30):** all new evidence comes from the Kaggle runs (2x T4, fp16). Results are compared with published work only, never with the earlier laptop runs. A new result is strengthened (controls, more facts, layer map) before it is claimed.

**Acceptance rule (2026-10-31, from the user):** everything handed over is run on Kaggle by the user, and the paper is accepted only if it has novelty. So every notebook and every claim is judged by what it adds that no paper has; the novelty claim is worded as "not found in prior work" until the five closest papers (2310.10378, 2601.17397, 2406.16416, 2504.09378, 2510.10280) are read in full.

**Model rule (2026-10-31, from the user):** run the models the published papers used so our numbers can be compared with theirs like for like, and use them to show our work with more work (baselines ROME, MEMIT and in-context on the same facts, the same benchmark slice and metrics). Whether ours is best is decided by the results and reported as found: today we have a new explanation (key similarity predicts transfer), not a better editor; the one place we could have beaten published editors, the shared-key fix, was null.

#### 2. Where we are
- Kaggle run 1 (Llama-3.2-3B-Instruct, 74 min, 2026-10-30):
  - shared-key edit gives no improvement (paired median gain 0.06-0.08 for Spanish at the traced layer, about 0 elsewhere): Plan B stands
  - loss beyond rewording: Hindi 0.83 (traced layer) and 0.66 (layer 14), Spanish traced 0.37, Spanish layer 14 0.13 (inconclusive)
  - Spanish keeps 0.50 of the English effect at the traced layer and 0.85 at layer 14; Hindi loses most of it
  - non-geography facts: untestable (53 English-Spanish and 13 English-Hindi facts, only 2 relations survived)
- **Extra result (first pass, 28 facts per cell):** at the traced layer, facts whose Spanish or Hindi subject key is closer to the English key transfer more: rho 0.55 (Spanish) and 0.52 (Hindi), Holm p 0.006 and 0.010. At layer 14: -0.06 and 0.02. With the edit layer also controlled: Spanish 0.45 [-0.01, 0.75], Hindi 0.06 [-0.32, 0.57]. Not a claim yet.
- **Next run (A28, registered before its data exist):** one notebook that traces the facts itself, edits about 85 facts at the traced layer and at layer 14 (English key and shared key), maps layers 2/6/10/18/22 on 40 facts, and runs the key-similarity test, wording control, shared-key analysis and graphs. About 6 h on 2x T4 (limit 10.5 h).
- **A28 result (2026-10-31, 86 facts, layer map on 40; verdicts in `docs/HYPOTHESIS.md`):** key similarity predicts which facts carry over at the traced layer: Spanish rho 0.41 [0.20, 0.59], Hindi 0.42 [0.19, 0.62] (Holm p 0.001 each); none at layer 14. Beyond the edit layer both pass narrowly (Spanish 0.26 [0.004, 0.49], Hindi 0.22 [0.011, 0.44]). Within-layer layer map: Spanish supported (0.29 [0.12, 0.44]), Hindi not (0.17 [-0.02, 0.36]). Shared-key edit: gain at most 0.047, null. Wording loss beyond rewording in all four cells (Spanish 0.39 and 0.26, Hindi 0.79 and 0.73). 7 of 8 registered sub-predictions met. One model so far.

#### 3. Plan (status after the Kaggle run)
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

#### 4. Run path (what to run, in order)
1. **Done 2026-10-31:** `C:\Users\Som Praneeth\Desktop\reasearch\babel-effect\kaggle\babel_llama.ipynb` (code id `bundle-f8371a452538`; the run already in progress is `bundle-21a9baef1028`, same science, only two Spanish analysis jobs fail there). Llama-3.2-3B-Instruct.
   - Only input: the Kaggle Model Meta Llama 3.2, Transformers, 3b-instruct (the same as run 1). No dataset: the code, fact files and tokenizer files are packed inside the notebook. Do not attach the old run's Output.
   - Needs: GPU T4 x2 and Internet on (package install, WikiText-2).
   - Download the Output and send it back; then I read the A28 verdicts.
2. Gemma-2-2B notebook: built and CPU-checked 2026-10-31 (`kaggle/babel_gemma2.ipynb`, A29 registered); waiting for the user to run it. Input: Kaggle Model Gemma 2, Transformers, gemma-2-2b.
3. Llama-3.2-3B base notebook (needs its Kaggle Model path).
4. Qwen2.5-3B, then BLOOM-1b7 (pilot kept only 34 facts, needs a larger fact pool).
5. One 7-8B model only after a T4 test (both T4s or 4-bit; untested; 4-bit changes the edited weights).
Each new model is registered in `docs/HYPOTHESIS.md` before its first run.

#### 5. Models
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

#### 6. Published results vs what the next notebook gives
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

#### 7. What decides the paper
- Spanish and Hindi both pass the layer-controlled key-similarity test: key similarity predicts transfer; this is the contribution (Q1 candidate).
- Only Spanish passes: partial claim (Q2).
- Neither passes: the diagnostic paper with the layer map, the wording gap and the null shared-key result (Q2-Q3).

#### 8. Checks done on the next notebook (2026-10-30)
- Full suite: 49 tests pass, 1 skipped; also run on the code unpacked from the notebook exactly as Kaggle assembles it.
- Every job command: script exists, every flag exists in the script, every dependency exists; the folder pattern each job reads is made by a job it depends on (unit test).
- Key-similarity and layer-map scripts rerun on the real Kaggle run-1 data; the layer-map graph tested including undefined and empty layers.
- Fixed: both analysis scripts print results after saving them; a print error would have failed the job and the retry would have hit the write-once folder. Fixed a stale help line.
- Untested on a T4: the tracing job in fp16 (no resume flag, a retry restarts it), the larger scale, the time estimate. The tracing code ran in fp16 on the laptop GPU for 2 facts (7.5 GiB), which is not a T4 test.

#### 9. Standing rules
Pre-register in `docs/HYPOTHESIS.md` before any data exist; write-once run folders; every number through `results/manifest.json`; never touch `..\RRAP`; commit or push only when the user says so; plain names in chat (no run codes); quote paths (they contain spaces). Kaggle notebooks must run with zero errors and keep both GPUs busy; rebuild the notebook (`.venv/Scripts/python kaggle/make_notebook.py`) after every code change.

#### 10. Needed from the user
Run the notebook and send back the Output; the Kaggle Model paths for Gemma-2-2B and Llama-3.2-3B base; someone to check the Hindi and Spanish facts; an OSF/Zenodo account.


## 10.2 Decision log

*Copied word for word from `docs/LOG.md` when this record was built. To change it, edit that file and rebuild.*

### LOG (decisions)

#### 2026-10-12 P0
- Project root: ...\reasearch\babel-effect (git toplevel verified). Sibling ../RRAP is archived: never touched.
- torch installed from the CUDA 12.8 wheel index (RTX 5060 laptop GPU, Blackwell needs cu128; default PyPI wheel is CPU-only on Windows).
- TransformerLens 4.x removed `HookedTransformer` -> pinned `transformer-lens==3.9.0` (needs re-freeze of requirements.txt).
- Model ladder substitutions (reason): Llama-3.2-3B-Instruct: gated, no HF token (GatedRepoError). Gemma-2-2B: gated, same. Pythia-1.4B: download stalled at 1.07 GB (HF throughput ~0.7 MB/s measured; Xet transfer also stalled earlier, so HF_HUB_DISABLE_XET=1 is set). GPT-2 used only as an engineering smoke test of the pipeline; it is English-only and NOT usable for the EN/ES/HI science.
- Estimated download time at ~0.7 MB/s: Llama-3.2-3B bf16 (~6.4 GB) ~2.5 h; Gemma-2-2B (~5 GB) ~2 h. Network speed may vary.
- ACTION (user): create HF account/token, accept Llama + Gemma licenses, run `hf auth login` in an external PowerShell.
- P1 part 1: built 154 candidate facts (65 capital_of, 39 landmark_city, 50 capital_country), EN/ES/HI, hand-written translations, 2 Hindi template variants (variant chosen later by whole-set accuracy, not per fact). Per-fact corruption seed stored (sha256 of id). NOT yet filtered: filtering needs the real model.
- P0 smoke test PASSED on GPT-2 (substitute, ladder rung 4): top-1 " Paris" p=0.0706, then London 0.0585, Amsterdam 0.0485, Berlin 0.0355, New 0.0229; peak VRAM 0.39 GiB (bf16). Pipeline (venv, CUDA, TransformerLens 3.9.0, hooks) verified. Gate is met only for GPT-2; re-run on Llama-3.2-3B / Gemma-2-2B once HF access exists. requirements.txt re-frozen with the transformer-lens pin.
- HF login done by user (oauth token, user praann07). Gemma-2-2B access GRANTED; Llama-3.2-3B-Instruct request PENDING (config download still GatedRepoError). Gemma shards are fp32 (9.7 GB total) and measured throughput is ~0.5 MB/s -> ~5 h. Started resumable background download of Gemma (analysis/_logs/dl_gemma.log). Llama (6 GB) would be faster/better for Hindi; switch to it if approval arrives and the Gemma download is far from done. Both fit 8 GB VRAM in bf16 (Gemma ~5.2 GB weights, tight with caches; plan: chunked caching).
- Download method: the stock Hugging Face downloader (python, `hf` CLI) stalls at 0 bytes on this network; plain curl/requests Range requests work at ~0.5 MB/s. Wrote code/dl_hf.py (chunked 16 MiB Range, resumable). Gemma-2-2B (9.7 GB fp32) abandoned for now (partial files deleted).
- MODEL SOURCE: using `unsloth/Llama-3.2-3B-Instruct` (ungated re-upload; shard sizes 4.62 + 1.36 GB match meta-llama's repo exactly) while Meta's approval is pending. Provenance caveat: weights not byte-verified against Meta's repo; verify checksums once Meta access is granted (compare sha256 of shards via the HF API LFS oid). Llama license terms still apply and the user has submitted the Meta acceptance form.
- ETA for 6.0 GB at 0.53 MB/s: ~3 h. Weights go to models/llama-3.2-3b-instruct/ (gitignored).
- P0 smoke test PASSED on the target model Llama-3.2-3B-Instruct (unsloth mirror, local weights): top-1 " Paris" p=0.6132; peak VRAM 7.49 GiB (bf16). Shard 1 finished by the user on a faster connection (4,965,799,096 B); both shards load (254 tensors).
- Decision: load with `from_pretrained_no_processing` (no LayerNorm folding / weight centering). `from_pretrained` segfaulted (exit 139) after weight loading; no_processing worked and is also the right choice for ROME-style edits, since weights stay unmodified. Local loading: HF model from models/llama-3.2-3b-instruct passed via `hf_model=` and `tokenizer=`; smoke_test.py gained `--local` and `--out`.
- VRAM note: 7.49 GiB peak for a single short forward pass leaves almost no room for caching. Plan: chunk the (layer, token) patching sweeps, free caches per fact, fall back to 4-bit weights + fp16 activations if OOM.
- P1 DONE: filter_facts.py on Llama-3.2-3B-Instruct gives n=52 of 154 (data/facts.json; table in analysis/run_20261012_p1filter/accuracy.md; raw generations in analysis/run_20261012_p1filter/generations.json). Correct = greedy continuation, stripped, starts with the exact object string. Hindi variant 0 (no trailing \"है\") won on whole-relation accuracy for all three relations.
- Caveats to carry into the paper: (1) n=52 is barely above the n>=50 floor, so bootstrap CIs will be wide; (2) landmark_city keeps only 4 facts, so per-relation results must be reported and landmark_city treated as too small to analyse alone; (3) exact-match is strict: some misses are spelling variants (Spanish 'Hanoi' vs 'Hanói', Hindi 'बेर्लिन' vs 'बर्लिन') or template artefacts ('la ciudad de México'), and facts that are known only in EN are excluded, so the kept set is biased toward facts the model knows well in all three languages. Not changed after seeing results.
- P1 v2 (2026-10-12): the first filter let 3 Hindi facts through whose first generated token was a newline (the object came on the next line), which breaks first-token tracing and leaves n=49 (< 50). Fix: only leading spaces are ignored. Candidate pool extended from 154 to 289 (65 more countries/capitals, appended so earlier ids/seeds are unchanged; capital_country now covers all 125 pairs). Result: 67 kept (capital_country 33, capital_of 32, landmark_city 2) in `data/facts.json`; v1 outputs stay in `analysis/run_20261012_p1filter`. Hindi variant by whole-relation accuracy: variant 0 for capital_country and capital_of, variant 1 for landmark_city (0.10 vs 0.08, 2 facts kept, so landmark_city is too small to analyse alone).
- P2 tracing implementation: batch of T rows per (layer, kind) with one patched position per row; score = RMSNorm-normalised last hidden state x norm weight, dot (W_U[:,obj]-W_U[:,comp]) in fp32 (agrees with full-logit path within ~0.1, bf16). Object = clean top-1 token (often only the first subword for ES/HI, e.g. ' Franc', ' फ'); competitor = top non-object token in the corrupted run (A2). Sanity check passes: restoring the final-layer residual at the last position recovers the clean score exactly (gap 0.000 in all pairs). Peak VRAM 7.51 GiB; ~8.7 s per fact-language pair.
- Limitation to carry: the object metric is first-token only, so ES/HI attributions measure the first subword of the object.
- PROVENANCE VERIFIED (2026-10-12): Meta approved access to meta-llama/Llama-3.2-3B-Instruct. Local shards (downloaded from the unsloth mirror) have the same SHA-256 as Meta's LFS objects: shard 1 13cbd6d16e927a0c5bad54102514e6e18b4a47b3a6eb911e39d678d328d19f55, shard 2 7b770216613ac5c34d7c54bdff1fa616bc4e338a9d0b20af6303e48c295ee23c (sizes 4,965,799,096 and 1,459,729,952 B). The weights are byte-identical to Meta's; the mirror caveat above no longer applies. Config/tokenizer files were not hash-compared.
- Note: the `dbg_overlap*` runs in analysis/ are pipeline debugging on the 10 P2 facts (a subset of the primary set); their numbers are not reported and no analysis choice was changed after seeing them.
- P3 RESULT (2026-10-12, run_20261012_p3overlap, primary set n=66 after the collapse/positive-effect filter; fixed statistics from HYPOTHESIS A3):
  - Primary (MLP profile at last subject token, per-fact cosine / Spearman, median over facts, null = 1000 shuffled-fact pairings): EN-ES cosine 0.748 [0.676, 0.801] vs null mean 0.570, z=6.06, p=0.001; Spearman 0.473 [0.395, 0.627] vs 0.309, z=4.84, p=0.001. EN-HI cosine 0.323 [0.106, 0.448] vs 0.304, z=0.39, p=0.35; Spearman 0.071 [-0.015, 0.154] vs 0.098, z=-0.75. ES-HI cosine 0.227 [-0.037, 0.349] vs 0.204, z=0.38; Spearman 0.079 vs 0.061, z=0.49. p-values floor at 1/1001 = 0.001. Of the 6 primary tests, only the 2 EN-ES tests pass Bonferroni (alpha 0.05/6 = 0.0083).
  - Secondary: residual-stream profile EN-ES cosine 0.961 vs null 0.944 (z=2.69), but all residual profiles are high under the null (0.82-0.94), i.e. a generic shared layer profile dominates. Top-k IoU: EN-ES z=0.77 (k=5), 2.73 (k=10), 2.11 (k=20); EN-HI and ES-HI |z| < 1.3 for every k.
  - Sensitivity: same conclusions on all 67 kept facts, without landmark_city (n=64), and within capital_country (n=33). Within capital_of (n=31) EN-ES cosine stays above null (z=2.10, p=0.006) but Spearman does not (z=1.15), and HI pairs are at or below null.
  - Critical MLP layer (n=66): EN median 2 (IQR 0.25-4), ES 2 (2-5), HI 3 (2-5.75); only 3-6% fall in layers 8-19. |EN-ES| median 2 (61% within 2 layers), |EN-HI| median 2 (58%), |ES-HI| median 2 (53%).
  - Verdict against the pre-registration: H1 holds for EN-ES, not for EN-HI or ES-HI (indistinguishable from the null). H2 holds in direction (EN-ES overlap well above EN-HI). H3: layer range 8-19 NOT met; Spanish within 2 layers of English on median (met, borderline); Hindi at or after English (met, median 3 vs 2). Under the fixed interpretation rules this is the partial/mixed outcome: a shared English-Spanish fact-specific locus, no detectable fact-specific alignment with Hindi.
  - Cautions: absence of Hindi overlap is not evidence of a separate Hindi circuit. Hindi subject-token effects are about half the size of EN/ES, subjects span more tokens, the object metric is the first subword only, and n=66 gives wide intervals; any of these can hide alignment. The shuffled-fact null measures fact-specific alignment above a shared generic profile, so the result does not say the layer profile is unshared. One model, one relation family (capitals).
- P4/P5 RESULT (2026-10-13, run_20261013_p4edit, tables in tables.md/tables.json; 10 edit facts, 30 edits; CIs are bootstrap over facts, n=10 so intervals are wide):
  - EN edit success (top-1 = new object's first token): traced layer 9/10 (90%), mid layer 14 10/10, random layer 10/10. Gate (>= 80% at the traced layer) met. Traced layers used: 0, 2, 3, 4, 5.
  - Transfer (median dP ratio, edits with EN success): traced ES 0.147 [0.000, 0.769], HI 0.002 [0.000, 0.267]; mid ES 0.203 [0.007, 0.691], HI 0.244 [0.010, 1.026]; random ES 0.005, HI 0.005. Top-1 flips to the new object: traced ES 40% / HI 30%, mid ES 40% / HI 60%, random ES 10% / HI 30%.
  - Specificity (median over facts of max abs dP over 20 controls): traced 0.045 [0.024, 0.400], mid 0.351 [0.049, 0.886], random 0.061 [0.025, 0.546]; worst single edit 0.981 (traced), 0.946 (mid), 0.917 (random). Damage (perplexity ratio on 100 WikiText-2 test sentences): median 1.0009 / 1.0021 / 1.0000; worst 1.0082 / 1.0165 / 1.0015.
  - Against pre-registration: H4 EN success met. H4 ordering 0 < HI < ES < 1 holds for the traced-layer edit (HI median 0.002 is only just above 0) and fails for the mid-layer edit (HI 0.244 >= ES 0.203, CIs overlap fully). Random-layer transfer is lowest, as predicted. H5: damage < 1.05 met everywhere; specificity < 0.05 met only by the traced-layer median, with single edits breaking unrelated facts badly (up to 0.98), so the edit is not specific in the worst case.
  - Observations, not pre-registered: the mid-layer edit moves HI more than the traced early-layer edit, but with much worse specificity. The delta norm often reaches the clamp (4 x |v0|; 10 of 30 edits at ratio >= 3.9: 6 traced, 4 random, 0 mid), so some edits are partly limited by the clamp. Layer-0 edits were not reproducible across two runs of the same code (EN dP 0.418 vs 0.939 for capital_country_000), so layer-0 numbers carry extra run-to-run uncertainty; the final run is reported as is.
  - Implementation notes: TransformerLens copies the weight matrices on every forward pass, which autograd keeps alive (about 1.2 GiB per pass) and spilled the first debug run to 14.5 GiB (shared system memory, violating the 8 GB budget). Fixed by re-laying out the weights (common.share_transposed_mlp_weights; logits identical, max abs diff 0.0), which brings peak VRAM to 7.58 GiB and a forward+backward to 0.2 s. A failed debug run directory (analysis/run_20261012_dbg_edit, config only, crashed before any output) was deleted by me; no result file was removed.
- REPRODUCTION STATUS (2026-10-13): `./reproduce.sh full repro` was stopped on the user's instruction during the tracing stage (171 of 201 fact-language grids written in analysis/run_20261013_p3trace_repro; the run has no records.json and is incomplete). Stage P1 reproduced exactly: the rerun filter keeps the same 67 facts (data/repro/facts.json identical to data/facts.json, see analysis/run_20261013_p1filter_repro). Overlap, editing and transfer were not re-run, so the P6 gate "reproduce.sh passes" is only met for `./reproduce.sh verify` (240/240 claims match the saved artifacts), not for the full rerun. The partial run directories are kept as raw outputs.
- MODEL-AGNOSTIC CODE (2026-10-14): scripts now take `--model llama|gemma2` (registry in code/common.py MODELS). Changes: Scorer applies the final-logit soft cap (Gemma-2: 30*tanh(x/30), no-op for Llama); knowledge_edit.py adds delta to the input of `ln2_post` for models with a post-MLP norm (Gemma-2: hook_mlp_out fires after ln2_post, so a W_out edit must be specified before that norm); mid layer = n_layers//2; weight re-layout handles models with full-attention parameter names; covariance cache is keyed by model. Regression check on Llama (capital_country_000, 3 languages): critical layers identical (0/5/2) and scores equal to the earlier run within bf16 noise (analysis/run_20261014_regress_llama). Gemma path NOT yet run.
- Model research report: reports/Models for multilingual fact editing.md (notes in research_notes/). Gemma-2-2B download resumed by the user after an interruption (shard 3 complete, shard 2 in progress at the time of writing).
- STUDY V2 DATA (2026-10-14): code/build_facts_v2.py writes data/v2/facts_candidates.json: 408 candidates, 5 relations, 3 phrasings per fact and language (0 declarative canonical, 1 variant, 2 QA-style). Added 42 landmarks, 40 inventions (with alternative name forms), 37 historical events (distinct years). Known flaws: some Spanish plural subjects take a singular verb ("Los Museos Vaticanos se encuentra"); Hindi spellings of foreign names vary, so many Hindi misses will be spelling variants (alt forms listed for some); years are digits identical across languages, so event_year can be partly copied across languages. v1 candidate and fact files are untouched.
- P1v2 RESULT (Llama-3.2-3B-Instruct, 2026-10-14): first v2 filter (run_20261014_p1v2_llama, original templates, exact-prefix on every phrasing) kept 38 of 408; after A7 (template selection, normalised matching, lenient paraphrases) the rerun (run_20261014_p1v2b_llama) kept 93: capital_of 27, capital_country 38, landmark_city 6, invention_inventor 0, event_year 22. Accuracy of the canonical phrasing: EN 0.72-1.00, ES 0.60-0.95, HI 0.03-0.68 depending on relation (data/v2 + accuracy.md in the run dir). Findings: Llama-3.2-3B cannot name Hindi inventors (canonical accuracy 0.03) or landmark cities except for a few (0.21), so those two relations cannot be analysed in Hindi; the model-limits finding itself is reportable. event_year facts have identical digit answers across languages, which may inflate transfer; they are analysed with and without (A5).
- SPAN CHECK (P1 gate): code/check_spans.py finds a contiguous subject span for every kept fact, language and phrasing (0 errors, data/v2/span_check_llama.json). Subject length differs by language (median 2 / 3 / 5 tokens EN / ES / HI; 61 of 93 facts differ by more than 2 tokens from EN). Design response: all statistics use each language's own last subject token (profile comparison) or sum over the subject span (grid cells), so no position-wise alignment between languages is needed; proportional indexing is not used.
- Process note: template selection and the lenient paraphrase rule were chosen after seeing the first filter's raw generations (A7); they use correctness only, not any tracing, overlap or edit result.
- GEMMA-2-2B WEIGHTS VERIFIED (2026-10-15): downloaded by the user with code/dl_hf.py from google/gemma-2-2b (official repo, access granted). Shard SHA-256 equal Hugging Face's LFS hashes: 00001 1425aa06...193d, 00002 96c111d3...0d35, 00003 4e08abc6...5f02 (sizes 4,992,576,136 / 4,983,443,424 / 481,381,384 B; list in analysis/_logs/gemma_lfs.txt). Not yet loaded or run; GPU runs are held pending orchestrator sign-off after the circuit-breaker notices.
- P3v2 RESULT (Llama-3.2-3B-Instruct, full-answer attribution, n=92 valid facts of 5 relations; run_20261015_p3trace_v2_llama, run_20261015_p3overlap_v2_llama):
  - Pooled shuffled-fact null (primary test of A5): MLP-profile cosine / Spearman, median over facts [95% CI], z. EN-ES 0.627 [0.557, 0.702] z=4.22 / 0.461 [0.428, 0.531] z=5.15; EN-HI 0.509 [0.335, 0.594] z=2.45 / 0.323 [0.233, 0.407] z=2.61; ES-HI 0.423 [0.313, 0.529] z=2.47 / 0.283 [0.213, 0.317] z=3.08. p: 0.001, 0.001, 0.003, 0.007, 0.006, 0.001, so all six pass Bonferroni (0.0083). Null means 0.519, 0.429, 0.328 (cosine). Unchanged on all 93 facts and on the 70 facts without event_year and invention (z = 3.85, 2.13, 3.51).
  - Caution that changes the interpretation: the pooled null shuffles facts across relations, so part of the significance can come from relation-level similarity (capital-type profiles resemble each other) rather than fact-specific alignment. Post-hoc sensitivity (added after seeing the pooled result, flagged as such): permutations restricted to swaps within the same relation give cosine z = 3.29 (EN-ES), 1.79 (EN-HI, p=0.050), 1.98 (ES-HI, p=0.020) and Spearman z = 3.82, 2.22 (p=0.018), 2.38 (p=0.012). Under Bonferroni only EN-ES survives. Per relation with its own null (cosine z, EN-ES / EN-HI / ES-HI): capital_country 2.75 / 1.33 / 2.19; capital_of 2.08 / 0.30 / 2.55; event_year 0.91 / -1.12 / 0.10. landmark_city (6 facts) and invention_inventor (0) are too small.
  - Critical MLP layer medians: EN 3, ES 4, HI 4; 3%, 2%, 11% fall in layers 8-19; median |layer difference| 2 for every pair (70%, 58%, 57% within 2).
  - Against the pre-registration: H6 holds for Llama (EN-ES z>2; EN-ES overlap above EN-HI overlap in value, 0.627 vs 0.509, CIs overlap). H9 (EN-ES z>2 in >= 2 of the 3 new or extended relations) is not supported: only event_year is testable and its EN-ES z is 0.91. H3's 8-19 peak range is again not met.
  - Difference from v1 (first-token metric, n=66): v1 found Hindi pairs at the null (z ~0.4); v2 finds them above the pooled null but not robustly within relation. Changed in v2: full-answer metric, 5 relations, n=92, 3 phrasings in the filter. The v1 Hindi null should be read as sample- and metric-dependent, not as evidence of separate Hindi circuits.
- GEMMA-2-2B FIRST RESULTS (2026-10-15, n=26 valid facts, below the n >= 50 requirement, so exploratory only; runs run_20261015_p1v2_gemma2, run_20261015_p3trace_v2_gemma2, run_20261015_p3overlap_v2_gemma2): the Gemma code path works (fast-path vs model log-prob gap at most 0.079 over all 78 pairs; one pair is not a teacher-forced top-1 match, soft cap handled, peak VRAM 7.15 GiB). MLP-profile cosine, median [95% CI], z vs pooled shuffled-fact null: EN-ES 0.419 [0.136, 0.630] z=2.56 (Spearman z=4.15, p=0.001); EN-HI 0.063 [-0.060, 0.168] z=-0.27; ES-HI 0.072 [-0.016, 0.244] z=-0.08. Critical MLP layer medians EN 6, ES 5.5, HI 6 (31-38% in layers 8-19; 26 layers), |EN-ES layer difference| median 4. Same qualitative pattern as the v1 Llama result (EN-ES shared, Hindi at the null) and unlike the v2 Llama result (Hindi weakly above the null), with the caveat that n=26 gives wide intervals and Hindi accuracy is very low (canonical 0.00-0.49), so only the 26 facts Gemma knows in Hindi are included. Peak locations differ from Llama (Gemma mid-network, Llama layers 2-4), consistent with the model-dependence flagged in the research report.
- Candidate-prompt effect: the Llama-selected templates lowered Gemma's canonical accuracy for some relations (e.g. capital_of EN 0.50 against 0.97 for phrasing 2), so per-model template selection (select_templates.py --model gemma2 writes data/v2/templates_gemma2.json) is run before concluding that Gemma cannot reach n >= 50.
- GEMMA-2-2B RESULT (2026-10-15; own selected templates, pool extended by 39 events per A8; n=58 valid facts: event_year 30, capital_of 11, capital_country 11, landmark_city 6; runs run_20261015_p1v2c_gemma2, run_20261015_p3trace_v2c_gemma2, run_20261015_p3overlap_v2c_gemma2):
  - Pooled shuffled-fact null, MLP-profile cosine, median [95% CI], z (Spearman z, p): EN-ES 0.464 [0.329, 0.542] z=4.81 (z=4.55, p=0.001); EN-HI 0.226 [0.168, 0.283] z=2.19, p=0.010 (z=1.35, p=0.087); ES-HI 0.137 [0.014, 0.267] z=0.08 (z=-0.48). Null means 0.292, 0.136, 0.134. Bonferroni (0.0083): only EN-ES (both statistics) passes.
  - Excluding the copyable relations (event_year; n=28): EN-ES z=2.51 (Spearman 2.51), EN-HI z=1.48, ES-HI z=0.22. Relation-stratified null (post-hoc): EN-ES z=3.85, EN-HI 1.68, ES-HI -0.45. Per relation the samples are 6-30 facts and mostly non-significant (capital_country n=11: EN-ES z=-0.74; capital_of n=11: EN-ES cosine z=1.14, Spearman z=2.34; event_year n=30: EN-ES z=2.07 / 3.91).
  - Critical MLP layer medians EN 6, ES 6, HI 6 (43%, 31%, 26% in layers 8-19; 26 layers); median |layer difference| 4 (EN-ES), 4 (EN-HI), 2.5 (ES-HI).
  - Against the pre-registration: H6 holds for Gemma (EN-ES z>2; EN-ES overlap above EN-HI overlap). The Gemma peak sits mid-network (layers ~6), unlike Llama (layers 2-4), so layer location is model-dependent while the cross-language pattern (EN-ES shared; Hindi weaker) is common to both. Caveats: Gemma's set is dominated by event_year (30 of 58) whose digit answers are identical across languages; Gemma is a base model and Llama an Instruct model; the two models' fact sets differ (Llama 92, Gemma 58), overlap of the sets not yet analysed.
- CROSS-MODEL (2026-10-15, code/compare_models.py, run_20261015_compare_models): the two models share 33 valid facts (capital_country 8, capital_of 6, event_year 16, landmark_city 3). On those 33 facts with the pooled shuffled-fact null, MLP-profile cosine median [95% CI], z: Llama EN-ES 0.544 [0.438, 0.601] z=1.88; EN-HI 0.310 [0.076, 0.552] z=-0.21; ES-HI 0.387 [0.106, 0.473] z=2.53. Gemma EN-ES 0.545 [0.354, 0.735] z=4.09; EN-HI 0.222 [0.162, 0.314] z=0.55; ES-HI 0.217 [0.041, 0.296] z=0.57. So on identical facts EN-ES is above the null in both models (Llama weaker, p=0.022, n=33), EN-HI is at the null in both, and ES-HI differs (above null in Llama, at null in Gemma). Half of the shared set is event_year, so this comparison is dominated by that relation.
- F2 HINDI TOKENIZER DIAGNOSTIC (2026-10-15, code/hindi_diagnostic.py, run_20261015_hindi_diag; exploratory, not pre-registered):
  - Accuracy by Hindi object token count (canonical phrasing, all candidates): Llama 0.42 (1-3 tokens, n=127), 0.40 (4-6, n=216), 0.23 (7+, n=65); Gemma 0.21, 0.21, 0.02 (n=137, 263, 47). Spearman of correct vs object token count: Hindi -0.15 (Llama), -0.07 (Gemma); English -0.21 and -0.10, so object length alone is not Hindi-specific; only very long Hindi answers (7+ tokens) collapse.
  - Per-fact EN-HI profile cosine vs Hindi subject span length (valid facts): Spearman -0.35 (Llama, n=92) and -0.39 (Gemma, n=58); vs Hindi target tokens +0.16 and -0.34. Median Hindi subject span 5 (Llama) and 8 (Gemma) tokens vs EN median 2.
  - Overlap split at the median Hindi span (Llama <= 5 tokens, n=49; Gemma <= 8, n=32). Cosine z, pooled / stratified null, Llama short span: EN-ES 3.17 / 3.23, EN-HI 1.69 / 1.54, ES-HI 3.14 / 2.70 (median cos EN-HI 0.597); long span (n=43): EN-ES 2.44 / 1.70, EN-HI -1.22 / -1.25, ES-HI 0.89 / 0.90 (cos EN-HI 0.199). Gemma short span: EN-ES 3.12 / 2.42, EN-HI 1.03 / 1.06, ES-HI 0.41 / 0.71; long (n=26): EN-ES 2.09 / 1.86, EN-HI 1.90 / 1.77, ES-HI -0.36 / -0.37.
  - Reading: the Hindi overlap is clearly weaker when the Hindi subject spans many tokens (the last subject token is then one fragment of a long name), so a tokenizer/span-length effect is part of the weak Hindi result. It does not explain it away: with short Hindi spans EN-HI is still weaker than EN-ES in both models and not significant after correction. The paper must say "weak and span-length dependent" rather than "absent" or "separate circuit".
- LLAMA-3.2-1B-INSTRUCT WEIGHTS VERIFIED (2026-10-15): model.safetensors (2,471,645,608 B) SHA-256 1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f equals the official Hugging Face LFS hash. Registered as `llama1b`; the pilot (F6) is queued behind the sweeps and runs the full pipeline only if it keeps >= 50 facts (A9).
- OUTAGES AND OVERNIGHT SETUP (2026-10-19): the Llama layer sweep died twice. (1) 2026-10-15 20:17 reboot/sleep at 39 of 280 rows; (2) 2026-10-18 16:05, the laptop entered Modern Standby and the sweep process died at 69 rows (Kernel-Power events show standby entries 23:53 and exits 00:59; the keep-awake guard was no longer running). AC sleep timeout was already 0; no power setting was changed. Resumed from the saved rows (`--resume`, no raw output touched) at 03:07 on 2026-10-19 with `keep_awake.ps1` (SetThreadExecutionState) and a new `scripts_watchdog.sh` that relaunches the sweep chain with `RESUME_LLAMA`/`RESUME_GEMMA` if it dies before ALL_DONE. The extra / in-context / Llama-1B chains still wait on the sweep chain's ALL_DONE and are not restarted by the watchdog (their template-selection step is write-once per day). Cost: about 3 days of wall time.
- CPU-ONLY ADDITIONS (2026-10-19): `code/make_figures.py` -> `paper/figures/` (fig1 per-layer MLP profiles, fig2 overlap vs null, fig3 Hindi accuracy by answer length; sources in sources.json); `tests/test_core.py` (13 unit tests for matching rules and overlap statistics, all pass); `./reproduce.sh check` = tests + manifest verify (356/356) + figures. Not yet done: a full end-to-end v2 rerun from a clean clone (F9).
- F7 AND H8 AUTOMATION (2026-10-19, before any F7/H8 data exists): `code/select_f7_layers.py` implements the A9 layer rule on the finished sweep: layers whose median specificity loss is below the median of the per-layer medians are eligible; choose the highest EN success rate (implementation tie-breaks, fixed now: lower median specificity loss, then lower layer index). `knowledge_edit_v2.py` gained `--skip-first N` (the first 10 edit facts are already in the sweep) and `--add-traced` (add each fact's own EN traced layer and L//2). `scripts_run_post.sh` waits for the extra chain, runs `h8_analysis.py` for both models, then after the whole queue (llama1b chain ALL_DONE) runs F7 with 30 facts (20 new) per model and `eval_sweep.py` on it. Defaults of the edit script are unchanged, so the running sweep and its resumes are unaffected.
- CRITICAL-LAYER DISTRIBUTION (2026-10-19, code/layer_hist.py, run_20261019_layer_hist; exploratory, written after the Gemma median profile showed a second peak near layer 14): Llama's critical MLP layer is concentrated at layers 0-6 for all three languages (EN 3% / ES 2% / HI 11% of facts in layers 8-19). Gemma's is multimodal: modes at layers 4, 6 and 14. Layer 14 holds the argmax for 17 of 58 facts in EN and 13 of 58 in ES but only 2 of 58 in HI (HI is spread over layers 0-9). The layer-14 cluster appears in capital_country, capital_of and event_year (EN counts 3, 5, 9), so it is not one relation. The median (6) hides this; the paper reports the histogram (paper/figures/fig4_layer_hist.png). Reading: in Gemma a late-middle MLP step is shared by EN and ES but rarely carries Hindi; the cause is not tested here (Hindi subject span of 8 tokens in Gemma is a candidate).
- BLOOM-1b7 ADDED AS THIRD FAMILY (2026-10-19): `models/bloom-1b7` downloading with code/dl_hf.py (model.safetensors, expected SHA-256 145ae4b66381746c9438c63b6deb22a34d97000bba633bb6672fff2d1dcaf924 from the HF API listing; verified inside scripts_run_bloom.sh before use); registered as `bloom1b7` in code/common.py; `load_model` gained a `device` argument (default unchanged). Pre-registration A10 written and committed before any BLOOM run. Not yet tested: the TransformerLens BLOOM load path and the tracing/editing hooks on this model (the GPU is occupied by the sweep and free RAM is 3.6 GB); the first failure in the chain is expected to be a TransformerLens/BOS compatibility problem and will be debugged when the GPU is free.
- LAYER SWEEP RESULTS (2026-10-19, run_20261015_p4sweep_llama 280 edits, run_20261019_p4sweep_gemma2 260 edits; 10 facts x every layer, English-only edit): EN success 0.84 (Llama) / 0.74 (Gemma) over all layers, 0.53 / 0.20 in the last three layers; ES success 0.41 / 0.48, HI 0.18 / 0.20 (max per layer 0.5). A5 conditions (traced / L//2 / random): EN success 0.9 / 1.0 / 0.78 (Llama), 0.9 / 0.8 / 0.67 (Gemma); median ES transfer ratio 0.66 / 0.52 / 0.42 (Llama), 0.38 / 0.50 / 0.09 (Gemma); HI 0.0001 / 0.010 / 0.007 (Llama), 0.095 / 0.016 / 0.036 (Gemma); specificity 0.030 / 0.084 / 0.059 (Llama), 0.027 / 0.089 / 0.129 (Gemma). H7 (near-mid minus layers 0-5, paired over 10 facts, median [95% CI]): Gemma ES +0.34 [0.01, 0.62], HI +0.07 [0.03, 0.13], specificity +0.04 [0.008, 0.19] (supported); Llama ES -0.31 [-0.47, -0.09] (contradicted), HI -0.013 [-0.095, 0.000], specificity +0.004 [-0.023, 0.035]. Verdict: H7 supported in Gemma, contradicted for ES in Llama; model-dependent, no pooled claim. Numbers are in the manifest (444 claims, verified); text in paper/paper_v2_draft.md section 4.4; figures paper/figures/fig5_sweep_llama.png and fig6_sweep_gemma2.png.
- NEW RUNS READ AND WRITTEN UP (2026-10-20): Llama extras (34 facts; non-copyable n=104 >= 50), Gemma extras (1 fact; non-copyable n=29, F4 not met), H8 supported in both models, in-context baseline (Llama EN-statement fails in English 0/10 but succeeds in ES/HI 10/10; not investigated), Llama F7 (20 new facts). Gemma F7, BLOOM-1b7 and Llama-3.2-1B were stopped by the user for other laptop use; partial Gemma F7 run run_20261020_p4f7_gemma2 is incomplete and kept untouched. Gemma overlap_all first failed on a run-dir name collision (same tag as Llama, write-once rule); retried under a new tag. Manifest now 537 claims verified. Paper sections 4.5 and 5 and the abstract updated. Desktop launchers added (Babel Dashboard, Start/Stop Experiments); Start Experiments.bat did not start the queue and needs a fix.
- EXTERNAL REVIEW AND CONTROLS (2026-10-21): an outside reviewer (another AI, harsh prompt in docs/REVIEW_PROMPT.txt) rated the draft reject for main/Findings/TMLR, workshop plausible, and named the missing same-language/centred/lexical-overlap control as the biggest gap. Pre-registered A11 and ran it on cached traces (no GPU): the alignment survives centring (z 3.61 Llama, 5.43 Gemma) and entity clustering, but in Llama it is concentrated in facts with similar EN/ES subject names (low-similarity half centred z 1.44, n=64; Spearman 0.29 between name similarity and per-fact cosine), and Llama EN-EN different-fact cosine (0.606) exceeds same-fact EN-ES (0.570). Gemma is clean. Paper section 4.6 added; abstract and discussion softened. Review points still to act on: in-context baseline bug check (n=10, English 0/10), calibration of z vs p (1000 -> 10,000 permutations for the main tables), Fig 4 tiny buckets, notation (L//2, A5/H-codes), uncited Dubey/Merity references, missing related work (verify before citing), Llama base vs instruct run, English-paraphrase control for edits, native Hindi audit. Valid review claims NOT verified yet: Table 1 vs Table 2 ES-HI z both 1.98 (Table 1 n=92 within-relation; Table 2 n=126 stratified: different sets, so coincidence is possible).

#### 2026-10-24
- Pushed 7 commits. Received external review (Opus 5.5): shared per-fact noise seed across languages (verified in code), centring unnormalised profiles, A12 S4 coding, Spearman tie bug. Added A12a before computing.
- Ran A12/A12a (CPU). The reviewer's premise that Llama has 46 identical-prefix facts is wrong: prefix-identical facts are rare (Llama 1, Gemma 4) because the ES templates are not subject-first. The equal-span effect is large and is the better lead for the noise artefact.
- Registered A13a (independent-noise re-trace) and wrote code/trace_seeds.py; needs GPU, not run yet.
- A13a run finished 03:12 (Llama 126 facts, Gemma 59, 16 draws each; about 1 h 50 min in total). G1 SURVIVES in both models. My predictions that shared noise inflated the alignment were wrong; shared noise matters only for Gemma equal-span facts. The earlier EQ vs OT gap with the original seed reflected single-draw noise. Next: covariates with the independent-noise profiles (name overlap, answer identity), then E2 (cross-language interchange), which needs its own pre-registration.

- 2026-10-24 ~11:55: E2 full Llama run stopped by user at 210/396 pairs (laptop needed). Nothing saved (script writes only at end). Plan: add checkpoint/resume to code/interchange.py, rerun when GPU is free. Run dir analysis/*_e2full_llama is empty/incomplete and kept (write-once).

- 2026-10-25: E2 finished overnight (Llama 01:54-04:26, Gemma 04:26-07:27, analysis 07:29). Verdict P2 SHARED in both models. Follow-up jobs (Gemma F7, Llama-1B pilot, BLOOM pilot) were skipped by the 06:00 cutoff because E2 ran longer than estimated. P6 fails for Llama (rho -0.04). Manifest now 1011 claims verified.

#### 2026-10-25 (evening) GPU chain finished
- Gemma F7 done (15:42). Llama-1B pilot kept 38 facts, BLOOM-1b7 pilot kept 34 (both < 50, full pipeline not run, A9/A10 rule). Results in docs/EXPERIMENT_LOG.md; manifest 1077 claims verified; paper_v2_draft section 4.5 updated, including a correction: the F7 "random" layer is the A9-selected layer (Llama 21, Gemma 3), not an independent random layer. GPU idle.

#### 2026-10-25 (night) Paper build + A15 registered
- `paper/build_tex.py` updated (edit table now has the Gemma follow-up and labels the follow-up third layer "A9 layer"; new Table 4 for the interchange test read from `analysis/run_20261025_e2_analysis_A13/results.json`); rebuilt `paper/latex/main.pdf` (12 pages, no LaTeX errors) and copied to `paper/The_Babel_Effect_draft_v2.pdf`. Draft note now says only the Hindi interchange (A15) is unrun.
- A15 (EN-HI interchange) registered and committed before any Hindi data (4492a3b); `code/interchange.py` and `code/e2_analysis.py` generalised to `--langs`; A13 results reproduced through the changed analysis code with max abs diff 0 (`analysis/run_20261025_e2_analysis_A13_regress`). GPU run (`scripts_night_e2h.sh`) waits for the laptop.

#### 2026-10-26 A15 Hindi interchange run (02:09-02:54, GPU, user-lent laptop)
Ran `scripts_night_e2h.sh`: filters, full runs (Llama 22 min, Gemma 15 min), analysis. Result: underpowered (23 / 12 source entities < floor 25); EN->HI gate failed in both models; HI->EN site A PARTIAL (I about 0.74), descriptive only. Manifest 1153 claims verified (`e2h_*` keys). Details in `docs/EXPERIMENT_LOG.md`.

#### 2026-10-26 A16 run (12:21-14:04, GPU, user-lent laptop)
`scripts_night_a16.sh` ran in 1h43: Hindi->English interchange on 66 (Llama) / 55 (Gemma) facts, 66 / 54 source entities (floor met). Site A: Llama I 0.74 [0.69, 0.78] PARTIAL; Gemma 0.76 [0.71, 0.81] SHARED (lower bound 0.71). Outside-A15 check within 0.02. Manifest 1209 claims verified. Details in `docs/EXPERIMENT_LOG.md`.

- 2026-10-26 A17 leakage-excluded re-analysis done: no verdict changed (details `docs/EXPERIMENT_LOG.md`); manifest 1289 verified.
- 2026-10-26 A17b clustering sensitivity: EN-ES SHARED under all clusterings; Hindi->English labels not clustering-robust (report as estimate ~0.74-0.80). Manifest 1469.

- 2026-10-27 A18 EN<->HI span-wise interchange done: gate passes both directions; I 0.81-0.93 (Llama 0.86/0.87, Gemma 0.93/0.81), direction-symmetric, higher than the A16 last-token 0.74-0.76. Clustering sensitivity on A18 not yet run. Manifest 1565.

#### 2026-10-29 CPU pass (no GPU used)
- A22 (CPU, stored traces): Llama EN-ES increment present with and without shared subject pieces (0.303 [0.192, 0.423] n = 54 vs 0.397 [0.291, 0.505] n = 49). Replaces the A11 "name overlap explains it" reading for Llama; paper Section 4.6 revised. Manifest 1941 verified.
- Paper: two citations added (2408.10646, 2507.09629, authors verified by the arXiv API), p-floor note, abstract/discussion/limitations updated (A19, A20, A22, native-speaker and benchmark limits), PDF rebuilt (`paper/The_Babel_Effect_draft_v2.pdf`).
- Pre-registered A23 (non-geography relations) and A24 (shared-key edit). Code written and syntax/unit tested on CPU only: `code/build_facts_a23.py` (150 unaudited candidates), `--set a23` in `code/e2_filter.py` and `code/e2_filter_hi.py`, `--key-modes` and `joint_update` in `code/knowledge_edit_v2.py` (2 new unit tests pass, 15/15 total), `code/keymode_analysis.py`, `scripts_a23.sh`, `scripts_a24.sh`. Nothing run on a GPU. `code/select_templates.py` now also knows the A23 template pools (no change for old relations).
- Native-speaker audit sheet: `data/audit/native_speaker_audit.csv` (362 facts used in any analysis) and `AUDIT_INSTRUCTIONS.md`, built by `code/make_audit_sheet.py`. Needs a Hindi and a Spanish checker.
- `GOAL.md` added at the repo root (goal + 8-item plan).
- Kaggle runner (A25): `kaggle/driver.py` does install, checks, model download, bf16/fp16 check, all three runs spread over both T4s with retries/resume, merging, analyses and packaging; the notebook only launches it. Code changes are backward compatible (`BABEL_DTYPE` default bfloat16, `--shard` default 0/1). Tests 21/21 on CPU; nothing run on a GPU yet.
- Kaggle rerun from scratch (A25 amended): one dataset (`kaggle/babel_bundle.zip`, 6.5 MB: code, tests, runner, seed candidate lists, wikitext; no laptop results) and one notebook per step (`kaggle/notebooks/`). Step 1 = fact preparation for Llama-3B-Instruct and Gemma-2-2B. Per-model bf16/fp16 check (`kaggle/dtype_check.py`). Code: every GPU script now honours `BABEL_DTYPE` (default bf16, laptop unchanged); `select_templates.py --tag`; `trace_seeds.py --main-run/--extra-run`. CPU dry run: candidate sets rebuilt exactly (408 / 447 / 156 / 156). GPU paths untested until Kaggle.
- 2026-10-29 Kaggle layout changed again (A25 unchanged): one self-contained notebook per model (`babel_llama`, `babel_gemma2`; Llama base next), nothing passed between notebooks; each notebook fetches the code from GitHub at start (commit id in RUN_SUMMARY.json) so no dataset has to be re-uploaded (`kaggle/babel_bundle.zip` is only a fallback); WikiText-2 fetched from Hugging Face (byte-identical to the laptop copy, same sha256); HF_TOKEN read first and exported to every job; fixed the package check (transformer_lens has no __version__); GPU starts staggered 45 s (CPU memory). Tests 30. Literature check (abstracts only): contribution narrowed to the key-similarity mechanism + shared-key edit vs published multilingual editors; see the plan in the Claude plans folder and docs/NEXT.md.

#### 2026-10-29 (night): Kaggle rerun dropped, laptop only
- Decision by the user: all runs stay on the laptop GPU. The Kaggle code (kaggle/, tests/test_kaggle.py) and the pre-filled token notebooks were deleted from the working tree; they remain in git history up to commit a14edcf. The notebook with the token was never committed.
- The scope line in the shared-key edit entry of docs/HYPOTHESIS.md that says results are regenerated on Kaggle no longer applies: those runs use the laptop. The pre-registered design, decision rules and predictions are unchanged.
- Also removed: Python caches, five untracked download logs, two untracked hard-stop scripts, and three tracked download logs (analysis/_logs/dl_bloom1b7.log, dl_gemma.log, dl_llama.log; restore with git checkout if wanted).

#### 2026-10-29 (night): Llama-3.2-3B-Instruct on Kaggle Models, no token needed
- The user got Meta's approval on Kaggle and wants the 3B model run there. Source: Kaggle Model `metaresearch/llama-3.2`, Transformers, variation `3b-instruct`, version 1 (6.43 GB, two shards, same file set as the Hugging Face repo; sizes match the local shards, 6,425,529,048 B). The base 3B has identical shard sizes, so only the byte check separates Instruct from base.
- Correction: the earlier note that TransformerLens fetches the gated config at load time (kaggle/driver.py `get_token`) does not hold for the installed version when a model object is passed in. Llama-3.2-3B has its config in a built-in table (`loading_from_pretrained.py`, the `Llama-3.2-3B` branch). Tested on the laptop with HF_HUB_OFFLINE=1, an empty HF_HOME and no token variable: `load_model("llama")` loaded in 94 s (bf16, 6.75 GiB) and predicted " Paris". The Gemma-2 branch is also name-based (not tested offline).
- New `code/model_manifest.py` (standard library only): per-tensor sha256 from the raw bytes in the .safetensors files plus the five small config/tokenizer files, independent of the shard split. `data/model_manifests/llama-3.2-3b-instruct.json` holds the laptop copy's fingerprint (254 tensors, 6.425 GB). Tests in tests/test_model_manifest.py (5, pass). Not yet run against the Kaggle copy.
- Kaggle runner rebuilt for one unified notebook (2026-10-29 night, uncommitted): `kaggle/driver.py` (no token; `prepare_model` links the attached Kaggle Model's weights, adds the laptop config/tokenizer files from `data/model_files/llama-3.2-3b-instruct.tar.gz`, then checks every tensor and small file against `data/model_manifests/llama-3.2-3b-instruct.json`; any difference stops the run), `kaggle/steps.py` (step `babel_llama`: smoke test, tokenizer comparison, paraphrase control, shared-key edit, non-geography facts, graphs; dev-only step `babel_llama_check`), `kaggle/layout.py` (code, tests, data and three saved laptop runs go to Kaggle), `kaggle/smoke.py`, `kaggle/tokenizer_compare.py`, `kaggle/make_plots.py`, `kaggle/notebook_cell.py` (the notebook cell). Registered as A27 in docs/HYPOTHESIS.md.
- Finding: the laptop model folder is the Unsloth mirror (weights byte-identical to Meta's, but config/tokenizer files differ; `unsloth_fixed: true`, pad token set, tokenizer.json 17.2 MB vs Meta's 9.1 MB). The user's Kaggle check confirmed: weights identical, all five small files differ, raw-tokenizer ids differ on 6/6 test strings. Not yet known whether ids differ on our fact prompts (the notebook measures it).
- Local dry runs of the runner (laptop GPU, check-only step) found five Windows-only problems, all fixed: cache-folder link, tar extraction warning, test temp-folder cleanup, UTF-8 reading of subprocess output, a test import path. Unit tests: 36 pass.
- Notebook carries its own code (2026-10-29 night, uncommitted): the user rejected the GitHub download, so `kaggle/make_notebook.py` packs `layout.members()` (code, tests, runner, data, three saved laptop runs; 392 files, 4.7 MB zip) into `kaggle/babel_llama.ipynb` (two cells: the packed code, then `kaggle/notebook_cell.py`). The pack is deterministic (fixed zip dates); the run prints its code id `bundle-<first 12 hex of sha256>` and checks the hash before unpacking. No push is needed to run it, but the notebook must be rebuilt after any code change. `.ipynb` files are excluded from the pack.
- Kaggle run cut by a lost connection (2026-10-30): the first full run of the self-contained notebook (code id bundle-6cfb32bf23fa) passed the model check, tests (37), number-format check (fp16 chosen on the T4s: 1.75x faster, agreement 1.00, gradient cosine 1.000), known-answer test and tokenizer comparison, then the user's session stopped during the first edit job; nothing was saved because results were only written at the end. Fix (uncommitted): `kaggle/driver.py` now saves progress to the notebook output after every finished job and every 15 min while jobs run, and on a rerun also imports run folders a stopped session left behind (not the three shipped laptop runs). 19 Kaggle tests pass.
- Progress lines (2026-10-30, uncommitted): the Kaggle log now prints every 5 min `PROGRESS jobs [###---] n/15` plus one line per running job (edit jobs: facts done of n-edit minus skip-first, edits written, time so far, time left once 3 facts are done and the job was not resumed; swap tests: pairs done of total). Runtime estimate before the first Kaggle result, from two laptop edit runs (60 edits in about 3 h each, about 3 min per edit): paraphrase control about 60 edits, shared-key edit about 60 edits with 2 key types; T4 speed unknown, so the 10.5 h stop may need a second pass.
- First full Kaggle run (2026-10-30, code id bundle-05074174f20e, 2x T4, fp16): 81.5 min in total, 13 of 15 jobs done; paraphrase control 46 min, shared-key edit 64 min, Spanish swap run 19 min, Hindi swap run 11 min; the progress lines worked and "progress saved" fired after every job. Failed: both Spanish swap-test analyses (exit 1 on all 3 attempts): `code/e2_analysis.py` for the EN-ES pair reads two saved laptop runs the notebook did not carry (`run_20261019_p3extra_10191057_llama`, `run_20261024_p3seeds_10240109_llama`, 1.7 MB together, both tracked in git). Reproduced locally from `layout.assemble` output (FileNotFoundError on records.json), fixed by adding both to `layout.RUNS`, both analyses then exit 0 on a real laptop Llama EN-ES run. New notebook id bundle-f122bd5c1df0. Results so far are from the two graphs shown in the notebook only (the zip has not been read); verdicts against the pre-registered criteria wait for results.json. Graph reading, not a verdict: English-key edit carries to unseen English phrasings (about 0.89 traced, 0.98 mid), to Spanish partly (0.50/0.33 traced, 0.85/0.67 mid), to Hindi barely (0.00 traced, 0.19/0.09 mid); shared-key edit raises Spanish traced-layer transfer (0.33 to 0.53 joint, 0.69 mean, intervals overlap) and does little for Hindi (about 0.03-0.06) or at the mid layer.
- Second full Kaggle run (2026-10-30, code id bundle-f122bd5c1df0): all 15 jobs done, 73.9 min, driver exit 0, nothing failed or skipped (paraphrase control 41 min, shared-key edit 58.5 min, Spanish swap run 18 min, Hindi swap run 9 min; both Spanish analyses now pass). Its two graphs are pixel-identical to the first run's, so the medians and intervals reproduced exactly across two independent full runs on 2x T4 (fp16). Still to read from babel_new_results.zip (not yet received): results.json of the paraphrase control, shared-key edit and non-geography analyses, tokenizer_compare.json, dtype_decision.json.
- Kaggle results read (2026-10-30): verdicts appended to docs/HYPOTHESIS.md (A21 partly supported and non-reproducing at single-edit level, A24 no improvement, A23 untestable, tokenizer ids identical apart from the default beginning token). Kaggle outputs copied into the repo without overwriting: `analysis/run_20261030_*` (12 folders), `analysis/tokenizer_compare_kaggle_llama.json`, `analysis/kaggle_dtype_decision.json`, `analysis/figures_kaggle/`, `analysis/_logs/kaggle_babel_llama/`, `data/v2/e2a23_facts_llama.json`, `data/v2/e2ha23_facts_llama.json`. Nothing committed.
- Key-similarity result and the Kaggle scale-up (2026-10-30, uncommitted): the registered key-similarity test (new `code/key_similarity.py`, run on the Kaggle English-key edits and the stored key cosines) finds that at the traced layer facts whose Spanish or Hindi key is closer to the English key transfer more (rho 0.55 and 0.52, Holm p 0.006 and 0.010, 28 facts) and at layer 14 there is no relation (-0.06, 0.02). An unregistered check with the edit layer as a further control weakens it (Spanish 0.45 [-0.01, 0.75], Hindi 0.06 [-0.32, 0.57]), so no mechanism claim yet; both prediction lines of the first pass missed. Registered A28 (scale-up and layer map) in `docs/HYPOTHESIS.md` before any of its data exist. The notebook now traces the facts itself (no saved laptop tracing, F7 or A20 run is an input any more), edits every valid fact of the relations with >= 10 kept facts (about 85) at the traced layer and layer 14 in an English-key run and a shared-key run, and maps layers 2, 6, 10, 18 and 22 on 40 facts in two shards (new `code/layer_map.py`; graphs added to `kaggle/make_plots.py`). Rebuilt: code id bundle-6d5af7908026, 467 files, 6.2 MB. Tests: full suite green, 49 tests (new tests for the statistics, the layer map and the job graph: every folder pattern a job reads is made by a job it depends on). Smoke on the laptop GPU only (not a T4): tracing in fp16 on 2 facts (7.5 GiB) and one fact edited with key modes en and mean_all (key cosines are written for both). Untested on a T4: the tracing job and the larger scale. Estimate: about 6 h on 2x T4 (limit 10.5 h).
- Goals file, model list and error review (2026-10-30, uncommitted): `GOAL.md` rewritten as the master tracking file (goal, status, plan table with current status, run path, models, published results vs the next notebook, decision rules, checks). Models existing work used were read from retrieved paper passages (not full reads): almost all use 6-8B models (Llama-2-7B, Qwen, Llama-3/3.1-8B, Mistral-7B, BLOOM family); only BLOOM-560M, mGPT-1.3B and Gemma 2B are as small as ours; none used Llama-3.2 or Gemma-2. New model list: committed Llama-3.2-3B-Instruct, Gemma-2-2B, Llama-3.2-3B base; to add Qwen2.5-3B and BLOOM-1b7; 7-8B models (Llama-2-7B, Llama-3.1-8B, Qwen2.5-7B, Mistral-7B-v0.1, optionally BLOOMZ-7b1) only after a T4 test (both T4s or 4-bit; untested). Error review of the next notebook: the code unpacked from the notebook and assembled as Kaggle does passes all 49 tests; every job command has an existing script and existing flags; found and fixed two risks: the key-similarity and layer-map scripts printed after saving, so a print error would have failed the job and the retry would have hit the write-once folder (now json-safe prints), and a stale help line named a merge script that does not exist. Key-similarity rerun on the real run-1 data reproduces rho 0.550 / 0.521 and the layer-map script ran on the real run-1 rows (small layers give undefined rho as designed); the layer-map graph was tested with undefined and empty layers. Rebuilt: code id bundle-21a9baef1028, 467 files, 6.2 MB. Inputs of the notebook: only the Kaggle Model Meta Llama 3.2, Transformers, 3b-instruct (no dataset; the old run Output should not be attached). Still untested on a T4: the tracing job in fp16, the larger scale, the time estimate.

#### Run 2 watch: two Spanish analysis jobs fail on Kaggle, packing gap fixed in source (2026-10-30, uncommitted)
- Run of bundle-21a9baef1028 (started 09:42): tracing took 108 min (guess was 30 min); GPU 1 sat idle for 73 min while everything else waited for tracing; Hindi non-geography analysis passed; Spanish non-geography analysis (two jobs) failed 3 of 3.
- Cause (reproduced by unpacking the notebook payload into a clean folder): the swap-test analysis reads the laptop tracing run analysis/run_20261015_p3trace_v2_llama (metric validity check via controls_overlap.SETS), and layout.RUNS did not pack it. My earlier check covered scripts and flags, not the data files each script reads; the other five analysis jobs (paraphrase, shared-key comparison, key similarity, layer map, graphs) were rerun in the clean payload and pass.
- Fix in source: kaggle/layout.py RUNS now includes that run (2.4 MB) and tests/test_kaggle.py has a guard test (50 tests pass). The running notebook was NOT rebuilt (code id stays bundle-21a9baef1028 on disk and on Kaggle); the next rebuild carries the fix. The two analyses are rerun on the downloaded Output (same code, CPU).
- Progress bars of the edit jobs count to the cap of 200; only 87 facts qualify (relations with at least 10 kept facts), so their "hours left" are about 2-3x too high.

#### Run 2 mid-run snapshot read (2026-10-30 21:13 IST, uncommitted)
- The user's zip (babel_new_results (1).zip) is a checkpoint at 226 min (13:29 on the Kaggle log clock = 19:00 IST), not the final Output: finished jobs only (tracing, base edit run on 86 facts, paraphrase analysis, non-geography runs). Shared-key edit was at 58/86 facts, layer map shard 0 had just started, key similarity, layer map and graphs had not run. Expected end 16:00-17:35 log clock = 21:30-23:10 IST.
- Paraphrase control, final on 86 facts (A21 rule, D = English-rewording transfer minus language transfer, CI above 0.15 = loss beyond wording): Spanish traced D 0.39 [0.29, 0.49]; Spanish layer 14 D 0.26 [0.15, 0.36] (lower bound 0.153, passes by 0.003); Hindi traced 0.79 [0.67, 0.89], layer 14 0.73 [0.64, 0.80]. Spanish transfer median 0.59 traced, 0.70 layer 14; Hindi 0.00 and 0.12. Prediction 7 (Hindi D above 0.15 in both) met.
- Non-geography facts, rerun on the snapshot with the fixed packing: numbers identical to run 1 to two decimals; still untestable (Spanish 53 facts but 5 source entities, floor is 25; Hindi 13 facts).
- PREVIEW only (not the registered result, 53-56 of 86 facts): key similarity traced Spanish rho 0.43 [0.16, 0.65], with edit layer controlled 0.29 [0.01, 0.55]; Hindi 0.51 [0.24, 0.72], layer-controlled 0.25 [-0.04, 0.52] (CI includes 0, rule (b) not met yet); layer 14 Spanish 0.07, Hindi -0.01. The registered analysis runs on the final Output.

#### Notebook rebuilt with the packing fix (2026-10-30 21:35 IST, uncommitted)
- kaggle/babel_llama.ipynb is now code id bundle-f8371a452538 (8.3 MB; adds the 2.4 MB laptop tracing run the Spanish swap-test analysis reads). Old file kept in the session scratchpad. Resume does not check the code id (only logged).
- Checked in the notebook unpacked into a clean folder: 50 tests pass; a23 Spanish (plain and leak-free) and Hindi (plain and leak-free) analyses, paraphrase, shared-key comparison, key similarity, layer map and graphs all run on the real run-2 snapshot data; job commands and dependencies checked. Edit and tracing jobs are no longer untested on a T4 (they ran on Kaggle in run 2).
- Measured on 2x T4, fp16: tracing 108 min, base edit run 112.5 min for 86 facts, shared-key run about 1.95 min per fact. Progress bars of the edit jobs count to the cap 200, not to the 86 facts.

#### A28 results read (2026-10-31, uncommitted)
- Fresh run of bundle-f8371a452538 finished: 20/20 jobs, 348 min; the two Spanish non-geography analysis jobs that failed before passed (packing fix works). Raw Output copied to analysis/ (14 new run folders and analysis/kaggle_output_20261031 with logs, figures, zip).
- Verdicts written into docs/HYPOTHESIS.md under A28: key similarity supported at the traced layer for both languages (rho 0.41/0.42, Holm p 0.001), none at layer 14; beyond the edit layer both pass narrowly (0.26 [0.004, 0.494], 0.22 [0.011, 0.438]); layer map pooled within-layer rho supported for Spanish (0.286 [0.117, 0.437]) and not for Hindi (0.171 [-0.024, 0.361]); wording loss in all four cells; shared-key gain at most 0.047. 7 of 8 sub-predictions met.

#### Comparison with published work after the A28 run (2026-10-31, abstracts and retrieved passages only, not a full review; uncommitted)
- New closest papers found: Qi et al. 2310.10378 (ROME on BLOOM-3b, English edit, 6 queries: the edit reaches Spanish and Vietnamese but not Hungarian and Greek, i.e. language-pair level, not per fact); 2504.09378 (instance-level alignment with English in middle layers predicts correct NLU predictions; patching English activations fixes them); 2510.10280 (subject and object entity alignment correlates with cross-lingual consistency; English-translation prompts help); 2601.03168 (embedding similarity predicts transfer across language pairs, rho 0.4-0.6, NLP tasks, 816 experiments). None of these ties a per-fact edit key to per-fact edit transfer within one language pair (not exhaustive).
- Agreement in direction with published work: same-script language reached, other script not (XME, Ifergan, Qi et al.); shared multi-language key gives no practical gain (published gains come from trained or neuron-based methods). Numbers are not comparable: different models (6-8B vs our 3B), editors and metrics.
- To do before writing: read Qi et al., CLM-Bench, LU-LAFNs, 2504.09378 and 2510.10280 in full and cite them in related work.

#### Who used this model or this work (search 2026-10-31, abstracts only; uncommitted)
- No paper found that does cross-lingual fact editing or tracing on Llama-3.2-3B or Gemma-2-2B. Llama-3.2-3B appears in other tasks: CaseEdit 2505.19383 (AlphaEdit, English commonsense editing), LAHIS 2511.07498 (language attention heads), RoSh 2609.34678 (factual judgement in 8 languages, steering), NeuronMoE 2603.05046. Gemma-2-2B: 2507.18918 (sparse autoencoder activations in 10 languages incl. Hindi).
- Also related to the key idea (monolingual): Keys to Robust Edits 2410.09338 (semantic keys cannot be both robust to paraphrase and specific; REP module re-learns keys).
- Venues of the cited papers not verified; cite by arXiv id.

#### Gemma-2-2B notebook built (2026-10-31; uncommitted, no GPU run yet)
- New step `babel_gemma2` in kaggle/steps.py and notebook `kaggle/babel_gemma2.ipynb`. Llama and Gemma jobs now come from one builder (`main_jobs`), so they cannot drift; the Llama step is identical to the run of bundle-f8371a452538 except that key_similarity names its default fact file explicitly. Gemma: middle layer 13 (26 // 2), edit cap 52 (= facts of relations with >= 10 kept facts; a test checks it, so progress bars count to the real total), no non-geography chain (untestable already at 93 Llama facts).
- Weights: Google's own Kaggle Model (Gemma 2, Transformers, gemma-2-2b), checked tensor by tensor against the laptop copy (`data/model_manifests/gemma-2-2b.json`, 288 tensors, 10.457 GB; config pack `data/model_files/gemma-2-2b.tar.gz`). The model is found by whole folder name and layer count (26), so 2b-it, Keras and wrong-model copies are refused with the reason.
- The user suggested the Kaggle dataset leimeng46/gemma-2-2b. Not used: every file size equals google/gemma-2b (first-generation Gemma, 18 layers: config 627 B, shards 4.95 GB + 67.1 MB, index 13.5 kB; checked on the Hugging Face file list), not Gemma-2-2B (26 layers, 3 shards, 10.5 GB). Could not download it here (no Kaggle login on this PC), so the match is by sizes. A29 registered in docs/HYPOTHESIS.md before any data.
- Each notebook now carries only its own model files, and the saved laptop runs only for the Llama step (Gemma 10.9 MB, Llama 8.3 MB). Tests that read repo-only files skip inside a notebook (they failed there before the fix and would have stopped the run at its start).
- Checks done on the CPU: 61 tests pass in the repo and inside both unpacked notebooks (as the notebook runs them); every job script and flag exists in both; the real model lookup, config pack and byte check run on the real Gemma files in a clean copy (23 s); the Gemma tokenizer gives identical ids for all 12714 strings of the data files; all 522 Gemma prompts give a subject span and a clean answer continuation; edit pool 52 facts. NOT checked (no GPU): the smoke job, the number-format check (Gemma-2 can overflow in fp16; bf16 is slower on a T4), tracing, and the three edit runs on Gemma on a T4. Old Llama notebook that produced the A28 results archived in analysis/kaggle_output_20261031/babel_llama_bundle-f8371a452538.ipynb; the rebuilt Llama notebook has the same jobs and a new code id.

#### 2026-10-31: Kaggle results added to the paper (Section 4.8)
- paper/paper_v2_draft.md: new Section 4.8 (wording control, key similarity, layer map, shared-key edit; Llama, Kaggle run bundle-f8371a452538), a Kaggle paragraph in Section 3, new abstract sentences, contribution (iv), related-work and position text (CLM-Bench arXiv:2601.17397 added, title and authors checked with the paper index), discussion, deviations, limitations and reproducibility updated. PDF rebuilt (20 pages, tables 5-8 and figures 7-8 are generated from the A28 result files by paper/build_tex.py; the literal tilde before 
ef in older references is fixed).
- results/manifest.json: 1941 to 2126 claims (a28_*, a26_*, a21k_*, a27_tok_*), all verified; the old 1941 values are unchanged. The A28 analysis files are saved under new names (see the note at the end of docs/HYPOTHESIS.md).
- Both notebooks rebuilt because code/make_manifest.py is packed: Llama bundle-44a41e614935, Gemma bundle-e17aedb76539 (only the unused manifest file differs from bundle-ac19dc0e46ce and bundle-18a19884bb78); 61 tests pass in the repo and inside both unpacked notebooks.
- Nothing committed. Not checked: the paper has not been read by a person; the Hindi and Spanish native-speaker check is still open.

#### 2026-10-31: first Gemma-2-2B Kaggle attempt stopped at the smoke job; fixed
- User run of babel_gemma2 (bundle-18a19884bb78, 7.6 min): model found at /kaggle/input/models/google/gemma-2/transformers/gemma-2-2b/1, every tensor and file matched, 61 tests passed, fp16 chosen, then smoke failed 3 times (exit 1), the rest skipped. Cause (reproduced on the CPU, local weights, bf16): Gemma-2-2B base answers "The capital of France is" with " a" (0.20), Paris is 0.05, so the check I wrote returned a wrong-answer exit. Fault of my check, not of the model.
- Changed: Gemma smoke prompt (steps.py MODEL gemma2 smoke), smoke.py prints the top five tokens on a wrong answer, dtype_check.py also tests finiteness under the tracing noise (fp16 overflow in Gemma-2 would otherwise only show as dropped facts), two tests added (63 tests). Amendment text at the end of A29 in docs/HYPOTHESIS.md. Lesson for the Llama base notebook: base models need a question-style smoke prompt.
- Still untested on a GPU: tracing, the three edit runs, the shared-key edit and the CPU analyses on Gemma-2-2B; knowledge_edit_v2.py has never run with Gemma (the earlier laptop Gemma edits used the first edit script).

#### 2026-10-31: Gemma-2-2B Kaggle run (bundle-b7d6962e96d7): 8 of 12 jobs done, shared-key edit failed; notebook rebuilt
- Run: 238 min on 2 T4, fp16 (1.70x), tracing 44 min, edit_base 80 min, layer-map shards 114 and 107 min. The Output zip was not sent, so no Gemma number has been read yet; only the console (saved as a notebook in Downloads) was seen.
- edit_shared failed 3/3 right after its first written edit; key_similarity, keymode_analysis and graphs were skipped. Cause unknown (no traceback in the console). Ruled out from files: identical EN/ES subject text (Llama had 24 such facts, no error).
- Changed (A30 in docs/HYPOTHESIS.md): knowledge_edit_v2.py lets a key mode after the first fail without stopping (mode_failures.json, WARNING line); driver.py prints the last 14 log lines of a job that fails for good and any WARNING line of a job that finishes. test_gpu_jobs_start_with_a_gap was timing-fragile (failed twice inside the Llama notebook folder, 1.17 s against 1.4 s): gap 3.0 s, limit 2.0 s. 66 tests pass in the repo and inside both unpacked notebooks. Notebooks: Gemma bundle-d2e183645bef, Llama bundle-9fe594b6cd95.
- Rerun plan: attach the old Gemma Output so the 8 finished jobs are skipped; edit_shared, keymode_analysis, key_similarity and graphs run.
- Laptop GPU dry runs (RTX 5060, fp16, peak 7.25 GiB; user said go): knowledge_edit_v2.py with both key modes on 3 Gemma facts (event_year_002/007/009, 2 layers each) and on capital_of_035 (Sri Lanka): no failure. Cause of the Kaggle failure found: capital_of_035 has English and Spanish key cosine exactly 1.0 (same subject text, same words before it); two equal keys make joint_update singular (numeric test: torch.linalg.solve raises). The first fact of the Kaggle job is probably capital_country_003 (Madrid), same property. Fixed in joint_update (A30, change 3): copies of a key are kept once, small solve in float64; 3 new tests; 69 tests pass in the repo and inside both notebooks. Notebooks: Gemma bundle-e186cf5d5e1b, Llama bundle-4c26396318b4.
- The old code was not run on the fact on the GPU. The English-key base runs needed to test keymode_analysis / key_similarity / graphs on Gemma data were stopped by the system for low memory after 1 of 8 edits; not restarted (rule: only when asked). Those three jobs are CPU-only and have not run on Gemma data.

#### 2026-11-01: commit and push; master record added
- Pushed to GitHub (praann07/babel-effect, main) at the user's word, in four commits: code and tests, notebooks and data, run outputs, docs and paper (manifest 2126 claims verified before the commit; 69 tests pass, 1 skipped).
- New go-to document: `paper/master_record.md` and `paper/The_Babel_Effect_MASTER_RECORD.pdf`, built by `paper/build_master.py` from hand-written chapters in `paper/master/` (goal and rules, story, problems faced, the user's words and the assistant's mistakes, results at a glance, open items, paths) plus the paper, `docs/*.md` and `GOAL.md` copied word for word, plus generated lists (repository layout, run folders, git history). Rebuild after any source change. Nothing in the Gemma Kaggle run (bundle-e186cf5d5e1b) has been read yet.

#### 2026-11-01: git author fixed so GitHub credits the commits
- All 133 commits were authored as `babel <babel@local>` (132) or the real GitHub identity (1), so the GitHub contribution graph showed nothing. At the user's word ("do the best") the author and committer of all 133 commits were rewritten to `Mamidala Som Praneeth Babu <187202597+praann07@users.noreply.github.com>` (GitHub no-reply address, real email not published) and `main` was force-pushed (`baa7a06` became `fb7f10f`; every hash changed, file contents and dates unchanged, tree hash identical). The old history is kept in the local branch `backup-before-author-fix`. GitHub now returns `praann07` as the author of the commits. The repo-local git identity is set to the same name and address, so new commits count.


## 10.3 Experiment log

*Copied word for word from `docs/EXPERIMENT_LOG.md` when this record was built. To change it, edit that file and rebuild.*

### EXPERIMENT LOG
| run ID | hypothesis | config | result | verdict |
|---|---|---|---|---|
| P0-smoke-gpt2 | model loads and predicts Paris | gpt2 bf16 cuda, prompt "The Eiffel Tower is located in the city of" | top1 Paris p=0.0706, VRAM 0.39 GiB | pipeline OK; not the target model |
| P0-smoke-llama3.2-3b | model loads and predicts Paris | unsloth/Llama-3.2-3B-Instruct (local), bf16 cuda, `from_pretrained_no_processing`, same prompt | top1 " Paris" p=0.6132 (then "..." 0.0688, " what" 0.0368, " which" 0.0368, "?\n" 0.0287); peak VRAM 7.49 GiB | P0 gate MET on target model; VRAM is near the 8 GiB cap, so later runs need chunked caching (artifact: analysis/p0_smoke_llama.json) |
| P1-filter-llama3.2-3b | enough facts are correct in EN+ES+HI | filter_facts.py, greedy, exact-object prefix match, HI variant per relation by whole-set accuracy | 154 candidates -> 52 kept (capital_country 26, capital_of 22, landmark_city 4); EN acc 0.95-1.00, ES 0.65-0.78, HI 0.18-0.58 (variant 0 won every relation) | P1 gate MET (n=52 >= 50, table saved: analysis/run_20261012_p1filter/accuracy.md); relation mix is skewed |
| P1-filter-v2 | stricter first-token rule + 289 candidates give n >= 50 | filter_facts.py (leading space only stripped), tag p1filter_v2 | 67 kept (capital_country 33, capital_of 32, landmark_city 2); v1 run (52 kept) superseded because 3 Hindi facts began with a newline | P1 gate MET (n=67); supersedes P1-filter-llama3.2-3b |
| P2-trace-v1 / v2 | tracing works on 10 facts | competitor = clean runner-up (v1), corrupted-run top (v2); facts from P1 v1 | v1: 5 of 30 pairs had negative total effect; v2 fixed that but used facts later superseded | superseded; kept as raw output (A2) |
| P2-trace-v3 | critical MLP layer at last subject token is mid-layer (H3: 8-19) | causal_trace.py, 10 facts x 3 langs, sigma=3x pooled subject-embedding std, run_20261012_p2trace_v3 | 30/30 collapsed, 0 negative total effects; critical MLP layer EN median 2.5, ES 2, HI 2 (range 0-5); 0/30 in layers 8-19 | H3 layer range NOT met; two-stage pattern (subject-token MLP/resid, then last token) visible; figs in analysis/figs_20261012_p2trace_v3 |
| P3-trace-all | full tracing of all kept facts | causal_trace.py --subset all, run_20261012_p3trace, 67 facts x 3 langs | 201/201 pairs traced, peak VRAM 7.51 GiB; 66 facts valid in all 3 languages (capital_of_021 excluded: no collapse/positive effect in at least one language) | primary set n=66 >= 50 |
| P3-overlap | H1/H2: profile overlap above permutation null; EN-ES > EN-HI | circuit_overlap.py, run_20261012_p3overlap, 1000 perms, 2000 bootstrap, seed 20261012, n=66 | MLP profile median cosine: EN-ES 0.748 [0.676,0.801] z=6.06 p=0.001; EN-HI 0.323 [0.106,0.448] z=0.39 p=0.35; ES-HI 0.227 [-0.037,0.349] z=0.38 p=0.37 | H1 holds for EN-ES only; H2 supported (EN-ES > EN-HI); H3 layer range not met (see LOG) |
| P4-edit | H4/H5: EN edit at traced layer succeeds >= 80%, transfers ES > HI, specific, little damage | knowledge_edit.py, run_20261013_p4edit, 10 facts x 3 conditions, ROME rank-1 at W_out, A4 design | EN success: traced 9/10, mid 10/10, random 10/10. Median dP: traced EN 0.949, ES 0.107, HI 0.004; mid EN 0.927, ES 0.180, HI 0.242; random EN 0.937, ES 0.005, HI 0.004. Peak VRAM 7.58 GiB | P4 gate MET (90% >= 80%) |
| P5-transfer | transfer ratio 0 < HI < ES < 1; specificity < 0.05; damage < 1.05 | eval_transfer.py on the P4 run, bootstrap 2000, seed 20261012 | Transfer ES/HI (EN-success edits): traced 0.147 / 0.002; mid 0.203 / 0.244; random 0.005 / 0.005. Specificity median (max abs dP, 20 controls): traced 0.045, mid 0.351, random 0.061. Damage (ppl ratio) median 1.0009 / 1.0021 / 1.0000, worst 1.0165 | H4 ordering holds for traced only (fails at mid: HI >= ES); H5 damage met in all conditions, specificity met only for traced and only at the median (worst 0.981); P5 gate MET (tables complete) |
| P1v2-templates-llama | answer-first templates raise filter yield (A7) | select_templates.py, 6 templates x 3 langs x 5 relations, 30 tuning facts, strict match | Best templates chosen; Hindi accuracy on tuning set: capital_of 0.60, capital_country 0.90, landmark_city 0.27, invention_inventor 0.03, event_year 0.60 | Hindi knowledge of inventors and landmarks is near zero for this model; see pool_accuracy.json in run_20261014_templates_llama |
| P1v2-filter-llama | n >= 50 facts correct in EN/ES/HI under 3 phrasings | filter_facts_v2.py, canonical strict + one lenient paraphrase per language, run_20261014_p1v2b_llama | 408 candidates -> 93 kept: capital_of 27, capital_country 38, landmark_city 6, invention_inventor 0, event_year 22 | P1v2 gate MET for the pooled set (93 >= 50); relations with >= 10 facts: capital_of, capital_country, event_year; landmark_city and invention_inventor too small to analyse alone |
| P3v2-trace-llama | full-answer tracing on all v2 facts | causal_trace_v2.py, run_20261015_p3trace_v2_llama, 93 facts x 3 langs, phrasing 0 | 279/279 pairs traced, peak VRAM 7.51 GiB, 92/93 facts valid (capital_of_021 excluded: no collapse in >= 1 language) | primary set n=92 |
| P3v2-overlap-llama | H6 (EN-ES z>2, EN-ES > EN-HI), H9 | circuit_overlap.py, run_20261015_p3overlap_v2_llama, 1000 perms, 2000 boot, seed 20261012 | MLP cosine: EN-ES 0.627 [0.557, 0.702] z=4.22; EN-HI 0.509 [0.335, 0.594] z=2.45; ES-HI 0.423 [0.313, 0.529] z=2.47 (null means 0.519 / 0.429 / 0.328); all 6 primary tests pass Bonferroni | H6 supported for Llama; H9 untestable/failed (see LOG) |
| P1v2-filter-gemma2 | n >= 50 facts correct in EN/ES/HI for Gemma-2-2B | filter_facts_v2.py --model gemma2, Llama-selected templates, run_20261015_p1v2_gemma2 | 408 -> 26 kept (event_year 17, landmark_city 5, capital_of 3, capital_country 1, invention_inventor 0); canonical Hindi accuracy 0.00-0.49 by relation | n=26 < 50: target NOT met with Llama-selected templates; per-model template selection tried next |
| P3v2-trace+overlap-gemma2 | H6 replication on a second model | causal_trace_v2.py --model gemma2 (peak VRAM 7.15 GiB) and circuit_overlap.py, n=26 valid | MLP cosine: EN-ES 0.419 [0.136, 0.630] z=2.56 (Spearman z=4.15); EN-HI 0.063 [-0.060, 0.168] z=-0.27; ES-HI 0.072 [-0.016, 0.244] z=-0.08 | exploratory only (n < 50): EN-ES above null, Hindi pairs at null; critical MLP layer median 6 (EN), 5.5 (ES), 6 (HI) |
| P1v2c-filter-gemma2 | n >= 50 Gemma facts (A8) | filter_facts_v2.py --model gemma2, own templates, 447 candidates (events extended by 39), run_20261015_p1v2c_gemma2 | 58 kept: event_year 30, capital_of 11, capital_country 11, landmark_city 6, invention_inventor 0 | n=58 >= 50 met; more than half the set is event_year |
| P3v2c-overlap-gemma2 | H6 replication on Gemma-2-2B | causal_trace_v2.py + circuit_overlap.py, n=58 valid, run_20261015_p3overlap_v2c_gemma2 | MLP cosine: EN-ES 0.464 [0.329, 0.542] z=4.81 p=0.001; EN-HI 0.226 [0.168, 0.283] z=2.19 p=0.010; ES-HI 0.137 [0.014, 0.267] z=0.08 | H6 supported for Gemma (EN-ES z>2; EN-ES > EN-HI); Hindi pairs weak or at the null |
| P3v2all-overlap-llama | H9, F4 (A9) | circuit_overlap.py on 92 + 34 extra facts, run_20261019_p3overlap_v2all_10191057 | n=126; MLP cosine EN-ES 0.642 z=5.06, EN-HI 0.500 z=2.90, ES-HI 0.413 z=2.63; non-copyable n=104: z 4.87 / 2.78 / 3.51; city_country n=27 EN-ES z=1.96 | F4 met; H9 not supported |
| P3v2all-overlap-gemma2 | F4 (A9) | same, run_20261020_p3overlap_v2all_10200339 | n=59 (1 extra fact); non-copyable n=29: EN-ES 0.507 z=2.53, EN-HI z=1.16, ES-HI z=0.42 | F4 not met for Gemma |
| H8-phrasings | H8 | causal_trace_v2.py --phrasings 1,2 --limit 30 + h8_analysis.py, run_20261019_h8_10191057_{llama,gemma2} | within vs cross cosine: Llama 0.754 vs 0.609 (d 0.139 [0.101, 0.228]); Gemma 0.741 vs 0.338 (d 0.341 [0.287, 0.463]) | H8 supported in both |
| F3-incontext | descriptive | incontext_baseline.py, run_20261020_incontext_10200339_{llama,gemma2}, 10 facts | EN-statement success EN/ES/HI: Llama 0/1.0/1.0 (EN result unexplained), Gemma 0.8/0.7/0.6 | descriptive |
| F7-llama | A9 follow-up edits | knowledge_edit_v2.py --n-edit 30 --skip-first 10 --add-traced, run_20261020_p4f7_llama, 20 facts | EN success traced 0.80 / L//2 0.95 / random 0.65; ES transfer 0.70 / 0.78; HI 0.005 / 0.105; specificity 0.06 / 0.57 | descriptive; Gemma F7 interrupted |
| A11-controls | A11 (pre-registered 2026-10-21 before computing) | code/controls_overlap.py on cached traces (Llama 126, Gemma 59 facts), 10,000 permutations, run_20261021_controls | Centred EN-ES stratified z: Llama 3.61, Gemma 5.43. Same-language baselines: Gemma same-fact 0.424 vs different-fact 0.242 (EN-EN 0.247); Llama same-fact 0.570 vs 0.480, but EN-EN 0.606. Lexical: 36.5% (Llama) / 11.9% (Gemma) identical subjects; Spearman(sim, cos) 0.29 (p<0.001) / 0.22 (p=0.097); low-similarity half z uncentred/centred: Llama 2.04 / 1.44 (n=64), Gemma 2.80 / 3.42 (n=30). Cluster-bootstrap CI unchanged | C1 holds in both; C2 holds for Gemma, fails for Llama after centring; lexical confound supported in Llama |

| run_20261024_confound_A12a | A12/A12a: is the EN-ES alignment explained by shared noise rows, prefix identity, magnitude or token overlap? (CPU, 10,000 derangement permutations) | code/confound_overlap.py, NC profiles, strata=relation | Prefix-identical facts are rare (Llama 1, Gemma 4). Mean NC increment D: Llama all 0.227, informative 0.239 [0.161, 0.310], S4a name-free target 0.127 [0.003, 0.250] (bootstrap CI includes 0); Gemma all 0.219, informative (n=25) 0.118 [-0.016, 0.275], S4a target -0.025 [-0.248, 0.198]. Equal-span (shared last-token noise row) vs other: Llama 0.436 vs 0.145; Gemma 0.492 vs 0.002. | Llama: present on both primary quantities; Gemma: unresolved. Equal-span pattern is consistent with a shared-noise artefact; needs the independent-noise re-trace (A13a). Predictions: Q2 Llama yes / Gemma no; Q3 yes; Q4 NO (Llama CI excludes 0); Q5 yes. |

| run_20261024_seeds_A13a_10240109 | A13a: does EN-ES NC alignment survive independent corruption noise per language? (GPU, 16 draws per fact-language; 10,000 derangement permutations) | code/trace_seeds.py + code/seeds_analysis.py; regression with original seed passed (median abs diff 0.011-0.025 nats, corr >= 0.99) | Informative facts: Llama D_ind 0.346 [0.271, 0.424]; Gemma 0.257 [0.130, 0.381]. Shared-vs-independent noise (Delta SH-IND5): Llama 0.022 [-0.025, 0.066]; Gemma 0.054 [-0.084, 0.179]; only Gemma EQ facts differ (0.185 [0.009, 0.357]). Within-language replicate ceilings: Llama 0.82/0.79, Gemma 0.59/0.54; reliability index 0.39 (Llama), 0.31 (Gemma). | G1: SURVIVES in both models. Predictions P1 (Gemma), P2 (Llama) and P3 (Llama) were WRONG; P3 (Gemma) and P4 held. Shared noise explains little of the alignment. The single-draw profiles of A12a were too noisy (that is why Gemma was 'unresolved'). Averaging 10 draws raises D. Interpretation (shared lookup) is still untested: E2. |

| run_20261024_seeds_covariates_A13b | A13b: does the name/answer/span/magnitude covariate model remove the independent-noise alignment? (CPU) | code/seeds_covariates.py, A12a S4a spec on A13a IND increments | Name-free target (J=0, LT=0, AI=0, NR=0): Llama informative (n=103) 0.319 [0.210, 0.428] present; Llama all non-prefix-identical (n=125) 0.341 [0.228, 0.453]; Gemma informative (n=25) 0.138 [-0.086, 0.362] unresolved; Gemma all non-prefix-identical (n=55) 0.216 [0.047, 0.385] present. | Prediction held. Llama alignment is not explained by token overlap, answer identity, span or magnitude; Gemma is unresolved at n=25 and present at n=55. Profile similarity still does not show shared retrieval: E2. |

##### A13 (E2) cross-language interchange — run_20261025_e2full_A_{llama,gemma2}; analysis run_20261025_e2_analysis_A13
- Config: `code/interchange.py` full (first 10 pairs excluded as pilot), `code/e2_analysis.py`; confirmatory = source EN/ES subject tokens share no piece (Jaccard 0). Llama 142 pair-directions/direction, 67 source entities; Gemma 162, 71.
- P1 gate (same-language flip >= 60%, delta >= 2 nats): passed in all 8 cells (flip 0.68-0.95, mean delta 19-27 nats).
- Site A (last subject token), ratio I = cross/same over band, 95% CI: Llama en2es 1.169 [1.092,1.277], es2en 0.901 [0.857,0.944]; Gemma en2es 1.068 [1.030,1.115], es2en 0.938 [0.909,0.967]. All meet the P2 SHARED rule (lower bound >= 0.7). Cross-language flip at l*: Llama 0.85/0.67, Gemma 0.90/0.87; self and ctrl flips 0.00.
- Site B (final token): I = 0.901 [0.846,0.960], 1.022 [0.974,1.067] (Llama); 1.000 [0.953,1.046], 1.107 [1.054,1.160] (Gemma). Verdict P2 SHARED by the rule, but the cross flip rate is only 0.34-0.45 against 0.67-0.94 same-language (delta ratio ~1 while argmax flips are fewer).
- P5 lexical (prediction at J=1 minus J=0 > 0.2): not met (Llama 1.17 -> 0.80, Gemma 1.03 -> 0.91; the J=1 values are LOWER, so name overlap does not inflate the ratio).
- P6 metric validity (Spearman of per-fact ratio vs A13a increment > 0.2): Llama rho = -0.036 (n=40, p=0.83) FAILS; Gemma rho = 0.233 (n=9, p=0.56) passes nominally but n is too small to mean anything. So the tracing-profile increment does not predict per-fact interchange strength.
- Verdict: P2 SHARED in both models and both directions at site A. Subject-token states carry language-agnostic fact identity. Not shown: that this holds for Hindi, other relations, or that the tracing-profile similarity measures the same thing (P6 fails).
- Caveats: 4 relations only (capital_country, capital_of, city_country, landmark_city); identity unit test tolerance 0.119 nats; pair sampling is two sources per target; one earlier Llama run (run_20261024_e2full_10240848_llama) was stopped at 210/396 and kept empty.

#### 2026-10-25 In-context baseline anomaly (Llama English 0/10), diagnostic
- Question: is the English 0/10 (F3, `analysis/run_20261020_incontext_10200339_llama`) a bug? Method: greedy generation (10 tokens, bf16, CPU, HF generate) for the first 6 edit facts, English and Spanish, with and without the context sentence (`code/incontext_diag.py`, output `analysis/run_20261025_incontext_diag_llama/generations.txt`). Descriptive, n=6, not in the manifest.
- Finding: with the context, English generations restate the original answer and reject the new one ("France, not Denmark", "actually Paris, not Warsaw"); Spanish generations repeat the new object. No code error found: the success rule only counts the new object as the answer. Reading: the instruct model corrects a counterfactual sentence that the question repeats verbatim.
- Consequence: the English figure is a prompt-design effect; reported as such in paper section 4.5. A fix (different context wording) would need a new GPU run; not done.

#### 2026-10-25 Overlap rerun with 10,000 permutations (A14/A14a)
- `code/circuit_overlap.py --perm 10000` (new flag, default 1000), same traces and seed; four runs `analysis/run_20261025_p3overlap_{v2,v2all}_perm10k_{llama,gemma2}`; CPU, about 1 minute each.
- EN-ES MLP cosine: Llama 4.22 to 4.01 (v2, n=92), Gemma 4.81 to 4.82 (n=58), p = 0.0001; all verdicts unchanged. Manifest 1059 claims verified.

#### 2026-10-25 Reference check
- All 23 arXiv identifiers in the v2 draft were checked against the arXiv API (title, first author, year): all match. One correction: arXiv lists Grattafiori first for the Llama 3 paper (2407.21783), not Dubey; fixed in `paper/paper_v2_draft.md`. Added Dumas et al. (2411.08745), Zhang and Nanda (2309.16042), Heimersheim and Nanda (2404.15255) with in-text citations in section 2. Only title/author/abstract were checked, not every claim attributed to each paper.
- `paper/paper.md` and `paper/latex/main.tex` are the older v1 text and still say Dubey; they are not updated (v2 is the working draft).

#### 2026-10-25 Reproducibility check
- `./reproduce.sh cpu repro1` (new mode): A12a, A13a, A13b, E2 analysis and the A14 overlap reruns regenerate from saved traces with max abs difference 0 on 4,000+ numeric leaves (`code/compare_results.py`; log `analysis/_logs/repro_cpu1.log`; dirs `analysis/run_20261025_*_repro1`).
- Clean clone of the GitHub repo: `./reproduce.sh verify` = 1059/1059 claims, `./reproduce.sh test` = 13 tests OK. The `cpu` mode and `full` still need `models/` (tokenizers/weights), which are not in git. `reproduce.sh` now quotes the python path (the first clone test failed on a path with spaces).
- Software Heritage archive of the repo requested 2026-10-25 09:44 UTC (save request id 2575106, https://archive.softwareheritage.org/api/1/origin/save/2575106/) as an external timestamp for the pre-registration history; snapshot swh:1:snp:37a249332b2835adb08442a51564faeeff43b98b (archived 2026-10-25). This is not a substitute for a Zenodo/OSF DOI.

#### 2026-10-25 Gemma F7, Llama-1B and BLOOM pilots (GPU, unattended; ran 14:48-16:25)
- Gemma F7 (30 edit facts, `analysis/run_20261020_p4f7_gemma2`, resumed into the 10-20 partial dir via `--resume`; records.jsonl was appended, not overwritten): traced / L//2 / A9 layer 3, n = 20 / 20 / 19. English success 0.80 / 0.65 / 0.89. Median Spanish transfer 0.65 / 0.42 / 0.66; Hindi 0.10 / 0.20 / 0.0005; specificity loss 0.084 / 0.078 / 0.032; perplexity ratio 1.0002-1.0005. 18 manifest keys `f7_gemma2_*` (manifest 1077 claims, verified).
- Finding to report carefully: the "random" condition in `eval_sweep.py` draws from swept layers other than traced and L//2; in F7 that is the A9-selected layer (Llama layer 21 for 20/20 facts; Gemma layer 3 for 19/20). So the "random" column is the A9 layer, not an independent random layer. Paper text fixed accordingly. Llama layer 21 gives ~0 Spanish transfer; Gemma layer 3 gives 0.66, a model difference. Descriptive, n = 19-20.
- Llama-3.2-1B-Instruct pilot: filter kept 38 facts (<50), full pipeline not run per the A9/A10 rule. BLOOM-1b7 pilot: kept 34 (<50), not run. Both are negative feasibility results: the small models know too few of the benchmark facts in all three languages.
- Software Heritage archived the repo (visit status full, save request 2575106): snapshot swh:1:snp:37a249332b2835adb08442a51564faeeff43b98b (taken after commit f116b1f, 2026-10-25). External timestamp for the pre-registration history up to that commit; it is not a DOI.

#### 2026-10-26 A15: English-Hindi interchange (run dirs `analysis/run_20261026_e2hfilter_*`, `*_e2hfull_A_*`, `analysis/run_20261026_e2h_analysis_A15`; manifest `e2h_*`)
- **Facts kept** (EN and HI both answered correctly, template accuracy >= 50% in both languages): Llama 30 facts, all `capital_of`; Gemma 16 facts, all `capital_country`. Hindi recall is the limit (tuning accuracy in Hindi 3-19 of 30 per relation). Pairs run: Llama 60 (100 pair-directions after the pilot), Gemma 32 (44).
- **Primary direction EN->HI (target prompt in Hindi): P1 gate FAILED in both models** (same-language flip at l* 0.12 Llama, 0.36 Gemma; mean effect 10.3 / 10.1 nats), so by A15 the model is uninformative for this direction and no claim is made. (Ratios I = 2.2 and 1.7 are not interpretable when the ceiling is that weak.)
- **Secondary direction HI->EN: gate passed** (same flip 0.80 Llama, 0.82 Gemma; 25.6 / 23.8 nats). Site A: Llama I = 0.74 [0.67, 0.81], cross flip 0.26 vs same 0.80; Gemma I = 0.74 [0.69, 0.82], cross flip 0.45 vs same 0.82; both PARTIAL by the rule (lower bound < 0.7 and upper 90% > 0.3). Site B: I = 0.85 / 1.20 (nominally SHARED) but the cross flip is 0.00, so the ratio is not meaningful there.
- **Floor not met:** the pre-registered floor was >= 25 distinct source entities; Llama has 23, Gemma 12. **A15 is therefore underpowered and makes no claim.** The HI->EN site-A numbers (PARTIAL, I about 0.74 in both) are descriptive only.
- **Prediction check:** Llama EN->HI predicted I 0.4-1.0 (P4 or P2): not testable (gate failed). Gemma predicted few facts or a failed gate: correct (16 facts, EN->HI gate failed).
- Not done: a relation-balanced Hindi set would need either larger fact pools or models with better Hindi recall (a new pre-registration).

#### 2026-10-26 A16: Hindi->English interchange, larger fact set (run dirs `analysis/run_20261026_e2h2filter_*`, `*_e2h2full_A_*`, `analysis/run_20261026_e2h2_analysis_A16`, `analysis/run_20261026_e2h2_outside_A15`; manifest `e2h2_*`, `e2h2out_*`)
- **Facts kept:** Llama 66 (capital_country 21, capital_of 30, city_country 15), Gemma 55 (capital_country 16, capital_of 9, city_country 20, landmark_city 10). Runs: Llama 264 pair-directions, Gemma 220, all HI->EN. Filter ~6-7 min each, interchange 45 + 43 min, analysis 1 min (total 1h43).
- **Floor met:** source entities Llama 66, Gemma 54 (>= 25). Gate P1 passed (same-language flip 0.80 Llama, 0.90 Gemma at l*; l* = 5 and 6).
- **Site A (primary), confirmatory pairs:** Llama I = 0.74 [0.69, 0.78], cross flip 0.48 -> **P4 PARTIAL** (lower bound < 0.7). Gemma I = 0.76 [0.71, 0.81], cross flip 0.64 -> **P2 SHARED** (lower bound 0.71 >= 0.7, by 0.01). Self and control flips 0.00-0.01.
- **Site B:** ratios 0.91 and 1.13 (nominally SHARED) but cross flip 0.00, so not interpretable by the pre-registered rule.
- **Per relation (descriptive, site A):** Llama 0.74 / 0.72 / 0.80 (capital_country / capital_of / city_country); Gemma 0.75 / 0.81 / 0.87 / 0.53 (… / landmark_city 0.53 [0.45, 0.62]).
- **Outside-A15 subset (both target and source facts not in the A15 kept set):** Llama I = 0.76 [0.71, 0.81] (36 source entities), Gemma 0.77 [0.71, 0.83] (38); within 0.02 and 0.01 of the full-set ratio (`code/a16_outside_a15.py`, band/l* from the full set).
- **Prediction check:** (1) kept facts Llama >= 80, Gemma >= 60: MISSED (66, 55); relations >= 2 each: met; floor met. (2) both P4 with I 0.55-0.90: Llama met; Gemma got P2 (the ~20% case) with I inside the range. (3) HI->EN lower than ES->EN: met in both (0.74 vs 0.90; 0.76 vs 0.94, intervals do not overlap). (4) outside-A15 within 0.15: met.
- **Reading:** Hindi retrieval is at least mostly shared with English in both models (ratio ~0.75, cross flip 0.48-0.64), but weaker than English-Spanish (0.90-0.94), and the two models straddle the 0.7 line, so the strict pre-registered verdict differs (PARTIAL vs SHARED) although the point estimates are the same. EN->HI was not tested (single-token patch limit, A15).

#### 2026-10-26 | A17 answer-in-subject leakage re-analysis (CPU, pre-registered in `docs/HYPOTHESIS.md` A17 before running)
- Rule (`code/leakage.py`): fact dropped if the folded answer (>=3 chars) is a substring of the folded subject in any language. Finds 22 leaky facts in the v2 pools (Llama overlap set 6/93, Gemma 10/58; E2 EN-ES 5/198 and 10/214; A16 1/66 and 6/55). The external audit quoted larger counts (16/93, 25/58); we could not reproduce its rule, so ours is the registered one.
- Runs: `analysis/run_20261026_{seeds_A13a,seeds_covariates_A13b,e2_analysis_A13,e2h2_analysis_A16}_A17`, `analysis/run_20261026_p3overlap_v2_perm10k_A17_{llama,gemma2}` (driver `scripts_a17.sh`, log `analysis/_logs/a17.log`).
- Result: no verdict category changed. G1 SURVIVES both (Llama D 0.366 [0.290, 0.448], Gemma 0.343 [0.233, 0.463]). A13b Llama informative 0.294 [0.180, 0.408] present (n=95; before 0.319); Gemma informative unresolved at n=15 (as before), all-non-IP present 0.340 [0.117, 0.563]. E2 EN<->ES stays P2 SHARED in all four cells (Llama 1.173/0.905, Gemma 1.074/0.932). A16 HI->EN: Llama 0.741 [0.699, 0.785] PARTIAL (unchanged; lower bound clears 0.65, not 0.70); Gemma 0.804 [0.763, 0.842] SHARED (before 0.763 [0.713, 0.811]), lower bound clears 0.75, so the Gemma SHARED verdict does not depend on the 6 leaky facts. Overlap EN-ES: Llama z 4.01->3.97 (n 92->86), Gemma 4.82->4.66 (58->48). One change worth stating: Gemma EN-HI pooled z 2.06->1.47 (p 0.056; was already marginal); Llama EN-HI 2.47->2.76.
- Predictions (1)-(5) met; (6) Gemma A16 survived (not flipped).
- Manifest 1289 claims verified (`a17_*` keys).

#### 2026-10-26 | A17b bootstrap-clustering sensitivity of the interchange ratio (CPU, registered before running; prompted by an external audit)
- `code/cluster_sensitivity.py` -> `analysis/run_20261026_cluster_A17b/results.json` (log `analysis/_logs/a17b.log`). Site A, confirmatory pairs, 5,000 draws, four clusterings: S source fact (the registered one), T target fact, C connected components, W two-way.
- EN<->ES (A13), all 8 cells (2 models x 2 directions x plain/leak-free): label SHARED under S, T, C and W; the lowest lower bound is 0.83 (Llama ES->EN, C). Prediction (1) met.
- HI->EN (A16): point estimates unchanged (Llama 0.741, Gemma 0.763 plain / 0.804 leak-free). Labels are NOT clustering-robust except Gemma leak-free. Llama (plain and leak-free): S 0.698-0.784 PARTIAL, T 0.711-0.769 SHARED, C 0.723-0.796 SHARED, W 0.685-0.796 PARTIAL. Gemma plain: S 0.715-0.808 SHARED, T SHARED, C 0.623-0.844 PARTIAL, W 0.698-0.820 PARTIAL. Gemma leak-free: lower bounds 0.732-0.782 under all four, SHARED throughout (the only robust Hindi cell). Predictions (2) and (3) met.
- Conclusion: the Hindi->English result is reported as an estimate (about 0.74-0.80, CIs reaching the 0.7 line) and not as a SHARED/PARTIAL label; the registered A16 labels stand as registered but are not used as the headline. EN-ES SHARED is unaffected. Manifest 1469 claims verified (`a17b_*`).

#### 2026-10-26/27 | A18 English<->Hindi span-wise subject interchange (GPU, pre-registered in `docs/HYPOTHESIS.md` A18 before running)
- Runs: `analysis/run_20261026_e2h3{end,bcast}_A_{llama,gemma2}`; analyses `..._e2h3{end,bcast}_analysis_A18` and `_leakfree` (driver `scripts_a18.sh`, log `analysis/_logs/a18.log`). 200 pairs per model per direction (Llama 66 facts, Gemma 55), site A only. Confirmatory pairs: Llama 190, Gemma 186 (leak-free 186 / 151); source entities 60 / 53 (59 / 47).
- Primary variant `end`: P1 gate passes in all 4 model-direction cells (flip at l*: Llama en2hi 0.71, hi2en 0.94; Gemma 0.81, 0.95). So `bcast` is not primary. Prediction (1) met.
- I (95% CI), `end`: Llama en2hi 0.860 [0.803, 0.926], hi2en 0.868 [0.822, 0.917]; Gemma en2hi 0.932 [0.878, 0.990], hi2en 0.807 [0.760, 0.858]. All P2 SHARED (lower bound >= 0.7). Leak-free: Llama 0.859 / 0.866; Gemma 0.953 [0.901, 1.013] / 0.848 [0.803, 0.892]; all SHARED.
- Symmetry (rule iii): |en2hi - hi2en| = 0.008 Llama, 0.125 Gemma (leak-free 0.007, 0.105), both <= 0.15, so direction-symmetric. Prediction (3) met.
- Prediction (2) (en2hi in 0.5-0.9): Llama yes (0.86); Gemma NO (0.93, above range). Prediction (4) (span-wise hi2en within 0.15 of A16 0.74/0.76): met (Llama 0.868, diff 0.127; Gemma 0.807, diff 0.044) but both moved UP, so the patch choice matters: the last-token Hindi estimate of ~0.74 looks like a lower bound, not the shared fraction.
- `bcast` en2hi (secondary): Llama gate FAILS (flip 0.41), I 1.148 not interpretable; Gemma gate passes (flip 0.64), I 1.179 [1.120, 1.253]. Ratios above 1 mean the broadcast carries more than the same-language swap does.
- Not done: A17b clustering sensitivity on A18 labels. Per rule (ii) the SHARED label is therefore not yet a headline; lower bounds under the registered clustering are 0.76-0.90 in all cells.
- Operational: Gemma `end` crashed on a Windows `PermissionError` at the checkpoint rename (partial.npz stale vs partial_meta.json). Repaired by copying `partial_tmp.npz` over `partial.npz`, added a 20x retry loop in `code/interchange.py`, resumed. Final `meta.json` has 400 pair-directions = 200 unique pairs x 2, no duplicates. First `end` analyses (run before Gemma finished) kept as `..._failed_nogemma`. Power cut: run stopped and resumed from checkpoints.
- Manifest 1565 claims verified (`a18_*`).

#### 2026-10-28 | A18b clustering sensitivity of the span-wise EN<->HI ratio (CPU; same A17b method, `code/cluster_sensitivity.py --set a18b`)
- `analysis/run_20261028_cluster_A18b/results.json`. All 8 cells (2 models x 2 directions x plain/leak-free): SHARED under S, T, C and W; labels clustering-robust. Lowest lower bound 0.703 (Gemma hi2en, C clustering, plain; thin), next 0.739 (W). Llama en2hi lower bounds 0.77-0.80. So the span-wise Hindi SHARED label can be used, with the Gemma hi2en plain 0.70 margin stated. Manifest 1565 + `a18b_*` verified.

#### 2026-10-29 | A19 Llama-3.2-3B base replication (GPU, 01:29-03:19; pre-registered in `docs/HYPOTHESIS.md` A19) and A20 more Llama edits (GPU, to 06:23)
- Facts kept by the base model: 81 English/Spanish, 76 Hindi. Prediction (4) (>=100 EN/ES; >=40 Hindi) half met: EN/ES missed, Hindi met.
- Swap ratio I (site A, 95% CI), base model: EN->ES 1.206 [1.107, 1.368] (79 pairs, 33 source entities); ES->EN 0.870 [0.811, 0.921]; HI->EN 0.859 [0.821, 0.896]; EN->HI 0.721 [0.675, 0.771] = PARTIAL (lower bound below 0.7). Same-language gate passes in all four (flip 0.66-0.91). Leak-free numbers within 0.01. Labels identical under all four clusterings (A19b) in all 8 cells.
- Against the instruct model (A13: EN->ES 1.17, ES->EN 0.90; A18: EN->HI 0.86, HI->EN 0.87): differences 0.04, 0.03, 0.01, 0.14. All within the registered 0.15, so all four cells replicate; EN->HI is the weakest (0.14) and is the only PARTIAL. Gate G-A (EN-ES shared on base Llama) met. Predictions (1)-(3) met except (3)'s "within 0.15 in one direction" (met in both).
- A20 (20 new Llama edit facts, 60 records, run dir `analysis/run_20261029_p4a20_llama`), pooled with F7 into 40 facts (the original 10 sweep facts are not pooled): English success traced 0.80 / middle layer 0.93 / layer 21 0.72. Median Spanish transfer 0.58 [0.11, 0.88] / 0.75 [0.64, 0.91] / 0.00. Hindi 0.00 / 0.10 [0.01, 0.36] / 0.00. Specificity loss (median) 0.038 / 0.148 / 0.029. Perplexity ratio 0.9996-1.0004. Predictions: (1) met, (2) NOT met (middle layer exceeds traced by 0.17 for Spanish), (3) met, (4) met, (5) met. Pooled medians were computed ad hoc from `records.jsonl` (not yet in the manifest).
- Manifest 1685 + `a19_*`, `a19b_*` verified.

#### A22 (2026-10-29, CPU)
- Run: `code/split_indnoise.py` -> `analysis/run_20261029_split_indnoise_A22` and `..._leakfree`. Hypothesis: A11 low-name-similarity drop in Llama is a noise artefact of the single-seed profiles.
- Result: Llama informative J = 0: mean D 0.303 [0.192, 0.423] (n = 54), present; J > 0: 0.397 [0.291, 0.505] (n = 49), present; leak-free J = 0: 0.312 [0.202, 0.441]. Gemma J = 0: 0.264 [0.139, 0.402] (n = 21), present; J > 0 has 4 facts. Predictions (1), (2) met; (3) not met; (4) met for Llama only. Verdict: alignment does not require shared subject pieces; name overlap adds about 0.09 in Llama. Manifest 1941 (`a22_*`).


## 10.4 Next steps

*Copied word for word from `docs/NEXT.md` when this record was built. To change it, edit that file and rebuild.*

### NEXT (to-do list, updated 2026-10-30)

#### 2026-10-31 (latest): A28 run read, next steps
- [x] Llama-3.2-3B-Instruct notebook run finished and read (verdicts in docs/HYPOTHESIS.md, A28). Key similarity predicts which facts carry over: Spanish in both tests, Hindi only in the layer-controlled test (narrow). Shared-key fix: no gain. Wording loss in all cells.
- [ ] User rule 2026-10-31: use the models published papers used (order: Gemma-2-2B, Llama-3.2-3B base, BLOOM-560M (XME, has Hindi), mGPT-1.3B (MEMLA), BLOOM-1b7, Qwen2.5-3B, then 7-8B after a T4 test) plus baselines and a benchmark slice on the same facts; Kaggle weekly GPU quota (about 30 h, unconfirmed) limits this to about 4 notebooks a week.
- [ ] **Run next:** `kaggle/babel_gemma2.ipynb` (Gemma-2-2B, registered as A29 in docs/HYPOTHESIS.md, built and CPU-checked 2026-10-31; see the last block of docs/LOG.md). Only input: Kaggle Model Gemma 2, Transformers, gemma-2-2b (base). NOT the dataset leimeng46/gemma-2-2b (it is the 18-layer first Gemma, refused by the notebook). Expected 3 to 5 h on 2x T4 if fp16 is accepted. Untested on a T4. After the Output: read the A29 verdicts, then Llama-3.2-3B base (same pattern, needs its own entry in kaggle/steps.py MODEL, common.py, a config pack and a fingerprint).
- [ ] Replicate on a second model: Gemma-2-2B first (built), then Llama-3.2-3B base. The key-similarity claim needs at least one more model before it is a headline.
- [ ] Rebuild the notebook for new models from the current source (layout fix included); run every job command in a clean unpacked payload first.
- [x] Manifest (2126 claims) and paper Section 4.8 for the A28 numbers, figures 7-8 added (2026-10-31). Still open: full read of the five closest papers before the novelty wording is firmed up; the same section for Gemma-2-2B after the A29 Output.
- [ ] Still open: Hindi and Spanish checker for the audit sheet; OSF/Zenodo timestamp; revoke the Hugging Face token and delete old Kaggle notebooks; more non-geography relations (register first); baselines and benchmark slice (A26 Tests 2 and 3).
- Commit and push only when the user says so.

#### 2026-10-30 (latest): run path and models (master file: GOAL.md)
- [ ] **Run now:** `kaggle/babel_llama.ipynb`, code id bundle-f8371a452538 (Llama-3.2-3B-Instruct; the run in progress is bundle-21a9baef1028). Only input: Kaggle Model Meta Llama 3.2, Transformers, 3b-instruct; no dataset; do not attach the old Output. About 6 h on 2x T4. Send back the Output.
- [ ] After the Output arrives: run the Spanish non-geography analysis twice (plain and leak-free) on the Output's e2a23_A run with the current code; the two Kaggle jobs failed because the laptop tracing run was not packed (fixed in kaggle/layout.py, notebook not rebuilt; rebuild before the Gemma and Llama-base notebooks).
- [ ] Read the A28 verdicts from the Output (key similarity beyond the edit layer, layer map, shared-key and wording rules at about 3x the facts).
- [ ] Next notebooks, in order: Gemma-2-2B, Llama-3.2-3B base (both need their Kaggle Model paths), Qwen2.5-3B, BLOOM-1b7 (needs a larger fact pool), then one 7-8B model after a T4 test. Register each model in docs/HYPOTHESIS.md before its run.
- [ ] Still open: Hindi and Spanish checker for `data/audit/native_speaker_audit.csv`; OSF/Zenodo timestamp; revoke the Hugging Face token and delete the old Kaggle notebooks; more non-geography relations (register first); benchmark slice and baselines (A26 Tests 2 and 3).
- Commit and push only when the user says so.

#### 2026-10-30 (later): Kaggle only, make the new results the best
- [x] Key-similarity test run on the Kaggle data: traced layer rho 0.55 (Spanish) and 0.52 (Hindi), layer 14 none; weakens when the edit layer is controlled (Hindi 0.06). Registered A28 to settle it. New notebook `kaggle/babel_llama.ipynb` (code id bundle-6d5af7908026): traces on Kaggle, about 85 facts, layer map, key similarity, all in one run (about 6 h on 2x T4).
- [ ] User runs the notebook and sends back the Output; then read the A28 verdicts (key similarity beyond the edit layer, layer map, shared-key and wording rules on the larger runs).
- [ ] If the within-layer relation holds: write the contribution as "key similarity predicts which facts transfer, at early layers"; if not: the diagnostic paper reports the layer map and the null shared-key result.
- [ ] Then the next model notebooks (Gemma-2-2B, Llama base; need their Kaggle Model paths) with the same step; baselines (MEMIT-style, in-context on held-out wordings) and the benchmark slice stay registered in A26 Test 2 and 3.
- Commit and push only when the user says so.

#### 2026-10-30: first Kaggle results (Llama-3.2-3B-Instruct, verdicts in docs/HYPOTHESIS.md)
- [x] Paraphrase control, shared-key edit, non-geography facts, tokenizer comparison run on Kaggle (73.9 min, 2x T4, fp16). Shared-key edit: no improvement (Plan B stands). Non-geography facts: untestable (53 EN/ES and 13 EN/HI facts, two relations). Paraphrase control: cross-language loss beyond wording for Hindi and for Spanish at the traced layer; single-edit numbers do not reproduce the laptop's (median-level numbers do).
- [ ] Decide with the user: next model notebook (Gemma-2-2B, then Llama base; same pattern, needs the Kaggle Model path), and what to do about the single-edit non-reproduction (rerun the laptop edits in fp16? report as a limitation?).
- [ ] A23 needs more relations that Llama answers (currency and profession failed the English template filter); register before any new facts are run.
- [ ] Key-similarity per-fact test (register first): the A24 records already hold key cosines per fact.
- [ ] Hindi + Spanish checker, OSF/Zenodo timestamp (still open). Commit and push only when the user says so.

#### 2026-10-29 (late): Kaggle rerun and the contribution
Chosen contribution: mechanism + fix. Claim: the read-out is shared, but a rank-1 edit is keyed on a language-specific form of the subject; key similarity across languages predicts per-fact edit transfer; a key built from several languages closes the gap without training. Compare with ROME, MEMIT, in-context and one published multilingual editor (language-agnostic factual neurons 2406.16416 or MEMAT 2502.02173) on our controlled facts and a Bi-ZsRE/MzsRE slice. Plan B: diagnostic paper (likely Q2-Q3).
- [x] Kaggle engine: per-model notebooks `kaggle/notebooks/babel_llama.ipynb`, `babel_gemma2.ipynb` (code from GitHub, no dataset to upload). Stage 1 (fact preparation) only; GPU parts untested until the first run.
- [ ] User runs both; I compare kept facts with the laptop (Llama 93, Gemma 58) and read the timings.
- [ ] Add the stages to the same notebooks: tracing, noise/phrasing, swap tests, edit sweep, in-context, paraphrase control, new relations, shared-key edit; then the Llama-base notebook.
- [ ] Pre-register before running: the key-similarity test (per-fact rho between EN-ES/EN-HI key similarity at the edit layer and transfer), the baselines, the benchmark slice (check which languages MzsRE has).
- [ ] Read in full: Fierro 2410.14387, Dumas 2411.08745, Ifergan 2408.10646, CLM-Bench 2601.17397, LU-LAFNs 2406.16416, MEMAT 2502.02173; update related work.
- [ ] Needs the user: Hindi + Spanish checker, OSF/Zenodo timestamp, Kaggle weekly GPU quota.

#### 2026-10-29: plan to raise the level (see `GOAL.md`)
Order: A21 -> A23 -> A24 (needs A21) -> bigger models (7-9B, 4-bit; needs a feasibility check and the user's decision on a free cloud GPU) -> more languages -> benchmark slice.
- [x] CPU: paper repairs (A22 reconciliation, 2 citations, p-floor note, abstract/discussion/limitations, PDF rebuilt); A23/A24 pre-registered and coded; native-speaker audit sheet built.
- [ ] GPU, needs the user's laptop window and "go": `./scripts_a21.sh` (about 3 h), `./scripts_a23.sh` (about 3-4 h), then `./scripts_a24.sh` (about 6 h, after A21).
- [ ] Needs the user: a Hindi and a Spanish checker for `data/audit/native_speaker_audit.csv`; yes/no on a free cloud GPU for 7-9B; Zenodo/OSF timestamp.
- [ ] CPU, after each run: manifest blocks (`a21_*`, `a23_*`, `a24_*`), paper paragraphs, rebuild PDF, refresh `..\evaluator_pack`.
- [ ] CPU later: 2-3 more languages (candidate facts), a mLAMA/Bi-ZsRE slice builder, 4-bit feasibility test script.

#### Older list (2026-10-25)

#### 2026-10-25 status
- [x] A12/A12a confound analysis, A13a independent-noise re-trace, A13b covariates, A13 E2 interchange (verdict SHARED EN-ES, both models; docs/EXPERIMENT_LOG.md). Paper section 4.7 drafted.
- [ ] GPU (needs laptop): E2 Hindi; E3 name-dissimilar facts (more relations); E4 base vs instruct. (Gemma F7 done 2026-10-25; Llama-1B pilot kept 38 facts and BLOOM-1b7 pilot 34, both below the 50 needed, so no full run.)
- [ ] CPU: related work + arXiv-verified citations; Llama in-context baseline anomaly (EN 0/10); main tables with 10,000 permutations; Fig. 4 buckets; reproduce script + clean-clone run; external pre-registration timestamp (Zenodo/OSF); full paper rewrite and abstract update.
- [ ] Open science question: tracing increment does not predict interchange per fact in Llama (rho -0.04); explain or drop the link.


Legend: [x] done, [~] running, [ ] to do. GPU items wait for the layer sweep (scripts_run_sweeps.sh). It was interrupted on 10-15 and 10-18 (laptop sleep) and resumed on 10-19 03:07 from saved rows; scripts_watchdog.sh restarts it if it dies. Estimate: Llama about 4-5 h more, Gemma about 6 h, then the queued chains.

#### Done
- [x] P0-P5 on Llama-3.2-3B (v1 pilot) and paper draft v1; repo public (praann07/babel-effect)
- [x] Model research report (reports/Models for multilingual fact editing.md), Gemma-2-2B weights verified
- [x] A5-A8 pre-registered; v2 data (5 relations, 3 phrasings); full-answer tracing; Llama v2 (n=92) and Gemma-2-2B v2 (n=58) overlap; cross-model table (33 shared facts); related work (20 verified refs); manifest 356 claims

#### Fixes to make the paper reviewer-proof (do one at a time, commit after each)
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

#### Hypotheses status (docs/HYPOTHESIS.md, Verdicts section)
- H6 supported (both models). H7 supported in Gemma, contradicted for ES in Llama. H8 supported (both). H9 not supported. F4 met for Llama, not Gemma. F3/F7 descriptive (Llama F7 done).
- B1-B3 (A10, BLOOM-1b7): pending.

#### Left to run (GPU; start with Desktop "Start Experiments.bat" or the two nohup commands in the queue2 script header)
1. Gemma F7 (20 new facts; partial dir analysis/run_20261020_p4f7_gemma2 is kept unused, the queue starts a fresh one)
2. BLOOM-1b7 pilot (A10; TransformerLens BLOOM path untested)
3. Llama-3.2-1B pilot (full pipeline only if >= 50 facts)

#### Left to write (CPU)
- Add Gemma F7 and the pilots to paper section 4.5/5 once run; related-work additions after verifying 2411.08745; README refresh; reproduce.sh for the v2 pipeline + one clean-clone rerun; fix Start Experiments.bat.

#### Waiting on the user
- Venue choice; optional Hindi-speaker spot check; keep the laptop plugged in while the GPU queue runs.

#### 2026-10-29 (night): laptop only; comparison with published work done
- Kaggle dropped. Registered in docs/HYPOTHESIS.md as A26: key-similarity test, baselines (MEMIT-style, in-context, published editor), Bi-ZsRE English-Chinese slice on Qwen2.5-3B.
- Next, in order: (1) go from the user, then the paraphrase control, shared-key edit and non-geography runs; (2) CPU: key-similarity analysis script; (3) write MEMIT-style editor, Bi-ZsRE loader and scorer, Qwen entry in code/common.py, with CPU tests; (4) read MEMAT (2502.02173) and the Fierro, Dumas, Ifergan papers in full; (5) native-speaker audit and OSF/Zenodo timestamp.
- Finding from the reading: Bi-ZsRE has English and Chinese only, so Spanish and Hindi stay on our controlled facts. LU-LAFNs needs 30 paraphrases per question (the authors used a 72B API), which is a deviation under the $0 rule.


## 10.5 Q1 roadmap

*Copied word for word from `docs/ROADMAP_Q1.md` when this record was built. To change it, edit that file and rebuild.*

### Roadmap to a Q1-level paper (written 2026-10-26)
Goal (user, m00927): target a Q1 journal. Honest gap: today = 2 small models (3B/2B), 4 geography relations for interchange, EN/ES/HI, mostly descriptive edits. Q1 reviewers will want breadth (models, languages, relations), a mechanism, and an intervention result. Venue is decided after the evidence; no quartile is asserted here (check SCImago/JCR).
Constraint check: $0 and one 8 GB GPU. Free 16 GB GPUs (Kaggle/Colab T4) would allow 7-9B models in 4/8-bit; needs the user's OK (outward-facing: uploads code/data).

#### Phase A - close the known holes (local, 8 GB, ~3 weeks)
A1 E4 Llama-3.2-3B base (overlap + interchange)       4-6 GPU h   removes instruct/base confound
A2 A18 EN->HI span-wise subject patch (pre-register)   2-3 GPU h   closes the one-direction Hindi gap
A3 Edits n=100 x {traced, L//2, A9 layer} + EN-paraphrase control   8-12 GPU h
A4 Paper-text fixes (abstract vs 4.7, A11 vs A13b, Hindi as estimate, inline n/CI, p-floor)   CPU
#### Phase B - breadth (what Q1 reviewers ask first)
B1 Models: add >=2 more families at >=7B (e.g. Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B) - needs 16 GB GPU or 4-bit; keep 2 small ones   ~30-40 GPU h
B2 Languages: add 2-3 typologically different (e.g. Chinese, Arabic, Swahili/low-resource) to EN/ES/HI; native-speaker audit of kept facts
B3 Relations: add non-geography (people/occupations, language-of-country, element symbols; Wikidata name-dissimilar facts = E3) so the claim is not "geography only"
#### Phase C - mechanism + intervention (what makes it a Q1 contribution, not a measurement note)
C1 Which components carry the shared state: attention-head vs MLP interchange, path patching on the subject->answer route; where language-specific parts (tokenisation/name) enter
C2 Predictive use: does the shared-ness score predict cross-lingual edit transfer per fact (P6-style validity across models)? A model of when edits transfer
C3 Practical method: a cross-lingual edit that targets the shared layer, with specificity/damage, compared with MEMIT/ROME-multilingual baselines (Khandelwal/Wang style)
#### Phase D - rigor and release
Pre-register each phase in docs/HYPOTHESIS.md before data; keep manifest/verify; external timestamp (Zenodo/OSF DOI - user); replication package; ablate every choice (thresholds, clustering, templates).
#### Decision gates
G-A (after Phase A): EN-ES shared holds on base Llama and EN->HI is not much lower than HI->EN -> continue to B. If not: narrow to the audit/pitfalls paper (TMLR/workshop).
G-B (after B): claim holds on >=4 models, >=5 languages, >=3 relation families -> do C. If it holds only on some, the paper becomes "when is it shared" (still publishable, stronger).
G-C: a mechanism or an edit method that beats baselines -> submit to Q1; otherwise TMLR.


## 10.6 Outside review of 2026-10-23

*Copied word for word from `docs/REVIEW_2026-10-23.md` when this record was built. To change it, edit that file and rebuild.*

### The Babel Effect: review, audit and 4-week plan (2026-10-23)

What I read: `01_STATE_OF_PROJECT.md`, `docs/*` (HYPOTHESIS A1-A12, EXPERIMENT_LOG, LOG, NEXT, FUTURE_WORK, both prompts, status report), `paper/paper_v2_draft.md` and `main.tex`, `results/*.json`, and the code (`causal_trace_v2.py`, `circuit_overlap.py`, `controls_overlap.py`, `h8_analysis.py`, `incontext_baseline.py`, `knowledge_edit_v2.py`, `select_templates.py`, `build_facts_v2.py`, `common.py`, `matching.py`).

What I could not do: recompute anything. The bundle has no `analysis/` grids and no `data/v2/`. Every number below is either copied from your files (with its source) or labelled **prediction**.

---

#### 1. Honest assessment

What the project demonstrates today, in 5 lines:

1. A transparent, pre-registered, fully logged pipeline (589 verified numbers) for comparing causal-tracing profiles across EN/ES/HI in two small models. The process is the strongest part of the work.
2. Same-fact EN-ES tracing profiles are more alike than different-fact profiles of the same relation, in both models (mean-cosine increment 0.090 in Llama, 0.182 in Gemma; A11 C0).
3. That increment sits where surface forms are shared (identical subject names in Llama, identical digit answers in Gemma). Also, the tracer reuses one noise seed per fact across languages (Section 2.1). So it is not yet evidence of shared lookup.
4. Hindi alignment is weaker and depends on span length. ES edit transfer is moderate and HI transfer near zero. Both are exploratory and underpowered.
5. There is no interventional test of sharing yet. The core question is still open.

Verdicts (my subjective estimates, roughly ±15 points):

| Tier | Accept today | After the 4-week plan, best realistic branch | Main reason |
|---|---|---|---|
| (a) Q1 journal (TACL, CL) | < 2% | 5-10% | Two small models, about 100 facts, 3 languages, and the core question is unanswered. A Q1 paper needs months more (Section 5, "What Q1 would take"). |
| (b) TMLR | ~10% | 50-60% (shared or partial result), 40-50% (negative), 25-35% (cautionary only) | TMLR asks whether claims are supported by evidence, not whether they are novel. Today the main claim rests on a confounded correlational statistic. |
| (c) Findings (ARR) | ~5% | 25-35% | Novelty against Dumas et al. (2411.08745) and Lu et al. (2505.20546, same Llama model). Small models. |
| (d) Workshop | 40-50% | 75-85% | Honest and well documented. The risk is a reviewer who spots the seed artefact. |

I am not sure whether TMLR has a Scimago quartile. Check this if a "Q1" label matters for you (for example, for a scholarship).

---

#### 2. Audit of your reasoning, controls and numbers

##### 2.1 A design artefact neither review raised: the same corruption noise is used in every language

- `build_facts_v2.py:235`: `seed = sha256(fact_id)`. It is one seed per fact.
- `causal_trace_v2.py:182`: `trace_one(..., f["seed"], 3.0*std[lg], ...)` uses the same seed for EN, ES, HI **and** for all phrasings.
- `causal_trace_v2.py:88-89`: `noise = randn(len(span), d, generator=seed) * sigma`. Subject token k receives noise row k in every language. Only sigma differs between languages.
- Consequence 1. The same-fact pair (EN_i, ES_i) shares its noise realisation. Every null pair (EN_i, ES_j) does not. The shuffled-fact null therefore does not hold the noise fixed. Same-fact alignment can come partly from shared noise alone. At 3 sigma, the noise vector dominates the corrupted subject embedding.
- Consequence 2. When EN and ES subjects have the same number of tokens, the measured last subject token gets the same noise row. When the lengths differ, only the earlier rows are shared.
- Consequence 3, the strongest. Most pool templates are subject-first ("{S} is the capital of", "{S} took place in the year"; all Hindi templates are subject-first). With a subject-first template and an identical name, the token prefix up to the measured site is identical in EN and ES (BOS + subject). The clean activations are then identical at every layer, and the corrupted ones are identical up to sigma. These facts carry **no** cross-language information by construction. Llama has 46 identical-name facts (36.5%).
- This one mechanism predicts four patterns you observed: EN-ES > EN-HI (Hindi spans of 5-8 tokens never share the last-token noise row with 2-token English spans); the Hindi span-length dependence; Llama's name-similarity dependence; and H8 (Llama EN within-phrasing cosine 0.985, because the prefix and seed are identical across phrasings).
- Status: a confound that must be ruled out. It is not proven to explain everything. The test is cheap (E0, Section 3.9).

##### 2.2 Centring turns magnitude into direction

- Profiles are raw log-probability effects. Their norm scales with the fact's total effect.
- If p_i = a_i (g + e_i), where g is the generic shape and a_i the magnitude, then the centred profile is c_i ≈ (a_i − ā) g + a_i e_i. When g is large (stereotyped profiles), the centred cosine of the same fact is roughly the sign of (a_i^EN − ā)(a_i^ES − ā). It becomes positive whenever total effects are correlated across languages. That is likely, because answer token counts are shared (identical for event_year).
- The symptom is in your numbers. In Gemma, centring raises z from 3.24 to 5.43. The observed median barely moves (0.436 → 0.410) while the null mean collapses (0.325 → 0.082).
- Fix: L2-normalise each profile (or divide it by its total effect) **before** centring, or project out the mean direction (p − (p·m̂)m̂). Report the increment D on these profiles. The uncentred cosine is scale-invariant and not affected. Only the "survives centring" claim is.
- A12 S1 relation-centres the same unnormalised profiles, so it inherits the problem.

##### 2.3 "EN-EN different-fact 0.606 > same-fact EN-ES 0.570, so the shape is language-specific" is a misreading

- Llama different-fact means: EN-EN 0.606, EN-ES 0.480, ES-ES 0.413. EN-ES lies **between** the two within-language values.
- If the generic shape were language-specific, EN-ES would sit below both. What the data show is that English profiles are more stereotyped (less variance across facts) than Spanish ones. The generic shape is shared.
- The right comparison is the increment against a within-language noise ceiling: two independent-seed replicates of the same language (E0 provides them). Delete the sentence in Section 4.6.

##### 2.4 Relation-stratified null

1. It is valid for H0 "EN and ES profiles are independent within relation". Rejecting that H0 does not identify shared lookup. Any fact-level covariate shared across languages rejects it: noise seed, answer tokens, answer length, total effect, subject identity, entity frequency. The z-values are real. The interpretation in the paper is not supported.
2. The permutations allow fixed points. Each stratum adds about one same-fact pair per permutation, and Gemma's single extra fact is a stratum of size 1, so it is always paired with itself. The null mean is slightly inflated, which makes the test conservative. Use within-stratum derangements, or the exact off-diagonal mean.
3. z from a non-Gaussian null of medians: report the empirical p, (1 + b)/(B + 1), and the effect size (observed − null mean, with CI). The 1000-permutation and 10,000-permutation runs give different z for the same statistic (Llama 4.24 vs 4.41, Gemma 3.18 vs 3.24). Keep the 10k run only.

##### 2.5 Name-similarity split

1. The thresholds are model-specific medians (Llama 0.833, Gemma 0.561). "Brazil/Brasil" (ratio about 0.83) falls in Llama's "low" half. The two low halves cover different ranges and should not be compared.
2. A character-level difflib ratio is the wrong variable. The computation sees tokens. Use: identical token prefix up to the measured site, equal span length, identical last-subject-token id, and token-level Jaccard.
3. **The event_year confound runs in opposite directions in the two models.** All 22 Llama event_year facts are in its low half, and Llama event_year has EN-ES within-relation z = 0.91 (LOG, n = 92 run). It *dilutes* Llama's low half. In Gemma, 20 of 30 low-half facts are event_year, and Gemma event_year has z = 2.07 (cosine) / 3.91 (Spearman). It *drives* Gemma's low half. "Gemma clean, Llama lexical" is therefore confounded with relation mix.
4. "Significant in one half, not in the other" is not a test of difference (Gelman & Stern 2006). The effect sizes are similar: Spearman 0.29 (Llama) vs 0.22 (Gemma). Gemma's p = 0.097 reflects n = 59, not cleanliness. Gemma's high-similarity half is *weaker* (uncentred z 1.95) than its low half (2.80), which fits event_year driving the result.
5. Code bug, `controls_overlap.py:24`: `spearman` breaks ties by order. In Llama, 46 of 126 similarities are exactly 1.0, and ids are sorted by relation name, so tied ranks follow relation order. Use average ranks (`scipy.stats.spearmanr`).

##### 2.6 Gemma's non-copyable alignment is already weak in your own logs

- With its own null per relation (LOG, n = 58): capital_country n = 11, EN-ES z = −0.74; capital_of n = 11, z = 1.14 (cosine).
- The non-copyable z = 2.53 (n = 29) is against the **pooled** null, which counts relation-level similarity.
- **Prediction:** A12 S1 on Gemma's non-event_year facts will have a CI that includes 0. If so, the honest reading is that in both models the fact-specific profile similarity sits where surface forms are shared: subject strings in Llama, answer strings in Gemma.

##### 2.7 Entity clustering is nearly vacuous here

- Gemma has 59 facts in 56 clusters, and Llama 126 in 81. The cluster CI lower bounds equal the plain-bootstrap bounds to 7 digits (0.59561, 0.32315). Bootstrap medians are order statistics, so clustering that merges few facts changes nothing.
- It also bootstraps the raw median cosine, which includes the generic shape, not the increment. Apply the cluster bootstrap to D (A12 S1 does this).

##### 2.8 Mean vs median, and other number checks

1. 0.570 and 0.424 (paper Section 4.6, A11 C0) are **means** of the per-fact diagonal. 0.642 and 0.436 (Table 2, A11 C1) are **medians** of the same vector. Both are computed correctly. Llama's mean is 0.07 below its median, so there is a long left tail. Use one primary statistic: the **mean increment D**, because means decompose (same − different). Report the median as robustness.
2. Gemma: the mean different-fact EN-ES cosine is *lower* within relation (0.242) than across relations (0.264). Yet the stratified null of the median statistic (0.325, A11) is higher than the pooled null (0.292, LOG n = 58). Mean and median disagree on the sign of relation-level similarity. This is one more reason to use a mean-based increment with an exact within-relation baseline.
3. Llama EN-HI stratified z is 2.6648 in the manifest. The text and the state file say 2.67 (Sections 4.5 and 5); Table 2 says 2.66. Fix the rounding.
4. ES-HI z = 1.98 in both tables is a genuine coincidence: 1.9796 (n = 92) vs 1.9753 (n = 126), both in the manifest.

##### 2.9 Claimed findings, re-graded

| Claim in the draft | Status |
|---|---|
| EN-ES alignment beyond the shuffled-fact null | Real as a dependence. Interpretation confounded (2.1, 2.4). |
| "Survives centring" | Possibly a magnitude artefact (2.2). Unknown until normalised. |
| "Gemma is clean of lexical overlap" | Not supported (2.5, 2.6). Likely event_year, i.e. answer identity. |
| "Llama profile shape strongly language-specific" | Misreading (2.3). |
| H8 "robust to rewording" | Largely mechanical: identical prefix and seed across phrasings (2.1). |
| "Entity clustering leaves the CI unchanged" | Near-vacuous control (2.7). |
| Hindi weaker and span-dependent | Confounded by noise-row alignment and by single-token restoration in long spans. |
| Llama critical layer 0-6 | Plausibly the model repairing embedding noise (detokenisation), not lookup. Gaussian-noise caveat (Zhang & Nanda 2309.16042). |
| Gemma layer-14 cluster | 9 of 17 English layer-14 argmaxes are event_year, so it is relation mix. |
| "Llama alignment does not depend on copyable answers" (non-copyable z 4.87) | True, but it depends on shared *subjects*. Pooled null. |
| In-context EN 0/10 | Probably not a code bug. The EN and ES code paths are identical except the test prompt (`incontext_baseline.py:44-53`), and the EN test prompt exactly repeats the context sentence, which an instruct model may "correct". Generations are not saved, so it cannot be diagnosed from disk. |

##### 2.10 Fix A12 before you compute it

Do not edit A12. Add **A12a**, committed and externally timestamped before any A12 number exists:

1. S4 intercept. With relation fixed effects, the intercept is the *reference relation*. It is evaluated at log(total effect) = 0 (1 nat, arbitrary) and at name similarity = 0, which is an extrapolation in Llama. Its CI will be huge, so A12's "no name-free alignment" rule can never fire. Fix: effect-code relation, centre the continuous covariates, and report the *predicted D at token-Jaccard = 0* with a CI.
2. Add covariates: identical-prefix flag, equal-span flag, identical last-token flag, token Jaccard, and log total effect in **both** languages.
3. Normalise profiles before centring (2.2), and report both versions.
4. The SESOI is "one third of Gemma's D". It is data-dependent, and Gemma's D may itself be an artefact. Add a fixed SESOI: 0.05 absolute increment in normalised-centred cosine (about half of Llama's raw increment).
5. Derangement nulls, average-rank Spearman, mean D primary.
6. "answer-identical" is collinear with the event_year fixed effect. It is identified only from non-event_year identical answers (e.g. "Madrid"). Say so.
7. S2 retrieval: chance differs by relation size (1/n_r). Report against per-relation chance.

Then run A12 as written **and** A12a. Report both.

---

#### 3. The core question: language-specific circuits vs shared space

##### 3.1 Is it answerable with the current method? No, for three reasons

1. **Identifiability.** A correlation between two separately computed causal profiles cannot separate "one shared mechanism" from "two mechanisms whose properties co-vary with fact-level covariates".
2. **The measured site is context-free in subject-first templates.** The last-subject-token state depends only on BOS + subject, so "language" there means "the subject's name". The comparison asks whether "Germany" and "Alemania" are enriched alike, not whether English and Spanish processing share a circuit.
3. **A 28-number depth profile is not a circuit.** Different circuits can have the same depth profile (Hanna et al. 2403.17806, on overlap vs faithfulness).

It is answerable in an operational form: *is the subject representation computed in language A at layer l functionally interchangeable, for the downstream recall computation, with the one computed in language B?* This is an interchange intervention (Geiger et al. 2106.02997). The binary "circuits vs space" is a false dichotomy. The literature points to language-specific input/output with a shared middle, so the answerable question is **where and how much** interchangeability holds.

##### 3.2 Candidate designs compared

1. **Cross-language subject interchange, different-fact source (swap).** It reads out a *specific counterfactual answer*, so generic shape, magnitude and corruption noise cannot produce a positive result. It is cheap and reuses your scorer. **Recommended primary.**
2. **Same-fact cross-language restoration** (corrupt ES, restore the EN activation of the same fact). This is weak: recovery can come from generic information ("a country") or from undoing noise. Adding the needed different-fact control turns it into (1). Keep it as the "self" control inside (1).
3. **Name-dissimilar facts.** A stratifier or covariate, not a design. It is needed for any analysis, interventional or correlational. EN-ES supply is limited (Section 3.5).
4. **Base vs instruct.** A confound check, not decisive for the core question. The cheapest version runs (1) on Llama-3.2-3B base (same tokenizer and code path). Optional for TMLR, recommended for Q1.
5. **Better, but too heavy or risky for 4 weeks:** DAS / RAVEL-style subspace interchange (2303.02536, 2402.17700) needs gradient training on 8 GB and carries the subspace-illusion risk (2311.17030). Cross-lingual LRE (2308.09124) needs Jacobians. Probing/RSA is cheap but has the same identifiability problem as now.
6. **Cheap add-ons that strengthen (1):** a final-position readout (Dumas et al. 2411.08745 style; answer language vs answer concept), and a language-offset patch (add μ_ES − μ_EN to the EN source; tests "shared content + language offset", Libovický et al. 2004.05160).

##### 3.3 Recommended design: E2 (A13), cross-lingual subject interchange

Facts:
1. Re-filter the full candidate pools (408/447 + extra relations) for **EN and ES correctness only**. Do not require Hindi for the EN-ES question: requiring it selects famous, name-sharing entities.
2. Use **prefix templates**: at least 2 language-specific tokens before the subject (EN "The capital of {s} is" / ES "La capital de {s} es"; "The country where {s} is located is" / "El país donde está {s} es"). This gives the source representation a language context. Keep the A7 template if it is a prefix template. Otherwise pick the best prefix template on the 30 tuning facts per relation, and **exclude those facts** from evaluation (held-out selection).
3. Exclude event_year and invention_inventor from the primary set (identical answers).

Pairs:
4. Ordered pairs (target t, source s), same relation, with different first answer tokens in the target language. Use two seeded sources per target.
5. The **confirmatory subset** is defined on the *source*: the EN and ES subject token sequences of s share no token (compare decoded pieces, lower-cased, leading-space marker stripped). Targets need no restriction.

Intervention:
6. Site A (primary): `blocks.{l}.hook_resid_post` at the last subject token of the target ← the source's value at its last subject token, one layer per row, l = 0..L−1 in one batch. Site B (secondary): the same at the final prompt token.
7. Pre-registered fallback, used only if the ceiling gate fails: patch the last min(|s|, |t|) subject positions aligned from the end.

Conditions per pair, layer and direction (EN→ES primary, ES→EN secondary):
- `same`: source = s in the target language. This is the ceiling.
- `cross`: source = s in the other language. This is the test.
- `self`: source = t in the other language. Retention: does the other-language representation of the same subject stand in?
- `ctrl`: source = s in the other language, but taken at the token just before the subject. This is the disruption floor.

Readout and statistic:
8. Readout: teacher-forced full-answer log-probabilities (your `SeqScorer`). Δ(l) = change in [lp(a_B(s)) − lp(a_B(t))] relative to the unpatched target. At site B, also score lp(a_A(s)) to separate concept from language.
9. Band B: layers where mean Δ_same(l) ≥ 0.5 × max_l mean Δ_same(l). It is computed from the ceiling only, before anyone looks at `cross`.
10. Primary statistic: I = Σ_{l∈B} mean_p Δ_cross(l) / Σ_{l∈B} mean_p Δ_same(l), EN→ES, confirmatory subset. A ratio of means avoids small per-pair denominators. CI: cluster bootstrap over source entities, 2000 resamples.
11. Secondary: the per-layer curve I(l); ES→EN; all pairs; flip rate (greedy decode = a_B(s)) at l* = argmax mean Δ_same.
12. Covariate model: per-pair r_p regressed on source token-Jaccard, span difference and relation (effect-coded). Report the predicted r at Jaccard = 0.
13. Validity test of your old metric: Spearman between the per-fact interchange score and the per-fact normalised-centred, independent-seed profile increment from E0.
14. Hindi (exploratory): EN→HI with an HI→HI ceiling, on facts correct in EN and HI.

##### 3.4 Controls and sanity tests (unit tests, before the real run)

1. Patching the target with its own clean activation gives Δ = 0 exactly.
2. Same-language swap on single-token subjects at layer 0 gives a large Δ.
3. TL vs HF logits on 20 prompts per model (max abs difference). Your current check compares TL against TL.
4. Behavioural code-switch reference: accuracy of "La capital de Germany es"-style prompts. If code-switched prompts work, a high I at early layers is expected; the informative layers are those after enrichment.

##### 3.5 Sample size and power

- Unit: pair. Dependence: through the source entity, so cluster by source.
- Assume the per-pair normalised effect has SD σ = 0.4-0.7 (unknown).
- CI half-width h = 1.96 σ / √n_eff. For h = 0.15: n_eff = 27 (σ = 0.4) to 84 (σ = 0.7).
- The decision thresholds (0.3 and 0.7) are 0.4 apart, more than 2h. So at most one of "shared" or "specific" can be declared. A true I near 0.5 lands in "partial", as it should.
- Detecting any sharing (I > 0) at α = 0.01 with power 0.9 for a true I = 0.3 and σ = 0.5 needs n = ((2.33 + 1.28) × 0.5 / 0.3)² ≈ 36.
- **Target: ≥ 120 pairs from ≥ 40 distinct token-disjoint source entities per model. Floor: 25 source entities.** Below the floor, extend with Wikidata EN/ES labels (free SPARQL) filtered by token-disjointness before any model run. Run a 10-pair debugging pilot and exclude it from the analysis. Do not re-estimate n from data.
- I do not know how many token-disjoint EN-ES sources your pool has. Count them first (CPU, tokenizer only).

##### 3.6 Pre-registered predictions and decision rules

| ID | Statement | Rule |
|---|---|---|
| P1 (gate) | Same-language swap works | Flip rate ≥ 60% at l* and mean Δ_same(l*) ≥ 2 nats, in both models. If it fails, use the fallback (item 7). If it fails again, E2 is uninformative (method failure, not evidence). |
| P2 | Shared | I (EN→ES, confirmatory) 95% CI lower bound ≥ 0.7. |
| P3 | Language-specific | 90% CI upper bound ≤ 0.3 (TOST-style). |
| P4 | Partial | Anything else. Report I and the layer curve I(l). |
| P5 | Lexical | Predicted r at Jaccard = 0 not lower than at Jaccard = 1 by more than 0.2. Otherwise interchange is also lexical. |
| P6 | Metric validity | ρ(old profile increment, per-fact I) > 0.2. If ρ ≈ 0, the tracing-profile metric has no validity as a sharing measure. That is reportable. |
| P7 (exploratory) | Hindi | I(EN→HI) < I(EN→ES). |

Direction priors: the literature (Dumas et al., Fierro et al., Lindsey et al. 2025) predicts P2 or P4 for EN-ES at middle layers. State that prior in A13.

##### 3.7 What each outcome means

- **Shared:** subject representations in the mid layers are interchangeable across EN and ES even for token-disjoint names. Facts are retrieved from a representation the Spanish context can use when it is computed in English. This does *not* prove a single "space" for all languages, and Hindi may differ.
- **Specific:** the Spanish recall computation cannot use an English-computed subject representation at that site, even though tracing profiles correlate. This is a strong cautionary result. Reviewers will push on the method, so the ceiling, ES→EN and the site-B readout must all be shown.
- **Partial or layer-dependent:** the degree of sharing is quantified. A typical pattern would be low at early layers for dissimilar names, rising in the middle, falling after extraction.
- **Ceiling fails twice:** no claim about sharing. The paper becomes the cautionary correlational paper.

##### 3.8 Compute and VRAM

- Per pair and direction: about 8 batched forwards of L rows (4 conditions × 2 sites) plus about 3 cache forwards, roughly 2-5 s on your GPU.
- 120 pairs × 2 directions × 2 models is about 40 min. Add Hindi and the Llama base model and it is still one night.
- Peak memory is similar to your tracing (around 7.5 GiB for Llama), because the batch of 28 short rows is small and your scorer already chunks the vocabulary.

##### 3.9 E0 (prerequisite, runs first): does the old result survive independent noise?

- **E0-CPU (A12a, 1 hour):** for every fact, classify EN-ES as *identical prefix* (token ids equal up to the measured site), *equal span length*, or *other*. Compute the normalised-centred D within each class and within relation. A sketch is in Appendix A.
- **E0-GPU (A13a, one evening):** a reduced tracer that patches only `hook_mlp_out` at the last subject token, all layers in one batch.
  - Seeds = sha256(id | lang | k), with K = 10 draws per fact and language, split into two independent replicate sets of 5.
  - Cost: 126 × 3 × 10 batched passes for Llama (~30 min), half that for Gemma.
  - Outputs: (i) D_EN-ES with independent seeds; (ii) the within-language replicate increment D_EN-EN′, D_ES-ES′ (the noise ceiling); (iii) a disattenuated sharing index D_EN-ES / √(D_EN-EN′ · D_ES-ES′); (iv) a regression test that one shared seed reproduces your cached profiles exactly for 3 facts.
- Gate G1 (pre-registered):
  - *Survives:* D_EN-ES (independent seeds, averaged, normalised-centred, non-identical-prefix, non-event_year) has a 95% cluster CI excluding 0.
  - *Artefact:* the CI includes 0 **and** the upper bound is < 0.05 (SESOI).
  - *Unresolved:* anything else.

---

#### 4. Fix list for the Section 8 problems

Key: W = workshop, F = Findings, T = TMLR, Q = Q1. **E** = essential, o = optional, cut = remove instead of fixing.

1. **Main claim not established.**
   - Fix: A12 + A12a (CPU, 1 day); E0-CPU + E0-GPU (half a day + 1 evening); E2 (3 days of code, 2 nights).
   - Done when: the G1 and G2 gates have pre-registered outcomes in the manifest.
   - Tiers: A12a and E0 are E for W, F, T and Q. E2 is o for W and E for F, T and Q.
2. **Method validity (profile cosine is not a circuit).**
   - Fix: E2 plus the P6 validity correlation. Rename "circuit overlap" to "tracing-profile similarity" everywhere.
   - Done when: the main claims rest on E2 only.
   - Cost: included in item 1. Tiers: the rename is E for all; E2 is E for F, T and Q.
3. **In-context baseline.**
   - Fix: cut it. If you keep it: save the generations, use a paraphrased query (phrasing 1, not an exact repeat), n ≥ 30, and add the Llama base model.
   - Done when: EN/ES/HI behave consistently, or the anomaly is explained from saved generations.
   - Cost: 1 evening. Tiers: cut (all tiers).
4. **Statistics.**
   - Fix: 10k permutations everywhere; empirical p; effect sizes with CIs; drop z or label it secondary; one primary statistic (mean D); derangement nulls; average-rank Spearman; interaction tests instead of split-half significance; TOST against the fixed SESOI for every "no effect" claim; post hoc labels kept.
   - Done when: no table shows z alone, and every "absent" claim has an equivalence bound.
   - Cost: CPU, 1-2 days. Tiers: E for all.
5. **Confounds.**
   - 5a. Instruct vs base: remove all Llama-vs-Gemma statements (E for all). Run E2 on Llama-3.2-3B base (download about 3.5 h at your speed; same Meta licence, verify access): o for T and F, E for Q.
   - 5b. Hand-written, template-tuned facts: held-out template selection for new data (E for T, F, Q); automatic label check against Wikidata es/hi labels, plus LanguageTool on the Spanish templates (E for T, F, Q; o for W). Spanish is unaudited too: LOG notes "Los Museos Vaticanos se encuentra".
   - 5c. Hindi span length: Hindi is exploratory only in every branch (E). The E2 Hindi run is o. BLOOM is o (Q only).
   - 5d. No Hindi audit: run the automatic Wikidata label check (E for T, F). A fluent, not necessarily native, speaker can check the ~21 selected Hindi templates in 30 minutes (E for Q, o otherwise). Without either, Hindi claims stay descriptive.
6. **Edit results.**
   - Fix: cut to an appendix for T and F, or cut entirely. If redone (Q): n ≥ 50 edits per condition, binary success on *all* edits, unconditional transfer rates with Wilson CIs, mean |ΔP| over ≥ 100 controls including 20 same-relation neighbours, English-paraphrase success as the within-language ceiling.
   - Cost: 1 night for a redo. Tiers: cut/appendix for W, F, T; E for Q if edits stay in scope.
7. **Presentation.**
   - Fix: cut Fig. 4, or show it as a strip plot with counts; define ⌊L/2⌋; move the A/H/F codes to one appendix table; cite Dubey and Merity in the text; related work per Section 6; extend reproduce.sh to the v2 pipeline plus one clean-clone run; refresh the README.
   - Done when: a reader who never opens HYPOTHESIS.md can follow the paper.
   - Cost: 2-3 days. Tiers: E for all.
8. **No external timestamp for the pre-registration.**
   - Fix today: push, tag, and make a Zenodo GitHub-release DOI or a Software Heritage snapshot (both free). Register A12a, A13a and A13 on OSF (free) **before** running them. Say plainly in the paper that A1-A12 have git-only timestamps.
   - Cost: 1 hour. Tiers: E for F, T, Q; o for W.
9. **New from this audit:** shared noise seed (E0), identical prefixes (E0-CPU), magnitude after centring (A12a), A12 S4 coding (A12a), the Spearman tie bug, and H8 (cut, or reframe as "prefix-identical by construction"). All are E for every tier, because each one alone can sink a review.

---

#### 5. Decision tree and 4-week plan

##### Next 48 hours (CPU only)
1. Push the repo, tag it, and make a Zenodo or Software Heritage snapshot.
2. Write A12a (Section 2.10) and A13a (E0-GPU). Commit and register both on OSF.
3. Write `confound_overlap.py`. Run A12 as pre-registered, then A12a and E0-CPU.

##### Week 1 (Oct 24-30)
- **D1-D2:** the 48-hour items above. **Gate G0:**
  - If Llama's identical-prefix facts carry D > 2× the D of the others within relation, those facts are declared uninformative by construction and excluded from every cross-language analysis.
  - If Gemma's non-event_year S1 CI includes 0, Gemma's alignment is attributed to event_year (answer identity), and "Gemma clean" is withdrawn.
- **D3:** build the reduced tracer and its regression test. **Night 1:** E0-GPU, both models.
- **D4:** run the G1 analysis. **Gate G1:**
  - Survives → keep the correlational part as exploratory "fact-specific profile similarity beyond shared noise, prefix and magnitude".
  - Artefact → withdraw all alignment z-claims. Part I becomes the cautionary section.
  - Unresolved → report it as such and build nothing on it.
  - E2 runs whatever G1 says.
- **D4-D5:** write A13 (E2) in full: sets, templates, pairs, conditions, band rule, thresholds, fallback, n. Register it on OSF. Code `interchange.py` and the unit tests from Section 3.4. Count token-disjoint sources.
- **Night 2:** EN+ES filter with prefix templates for both models; start the Llama base download in the background.
- **Night 3:** 10-pair debugging pilot (excluded), then E2 on Llama-Instruct.

##### Week 2 (Oct 31-Nov 6)
- **Night 4:** E2 on Gemma. **Night 5:** E2 on Llama base (optional) and Hindi exploratory.
- **D8-D9:** analysis, then **Gate G2:**
  - The P1 ceiling fails twice → branch **C**.
  - P2 → branch **S**. P3 → branch **L**. P4 → branch **P**.
  - If the models disagree, report each separately, title the paper on the weaker common claim, and make no pooled claim.
- **D10:** freeze the claims, write the outline, enter manifest entries.

##### Week 3 (Nov 7-13)
- If token-disjoint sources were below 40: Wikidata extension (registered as A13b before running), then rerun E2 on the new pairs (one night).
- Site-B (final-position) analysis and the P6 validity correlation (CPU).
- Branch S or P only: optional language-offset add-on (one night).
- Write Methods and Results.
- **Gate G3:** if any T-essential fix from Section 4 is missing at the end of the week, target a workshop instead of TMLR.

##### Week 4 (Nov 14-20)
- Related work, limitations, deviations table, appendix.
- reproduce.sh v2 and a clean-clone run (one night).
- Red-team the draft with your review prompt.
- Submit to TMLR, or to the next ARR cycle or a workshop. I am not sure of the 2026-27 deadlines: check aclrollingreview.org and the workshop pages. The EMNLP 2026 workshop deadlines have probably passed.

##### Branch → paper → venue

| Branch | Condition | What the paper is | Venue and type |
|---|---|---|---|
| S + G1 artefact | E2 shared; old alignment was an artefact | "Interchange, not overlap". Strongest methods story: profile similarity misleads, interchange shows sharing. | TMLR regular; Findings as backup |
| S + G1 survives | E2 shared; old alignment real | Shared mid-layer subject representations; profile similarity is a weak but non-zero proxy (P6 quantifies it). | TMLR; Findings |
| P | Partial or layer-dependent | Layer-resolved account of where EN-ES factual representations become interchangeable. | TMLR; Findings |
| L | Language-specific | "Similar profiles, separate representations". Cautionary negative result with heavy method checks. | TMLR; workshop as backup. Findings is risky because it runs against the literature. |
| C | E2 not interpretable or not finished | "Pitfalls in comparing causal-tracing profiles across languages": E0, A12a, checklist. | Workshop (archival or not); TMLR only if E0 is clear-cut |

##### What Q1 would take (not achievable in 4 weeks under your constraints)
- E2 on at least 4 models: 2 families × base and instruct, plus a third family.
- At least 5 languages across at least 3 scripts.
- At least 300 Wikidata-sourced facts per language pair, with native-speaker audit.
- A theory-level contribution, for example a decomposition into shared content plus a language offset, validated by intervention.
- Realistically 3-6 months.
- If "one laptop" could ever include free Kaggle or Colab notebooks ($0), a 7-8B model becomes feasible. I do not assume this.

---

#### 6. Related work to cite and position against

Confidence: **[sure]**, **[fairly sure: verify ID]**, **[not sure: verify title and ID]**. Keep your 20 already-verified references. Wendler, Fierro, Lu et al. and Qi et al. become central.

##### Must engage (closest)
1. Dumas, Wendler, Veselovsky, Monea, West, "Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers", arXiv:2411.08745 **[fairly sure]**.
   - What they do: patch final-token latents between translation prompts; concept and output language separate.
   - What we add: factual recall, the subject site, same-language ceilings, token-disjoint controls, Hindi.
2. Lu, Zhang, Eickhoff, Pavlick, "Paths Not Taken", arXiv:2505.20546 **[already cited; check details]**.
   - What they do: the multilingual recall pipeline of the same Llama-3.2-3B, English-centric.
   - What we add: a direct interchange test of subject-level sharing, plus Hindi.
3. Fierro, Foroutan, Elliott, Søgaard, arXiv:2410.14387 **[already cited]**.
   - What they do: claim enrichment is language-independent and extraction language-dependent.
   - What we add: E2 tests enrichment-site interchangeability directly on Llama-3.2 and Gemma-2.
4. Hanna, Pezzelle, Belinkov, "Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding and Comparing Circuits", arXiv:2403.17806 **[sure]**.
   - What they do: argue that overlap is not functional equivalence.
   - What we add: a concrete case, with P6 as a direct test.
5. Ferrando & Costa-jussà, "On the Similarity of Circuits across Languages: a Case Study on the Subject-verb Agreement Task", arXiv:2410.06496 **[fairly sure]**.
   - What they do: compare EN vs ES circuits in Gemma-2B, for agreement.
   - What we add: factual recall, and Hindi.
6. Zhang, Yu, Zang, Eickhoff, Pavlick, "The Same But Different: Structural Similarities and Differences in Multilingual Language Modeling", arXiv:2410.09223 **[fairly sure]**.
   - What they do: shared and language-specific circuit components, English vs Chinese.
7. Brinkmann, Wendler, Bartelt, Mueller, "Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages", arXiv:2501.06346 **[fairly sure]**.
   - What they do: shared SAE features with causal tests.
8. Wu, Yu, Yogatama, Lu, Kim, "The Semantic Hub Hypothesis", arXiv:2411.04986 **[fairly sure]**.
   - What they do: a shared representation space with interventions.
9. Lindsey et al. 2025, "On the Biology of a Large Language Model", Transformer Circuits Thread, not on arXiv **[sure it exists]**.
   - What they do: multilingual features are shared, input and output are language-specific, and sharing grows with scale.
10. Ifergan, Choshen, Aharoni, Szpektor, Abend, "Beneath the Surface of Consistency: Exploring Cross-lingual Knowledge Representation Sharing in LLMs", arXiv:2408.10646 **[not sure of ID]**.
    - What they do: measure representation sharing via editing; script similarity matters.
11. Qi, Fernández, Bisazza, arXiv:2310.10378 **[already cited]**.
    - Why it matters: cross-lingual consistency relates to vocabulary overlap. This is the behavioural counterpart of your lexical finding.

##### Method (patching, interchange, corruption)
12. Zhang & Nanda, "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods", arXiv:2309.16042 **[sure]**. Gaussian noise vs symmetric token replacement; justifies E0 and the swap design.
13. Heimersheim & Nanda, "How to use and interpret activation patching", arXiv:2404.15255 **[sure]**.
14. Geiger, Lu, Icard, Potts, "Causal Abstractions of Neural Networks", arXiv:2106.02997 **[fairly sure]**. Interchange interventions.
15. Geiger, Wu, Potts, Icard, Goodman, DAS, arXiv:2303.02536 **[fairly sure]**.
16. Huang, Wu, Potts, Geva, Geiger, "RAVEL", arXiv:2402.17700 **[sure]**. Entity-attribute interchange, monolingual.
17. Makelov, Lange, Nanda, "Is This the Subspace You Are Looking For?", arXiv:2311.17030 **[fairly sure]**. Cite only if you use subspace patching.
18. Hernandez et al., "Linearity of Relation Decoding in Transformer Language Models", arXiv:2308.09124 **[sure]**.
19. Ghandeharioun et al., "Patchscopes", arXiv:2401.06102 **[sure]**.
20. Chughtai, Cooney, Nanda, "Summing Up the Facts", arXiv:2402.07321 **[fairly sure]**.
21. Prakash et al., "Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking", arXiv:2402.14811 **[fairly sure]**. Base vs instruct prior.
22. Merullo, Eickhoff, Pavlick, "Circuit Component Reuse Across Tasks in Transformer Language Models", arXiv:2310.08744 **[fairly sure]**.

##### Multilinguality, lexical overlap, tokenisation
23. Pires, Schlinger, Garrette, "How Multilingual is Multilingual BERT?", arXiv:1906.01502 **[sure]**. Transfer without lexical overlap.
24. K, Wang, Mayhew, Roth, "Cross-Lingual Ability of Multilingual BERT: An Empirical Study", arXiv:1912.07840 **[fairly sure]**.
25. Conneau, Wu, Li, Zettlemoyer, Stoyanov, "Emerging Cross-lingual Structure in Pretrained Language Models", arXiv:1911.01464 **[fairly sure]**.
26. Libovický, Rosa, Fraser, "On the Language Neutrality of Pre-trained Multilingual Representations", arXiv:2004.05160 **[fairly sure]**. Language-mean centring; precedent for your C1.
27. Chang, Tu, Bergen, "The Geometry of Multilingual Language Model Representations", arXiv:2205.10964 **[fairly sure]**.
28. Tang et al., "Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models", arXiv:2402.16438 **[sure]**.
29. Kojima et al., "On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons", arXiv:2404.02431 **[fairly sure]**.
30. Zhao et al., "How do Large Language Models Handle Multilingualism?", arXiv:2402.18815 **[fairly sure]**.
31. Gao et al., "Multilingual Pretraining and Instruction Tuning Improve Cross-Lingual Knowledge Alignment, But Only Shallowly", arXiv:2404.04659 **[not sure of ID]**.
32. Feucht, Atkinson, Wallace, Bau, "Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs", arXiv:2406.20086 **[fairly sure]**. Last token of multi-token names; relevant to Hindi spans.
33. Kaplan, Oren, Reif, Schwartz, "From Tokens to Words: On the Inner Lexicon of LLMs", arXiv:2410.05864 **[not sure of ID]**.
34. Petrov, La Malfa, Torr, Bibi, "Language Model Tokenizers Introduce Unfairness Between Languages", arXiv:2305.15425 **[sure]**.
35. Hämmerl et al., "Understanding Cross-Lingual Alignment: A Survey", arXiv:2404.06228 **[not sure of ID]**.

##### Statistics (journal papers, no arXiv)
36. Phipson & Smyth (2010), "Permutation P-values Should Never Be Zero", *Stat. Appl. Genet. Mol. Biol.* **[sure]**.
37. Gelman & Stern (2006), "The Difference Between 'Significant' and 'Not Significant' is not Itself Statistically Significant", *The American Statistician* **[sure]**.
38. Lakens (2017), "Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses", *Social Psychological and Personality Science* **[sure]**.

Background, optional: Geva et al. 2021 (arXiv:2012.14913), Dai et al. 2022 (arXiv:2104.08696).

Positioning sentence, after E2:
> Prior work patches concept latents between translation prompts (Dumas et al.) or traces multilingual recall (Fierro et al.; Lu et al.). We test whether factual *subject* representations are interchangeable across languages, normalised by a same-language ceiling and controlled for token overlap. We also show that tracing-profile similarity, the measure used in our own earlier analysis, [is / is not] a valid proxy for it.

---

#### 7. What to cut; titles and abstracts per branch

**Cut in every branch:**
- The in-context baseline.
- The edit sweep, H7, F7 and the damage panel: appendix "pilot editing results" at most.
- Critical-layer claims and Fig. 4.
- H8 as "robustness".
- Pooled-null tables.
- The 33-fact cross-model comparison and any Llama-vs-Gemma contrast.
- "p = 0.001" and z-only tables.
- BLOOM and Llama-1B "pending" notes: one future-work line.
- Amendment codes in the main text.
- The v1 pilot discussion: one sentence and an appendix.

**Keep:** E0 + A12/A12a as a short "Why profile similarity is not enough" section, E2 as the main result, P6, and Hindi as exploratory.

Abstracts use [brackets] for numbers not yet known. Fill them only from manifest entries.

##### Branch S (shared)
**Title:** Interchangeable Across Languages: Testing Factual Subject Representations in Two Small Language Models

> Is a fact's subject represented the same way when a language model reads it in English and in Spanish? Comparing causal-tracing profiles across languages cannot settle this: in Llama-3.2-3B-Instruct and Gemma-2-2B, same-fact profile similarity [S-a: is explained by shared corruption noise and by identical subject or answer strings] [S-b: co-varies with identical subject and answer strings]. We therefore test interchangeability directly. We replace the subject representation in a Spanish factual prompt with that of a different subject computed in English, and measure how far the answer moves toward the English subject's answer, relative to a same-language swap. For [n] pairs whose English and Spanish names share no subword token, the cross-lingual swap reaches [I] (95% CI [a, b]) of the same-language effect in layers [x-y] of Llama and [I′] in Gemma; the reverse direction gives [..]. Hindi results are exploratory ([..]). All analyses were pre-registered; code, raw runs and a verified claim manifest are released.

##### Branch L (language-specific)
**Title:** Similar Profiles, Separate Representations: Cross-Lingual Subject Interchange in Two Small Language Models

> Do small language models share factual subject representations across languages? English and Spanish causal-tracing profiles of the same fact are more alike than those of different facts in Llama-3.2-3B-Instruct and Gemma-2-2B [, but much of this similarity is explained by shared corruption noise and identical strings]. We test interchangeability directly by swapping a subject representation computed in English into a Spanish factual prompt and comparing the answer change with a same-language swap that succeeds in [p]% of pairs. For [n] token-disjoint pairs, the cross-lingual swap reaches only [I] of the same-language effect (90% CI upper bound [u]); the reverse direction and a final-token readout agree [..]. At the last subject token, subject representations in these models are therefore not interchangeable across languages. We discuss why this differs from concept-level patching results and release pre-registered analyses, code and a verified claim manifest.

##### Branch P (partial or layer-dependent)
**Title:** Where Do Factual Representations Become Shared Across Languages? A Layer-Resolved Interchange Study

> We ask at which depth a fact's subject representation becomes interchangeable between English and Spanish in Llama-3.2-3B-Instruct and Gemma-2-2B. Comparing causal-tracing profiles is not a reliable answer: same-fact profile similarity [co-varies with / is largely explained by] shared corruption noise and identical subject or answer strings. Instead, we swap the subject representation of a Spanish prompt with that of a different subject computed in English and normalise the answer change by a same-language swap. For [n] pairs whose names share no subword token, interchangeability rises from [a] in layers [..] to [b] in layers [..] and falls after [..], reaching [I] overall (95% CI [..]); it is [lower] for Hindi ([..], exploratory). Sharing is therefore partial and depth-dependent rather than all-or-none. All analyses were pre-registered; code, raw runs and a verified claim manifest are released.

##### Branch C (cautionary only)
**Title:** Pitfalls in Comparing Causal-Tracing Profiles Across Languages

> Cross-lingual comparisons of causal-tracing profiles are used to argue that facts are stored in shared or language-specific circuits. We audit such a comparison in Llama-3.2-3B-Instruct and Gemma-2-2B for English, Spanish and Hindi (126 and 59 facts). Same-fact English-Spanish profiles are more similar than different-fact profiles (mean cosine increment 0.090 and 0.182). [X]% of this increment disappears when each language receives independent corruption noise, when prompts with token-identical prefixes are excluded, and when profiles are normalised before centring; the remainder concentrates in facts with identical subject names (Llama) or identical answers (Gemma). Hindi alignment depends on subject span length in a way the same artefacts predict. We give a checklist for cross-lingual causal comparisons. All analyses were pre-registered; code, raw runs and a verified claim manifest are released.

All four are under 180 words. Branch C is defensible only after E0 gives [X]. If E0 says "survives", rewrite its third and fourth sentences.

---

#### 8. What a strong reviewer would raise that neither review did

1. Shared corruption noise across languages and phrasings (2.1).
2. Subject-first templates make the measured site context-free, so "cross-language" means "cross-name". Identical names give identical computations (2.1, 3.1).
3. One noise draw per fact. There is no reliability estimate, no noise ceiling, and the argmax "critical layer" from a single draw is a high-variance estimator. The ROME reference code averages several noise samples per prompt (I believe the default is 10; verify).
4. Centring unnormalised profiles turns magnitude correlation into alignment (2.2).
5. The Spearman tie-breaking bug (2.5).
6. A12 S4's intercept is an extrapolation, and its SESOI is data-dependent (2.10).
7. Requiring correctness in all three languages biases the EN-ES sample toward famous, name-sharing entities. Do not condition EN-ES questions on Hindi.
8. Novelty: Lu et al. (2505.20546) study multilingual recall in the *same* Llama-3.2-3B, and Dumas et al. already do cross-lingual patching. "What is new?" needs a one-sentence answer (Section 6).
9. Gaussian-noise corruption and early-layer peaks; no symmetric-token-replacement check (Zhang & Nanda).
10. Digit tokenisation differs by model (Gemma splits digits singly; Llama-3 groups up to three, I am fairly sure). This may explain why event_year aligns in Gemma but not in Llama. Speculative; check.
11. Corruption strength differs by language: per-language sigma, and 8 corrupted Hindi tokens vs 2 English ones. Report P_corr/P_clean by language. The collapse criterion (< 0.5 × clean) is weak.
12. The TransformerLens-vs-Hugging Face logit check is missing for Gemma-2 (soft-caps, post-norms).
13. The Instruct model is prompted without its chat template. State it. It may also explain the in-context anomaly.
14. Facts and their reverses (France→Paris, Paris→France) are both in the set. The entity dependence crosses relations, but the stratified null treats them as independent.
15. Spanish is unaudited too (grammar slips already logged), not only Hindi.
16. Forking paths: about 100 z-values on the same 126 and 59 facts. Only new data (E2) can be confirmatory. Say this in the paper.
17. No per-fact plots and no raw example traces. Reviewers trust a figure of D_i per fact, coloured by relation and by prefix class, more than a z.

---

#### Appendix A. E0-CPU sketch (untested: I had no grids; register it in A12a before running)

```python
# Run from code/. Classify each fact by what EN and ES share up to the measured site; normalised-centred increment D per class.
import numpy as np
from pathlib import Path
from controls_overlap import SETS
from circuit_overlap import load, is_valid, cos_matrix

for m, (r1, r2, _, _) in SETS.items():
    arr, recs, toks = {}, {}, {}
    for r in (r1, r2):
        A, R = load(Path(r)); arr.update(A); recs.update(R)
        for fid, lg in R:
            z = np.load(Path(r) / "grids" / f"{fid}_{lg}.npz")
            toks[(fid, lg)] = (z["tokens"].tolist(), z["span"].tolist())
    ids = sorted({f for f, _ in recs if all((f, l) in recs and is_valid(recs[(f, l)]) for l in ("en", "es", "hi"))})
    rel = np.array([recs[(i, "en")]["relation"] for i in ids])

    def cls(i):
        (te, se), (ts, ss) = toks[(i, "en")], toks[(i, "es")]
        if te[:se[-1] + 1] == ts[:ss[-1] + 1]:
            return "identical_prefix"
        return "equal_span" if len(se) == len(ss) else "other"
    c = np.array([cls(i) for i in ids])
    E = np.stack([arr[(i, "en")]["mlp"] for i in ids]); S = np.stack([arr[(i, "es")]["mlp"] for i in ids])
    E /= np.linalg.norm(E, axis=1, keepdims=True); S /= np.linalg.norm(S, axis=1, keepdims=True)  # normalise BEFORE centring
    M = cos_matrix(E - E.mean(0), S - S.mean(0))
    same = (rel[:, None] == rel[None, :]) & ~np.eye(len(ids), dtype=bool)
    D = np.array([M[k, k] - M[k, same[k]].mean() if same[k].any() else np.nan for k in range(len(ids))])
    for k in ("identical_prefix", "equal_span", "other"):
        mk = c == k
        print(m, k, int(mk.sum()), round(float(np.nanmean(D[mk])), 3), dict(zip(*np.unique(rel[mk], return_counts=True))))
```

Compare classes **within relation**: identical prefixes occur only in subject-first relations. Add the cluster bootstrap and the derangement null before anything enters the manifest.


## 10.7 Draft of the control analysis (A12a)

*Copied word for word from `docs/A12a_DRAFT.md` when this record was built. To change it, edit that file and rebuild.*

- **A12a (2026-10-23, after an external review of A12 and before any A12 or A12a statistic is computed; CPU only, cached traces, no GPU). A12 is not edited and is run exactly as written; A12a adds analyses and corrects how two of them are read.**
  - **Why.** Code inspection and numbers already in the manifest show four problems in A12 as written. None of them comes from an A12 result.
    - (1) The corruption noise is shared across languages. `causal_trace_v2.py` draws the subject noise from the per-fact seed (sha256 of the fact id) for every language and every phrasing. Subject token k therefore gets the same noise row in EN, ES and HI. With subject-first templates and identical names, the EN and ES token prefixes up to the last subject token are identical, so the measured site is computed identically in both languages, apart from sigma.
    - (2) A12 centres unnormalised profiles. A centred profile then points along the mean profile when the fact's total effect is above average. A correlation of total effects across languages therefore shows up as alignment.
    - (3) The S4 intercept is evaluated at the reference relation, at log total effect = 0 and at name similarity = 0. This is an extrapolation, and it makes the A12 absence rule unusable.
    - (4) The A12 smallest effect of interest is defined from Gemma's own D.
    - Also: the A11 Spearman broke ties by fact order (46 Llama similarities equal 1.0), and the A11 and A12 permutations allow fixed points.
  - **Seen before writing this.** The A11 results, the per-relation z-values in docs/LOG.md, the code, and the external review's predictions (Q1-Q5 below). No A12 or A12a statistic and no prefix-class count has been computed.
  - **Sets.** As A12: Llama 126 and Gemma 59 valid facts (trace runs as in `code/controls_overlap.py` SETS), phrasing 0. Relations with < 10 valid facts are merged into "other" (as in A12). These merged labels are the strata for centring, for the different-fact baselines and for the permutations.
  - **Normalised-centred (NC) profiles.**
    - p = the MLP-output indirect effect per layer at the last subject token (as in A3/A5).
    - u = p / ||p||. The NC profile is u minus the mean u of the fact's stratum, per language.
    - Secondary variant: u − (u·m̂)m̂, where m̂ is the unit stratum mean.
    - Facts with ||p|| < 1e-8 are excluded and listed.
  - **Prefix classes (EN vs ES, from the token ids stored in the cached grids).**
    - IP: identical token ids from BOS up to and including the last subject token.
    - EQ: not IP, and the two subject spans have the same number of tokens, so the last subject token gets the same noise row.
    - OT: all other facts.
    - **Rule fixed now, whatever the counts:** IP facts are uninformative about cross-language sharing by construction. They are excluded from every cross-language claim from now on and reported only descriptively.
  - **Token-level covariates.** Computed with each model's own tokenizer files; no weights are loaded.
    - J: Jaccard of the sets of EN and ES subject token pieces (decoded, whitespace stripped, lower-cased, NFKD accents removed, empty pieces dropped).
    - LT: last subject token id identical.
    - NR: equal span length, i.e. a shared last-token noise row.
    - SD: |EN span − ES span| in tokens.
    - AI: answer strings identical after `matching.norm`.
    - TE_en, TE_es: log total effect (score_clean − score_corr) in each language, centred at the model mean.
    - The A12 difflib similarity is kept as a secondary covariate.
  - **S1a (NC increment).**
    - D_i = s_ii − mean over same-stratum j ≠ i of s_ij, on NC profiles (the A12 S1 definition).
    - Report the mean D with a 95% entity-cluster bootstrap CI (A11 clusters, 2000 resamples).
    - Empirical p from 10,000 within-stratum derangements of the ES profiles (no fixed points): p = (1 + #{null ≥ obs}) / (B + 1).
    - Sets: (i) **informative = not IP and not event_year (primary)**; (ii) all valid facts; (iii) event_year only; (iv) each prefix class, overall and within each stratum that has ≥ 5 facts in that class.
  - **S1b (magnitude diagnostic).**
    - Spearman (average ranks) of TE_en vs TE_es within stratum: ranks demeaned per stratum, with a stratified permutation p.
    - Mean D on unnormalised relation-centred profiles (A12 S1), reported next to S1a on the same sets.
  - **S4a (covariate model).** It replaces the coding of A12 S4; A12 S4 is still reported as written.
    - On informative facts, D_i (NC) is regressed by OLS on J, LT, AI, NR, SD, TE_en, TE_es and stratum (sum-to-zero coding), with entity-cluster robust SEs (CR1).
    - Target quantity: the predicted D at J = 0, LT = 0, AI = 0, NR = 0, SD = 1, TE at the mean, strata averaged with equal weight. This is "name-free, no shared noise row, non-copyable" alignment.
    - Report it with a 95% CI from the robust SE, plus a cluster-bootstrap CI as robustness.
    - Report how many facts have J = 0 and how many have NR = 0. If fewer than 10 facts have J = 0, the target is labelled an extrapolation.
  - **Corrections of A11 (labelled as such).**
    - C2 Spearman is recomputed with average ranks and a within-stratum permutation, on raw cosine and on NC D.
    - C1 is recomputed with derangements.
    - Both the old and the corrected versions are reported.
  - **Fixed smallest effect of interest.** δ = 0.05 cosine units of NC D, for both models. It is about half the smallest A11 same-minus-different increment (0.090, Llama) and was chosen without any A12 or A12a number. A12's own SESOI is also reported.
  - **Decision rules.** Per model, on the two primary quantities (S1a on informative facts and the S4a target), using 97.5% CIs (Bonferroni over 2):
    - Present: lower bound > 0.
    - Absent: upper bound < δ.
    - Unresolved: anything else.
    - A12a cannot remove shared-noise effects from EQ and OT facts. A "present" verdict is therefore provisional until the independent-seed re-trace (A13a), and it is worded "not explained by prefix identity, profile magnitude or token overlap".
  - **Predictions (directions, from the review, stated before computing).**
    - Q1: within the same stratum, mean D is higher for IP than for OT facts in Llama.
    - Q2: on all valid facts, Gemma's D on NC profiles is smaller than its D on unnormalised centred profiles (A12 S1).
    - Q3: Gemma S1a on informative facts has a CI that includes 0.
    - Q4: the Llama S4a target has a CI that includes 0.
    - Q5: TE_en and TE_es are positively correlated within stratum in both models.
    - Results against these predictions are reported as such.
  - **Implementation.** `code/confound_overlap.py` with the flag `--a12a`. Seed 20261023 for the bootstrap and the permutations. Results go to a new write-once run directory, and every reported number enters the manifest. Results are reported whatever they are, and this amendment is not edited afterwards.
  - **Not covered.** Independent noise seeds per language with averaged noise draws (A13a) and cross-language interchange patching (A13) need GPU runs. Each gets its own amendment before any data exist.
