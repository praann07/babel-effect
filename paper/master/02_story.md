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
