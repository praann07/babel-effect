# The Babel Effect: Cross-Lingual Overlap and Transfer of Localised Factual Edits in Llama-3.2-3B-Instruct

*Draft. All numbers are read from `results/manifest.json` (240 claims, each mapped to an artifact path and key; `./reproduce.sh verify` re-checks them). Hypotheses were pre-registered in `docs/HYPOTHESIS.md` before the overlap analysis; four dated amendments (A1-A4) are listed in Section 6.*

## Abstract

We ask whether a language model stores a fact in language-specific circuits or in a shared, language-agnostic space. On Llama-3.2-3B-Instruct we trace the causal effect of restoring clean activations in English (EN), Spanish (ES) and Hindi (HI) for 66 capital-city facts that the model answers correctly in all three languages. Against a shuffled-fact permutation null, the per-layer MLP attribution profile at the last subject token is fact-specifically aligned between EN and ES (median cosine 0.748, 95% CI [0.676, 0.801], null mean 0.570, z = 6.06) but not between EN and HI (0.323, z = 0.39) or ES and HI (0.227, z = 0.38). We then apply a rank-1 ROME-style MLP edit in English only. An edit at the layer where English tracing peaks (layers 0-5) succeeds in 9 of 10 cases but transfers weakly (median transfer ratio 0.147 to ES, 0.002 to HI). An edit at layer 14 transfers more evenly (0.203 ES, 0.244 HI) but breaks unrelated facts (median specificity loss 0.351). Intervals are wide (n = 10 edit facts), the sample is one model and one relation family, and Hindi results are limited by weak signal. The evidence supports a partly shared English-Spanish locus and does not support a shared Hindi locus at this sample size.

## 1. Introduction

Multilingual language models answer factual questions in many languages. Whether the underlying fact is stored once and read out through language-specific front and back ends, or stored separately per language, determines whether an edit made in one language should carry over to others. We test this with two measurements on one open-weight model: overlap of causal-tracing profiles across languages, and transfer of a localised edit from English to Spanish and Hindi.

Contributions: (i) a cross-lingual causal-tracing comparison with a fact-shuffling permutation null; (ii) a measurement of edit transfer, specificity and damage at three layer choices; (iii) a pre-registered design with logged deviations, all code, run directories and a verifiable claim manifest.

## 2. Related work

Causal tracing and rank-one model editing for factual associations follow Meng et al. (2022, arXiv:2202.05262). The model is from the Llama 3 family (Dubey et al., 2024, arXiv:2407.21783). Perplexity damage is measured on WikiText-2 (Merity et al., 2016, arXiv:1609.07843). Prior work on multilingual factual recall, cross-lingual knowledge editing, and language-neutral representations in multilingual models is relevant but is not cited here: [CITATION NEEDED] (to be added only after each reference is verified).

## 3. Method

**Model and tooling.** Llama-3.2-3B-Instruct (28 layers, d = 3072), bf16, TransformerLens 3.9.0 without weight processing; all runs on one 8 GB GPU (peak 7.58 GiB). Weight shards are byte-identical (SHA-256) to Meta's repository.

**Facts.** 289 candidate triples in three relations (capital_of, capital_country, landmark_city) were written with hand-made EN/ES/HI forms and templates. A fact is kept if the greedy continuation, ignoring leading spaces, begins with the exact object string in all three languages; Hindi has two template variants chosen once per relation by whole-set accuracy. 67 facts remain (33 capital_country, 32 capital_of, 2 landmark_city). English accuracy before filtering is 0.95-0.99; Spanish 0.59-0.72; Hindi (variant 0) 0.08-0.39.

**Causal tracing.** For each fact and language we add Gaussian noise (sigma = 3 x the pooled std of subject-token embeddings, per-fact seed) to the subject-token embeddings, then restore the clean activation at one (layer, token) site of the residual stream or MLP output, and measure the change in logit(object) - logit(competitor) relative to the corrupted run. The object is the model's clean top-1 first token; the competitor is the highest non-object token in the corrupted run. A pair is valid if P(object) falls below half its clean value and the total effect is positive; 66 of 67 facts are valid in all three languages. The critical layer is the argmax of the MLP-output effect at the last subject token.

**Overlap.** Per fact and language pair we compute the cosine and Spearman correlation of the 28-layer MLP-output profile at each language's own last subject token, and the top-k IoU (k = 5, 10, 20) on a 112-cell grid. The statistic is the median over facts. The null pairs fact i in one language with a randomly permuted fact in the other (1000 permutations); z and a one-sided p come from this null, and 95% CIs from 2000 bootstrap resamples of facts.

**Editing.** For 10 facts (4 capital_country, 4 capital_of, 2 landmark_city) we replace the object by the object of another fact of the same relation, using a rank-1 update of the MLP down-projection at the last subject token (keys from five English contexts, covariance from 30,000 WikiText-2 tokens, 25 Adam steps for the value vector). Three layer conditions per fact: the fact's own English critical layer ("traced"), layer 14 ("mid"), and a random other layer. Only English prompts are used to build the edit. Metrics: dP of the new object's first token per language; transfer ratio dP(L2)/dP(EN) over edits whose English edit succeeded; specificity = max |dP| of the original first-token probability over 20 unrelated English facts; damage = perplexity ratio on 100 WikiText-2 test sentences.

## 4. Results

### 4.1 Tracing

All 30 pairs in the 10-fact pilot and 66 of 67 facts in the full set collapse under corruption with a positive total effect. The English critical MLP layer has median 2 (all 66 facts), Spanish 2, Hindi 3; 3%, 6% and 5% of facts, respectively, fall in layers 8-19. The residual stream at the last subject token carries the effect up to about layer 20 before it moves to the last token (Figure 1, `analysis/figs_20261012_p2trace_v3/profiles_last_subject.png`). The Hindi effect at the subject token is about half the English/Spanish size.

### 4.2 Overlap against the null (n = 66)

| pair | statistic | observed median [95% CI] | null mean | z | p |
|---|---|---|---|---|---|
| EN-ES | MLP cosine | 0.748 [0.676, 0.801] | 0.570 | 6.06 | 0.001 |
| EN-ES | MLP Spearman | 0.473 [0.395, 0.627] | 0.309 | 4.84 | 0.001 |
| EN-HI | MLP cosine | 0.323 [0.106, 0.448] | 0.304 | 0.39 | 0.352 |
| EN-HI | MLP Spearman | 0.071 [-0.015, 0.154] | 0.098 | -0.75 | 0.769 |
| ES-HI | MLP cosine | 0.227 [-0.037, 0.349] | 0.204 | 0.38 | 0.365 |
| ES-HI | MLP Spearman | 0.079 [-0.029, 0.164] | 0.061 | 0.49 | 0.312 |

p-values are bounded below by 1/1001. Only the two EN-ES tests pass a Bonferroni correction over the six primary tests (alpha 0.0083). Residual-stream profiles are highly similar under the null for every pair (null means 0.82-0.94), so they carry little fact-specific information (EN-ES cosine 0.961 vs null 0.944, z = 2.69). Top-k IoU agrees in direction: EN-ES z = 2.73 (k = 10) and 2.11 (k = 20); EN-HI and ES-HI |z| < 1.3 at all k. Conclusions are unchanged on all 67 kept facts, without landmark_city, and within capital_country; within capital_of (n = 31) the EN-ES Spearman test is not significant (see `docs/LOG.md`). The median absolute difference between critical layers is 2 layers for every pair (61%, 58% and 53% within two layers for EN-ES, EN-HI, ES-HI).

### 4.3 Edit success, transfer, specificity, damage (10 facts per condition)

| condition | EN success | dP EN | dP ES | dP HI | transfer ES | transfer HI |
|---|---|---|---|---|---|---|
| traced | 9/10 | 0.949 | 0.107 | 0.004 | 0.147 [0.000, 0.769] | 0.002 [0.000, 0.267] |
| mid (L14) | 10/10 | 0.927 | 0.180 | 0.242 | 0.203 [0.007, 0.691] | 0.244 [0.010, 1.026] |
| random | 10/10 | 0.937 | 0.005 | 0.004 | 0.005 [0.000, 0.139] | 0.005 [0.000, 0.340] |

| condition | specificity (median max abs dP) [CI] | worst edit | perplexity ratio (median) | worst |
|---|---|---|---|---|
| traced | 0.045 [0.024, 0.400] | 0.981 | 1.0009 | 1.0082 |
| mid | 0.351 [0.049, 0.886] | 0.946 | 1.0021 | 1.0165 |
| random | 0.061 [0.025, 0.546] | 0.917 | 1.0000 | 1.0015 |

The traced-layer edit meets the pre-registered 80% English success criterion (90%). Its transfer is small and ordered ES > HI; the mid-layer edit transfers about equally to both languages (HI at least as large as ES, intervals overlapping fully) but its median specificity loss is 0.351. Random-layer edits transfer least. Perplexity damage is below 1.05 in every condition. Top-1 flips to the new object (ES/HI): traced 40%/30%, mid 40%/60%, random 10%/30%.

## 5. Discussion

The tracing result is consistent with a shared English-Spanish locus at the level of which layers carry each fact; it does not show a shared Hindi locus. Absence of Hindi alignment is not evidence of a separate Hindi circuit: Hindi subject effects are smaller, subjects span more tokens, the object metric is the first subword, and n = 66 gives wide intervals. The layer-14 edit moving the Hindi answer as much as the Spanish one is compatible with a more language-neutral representation at mid depth than at the early layers where English tracing peaks; this is a hypothesis suggested by one n = 10 measurement and was not tested directly. Under the pre-registered interpretation rules the outcome is the partial/mixed case.

## 6. Deviations from the pre-registration

- The English peak layer prediction (layers 8-19, H3) is not met: the peak is in layers 0-5.
- A1 (before tracing): critical site = MLP-output argmax; noise scale pooled per language; collapse rule.
- A2 (after a 10-fact pilot, before overlap): the competitor token is taken from the corrupted run, because a clean-runner-up competitor gave negative total effects in 5 of 30 pairs. The superseded run is kept on disk.
- A3 (before overlap): definition of all P3 statistics and sets.
- A4 (before editing): the editing design and metrics.
- P1 was re-run after the first filter passed 3 Hindi facts whose first generated token was a newline (n fell to 49); the stricter rule and 135 more candidate facts gave n = 67. The first-run outputs are kept.
- Layer-0 edits were not reproducible between two runs of the same code (English dP 0.418 vs 0.939 for one fact), so layer-0 numbers have extra uncertainty. 10 of 30 edits reached the delta-norm clamp (6 traced, 4 random, 0 mid).

## 7. Limitations

One model, one relation family (capitals and landmarks), three languages, 66 facts for tracing and 10 for editing; wide intervals. The kept facts are those the model already answers correctly in all three languages, which biases toward well-known facts. The object metric is the first subword, so for ES/HI "transfer" means the first subword of the new object moves. Specificity is poor in the worst case in all conditions. The exact-match filter rejects some spelling variants. No comparison to other editing methods or to other models.

## 8. Reproducibility

Code, configs, write-once run directories, logs (`docs/LOG.md`, `docs/EXPERIMENT_LOG.md`), seeds and the claim manifest are in the repository. `./reproduce.sh verify` re-checks the manifest in seconds; `./reproduce.sh full` re-runs P1-P5 into new run directories and compares headline numbers.

## References

- Meng, K., Bau, D., Andonian, A., Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. arXiv:2202.05262.
- Dubey, A. et al. (2024). The Llama 3 Herd of Models. arXiv:2407.21783.
- Merity, S., Xiong, C., Bradbury, J., Socher, R. (2016). Pointer Sentinel Mixture Models. arXiv:1609.07843.
- [CITATION NEEDED] multilingual factual recall / cross-lingual knowledge editing / language-neutral representations.
