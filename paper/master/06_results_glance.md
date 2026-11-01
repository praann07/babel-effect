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
