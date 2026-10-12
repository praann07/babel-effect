# HYPOTHESIS (pre-registration)

Written and committed before any causal tracing or circuit-overlap computation. At the time of writing, `code/causal_trace.py` and `code/circuit_overlap.py` do not exist and no attribution has been computed on any fact. Only P0 (smoke test) and P1 (fact filtering by greedy accuracy) have been run. Later edits go in the "Amendments" section at the bottom, with a date and a reason. Nothing above that line is edited after the P3 computation.

## Setup (fixed)
- Model: Llama-3.2-3B-Instruct (28 layers, d=3072), bf16, TransformerLens 3.9.0, `from_pretrained_no_processing`.
- Facts: `data/facts.json`, n=52 (26 capital_country, 22 capital_of, 4 landmark_city), correct in EN, ES and HI under greedy decoding.
- Tracing: Gaussian noise of 3x the subject-embedding std, per-fact seed; attribution = logit(object first token) minus logit(top competitor), patched minus corrupted; residual stream and MLP-output sites.
- Critical site per language = argmax attribution at that language's last subject token.
- Primary overlap statistic: per-layer attribution profile at each language's own critical token, compared by cosine and Spearman. Secondary: span-pooled top-k grid IoU, k in {5, 10, 20}.
- Null: shuffle fact IDs across languages (1000 permutations) so the EN profile of fact i is paired with the ES or HI profile of a different fact; z-score and one-sided p from the permutation distribution. Bootstrap 95% CIs over facts.
- Analysis set: all n=52 facts. Robustness set: the 48 facts excluding landmark_city (n=4 is too small to analyse alone).

## Predictions (directions, stated before computing)
Priors come from the causal-tracing literature (Meng et al. 2022, arXiv:2202.05262, for English GPT-style models), not from any measurement here.

- **H1 (shared locus).** Per-layer profile cosine and Spearman for EN-ES and EN-HI are above the permutation null (z > 2, p < 0.05, bootstrap CI above the null mean). Reason: factual recall in English models concentrates in mid-layer MLPs at the last subject token, and this layer-wise structure is likely shared.
- **H2 (distance ordering).** EN-ES overlap > EN-HI overlap, on both primary and secondary statistics. Reason: Spanish shares script and subword vocabulary with English; Hindi is a different script, uses multi-token subjects, and has lower accuracy (0.18-0.58 vs 0.65-0.78).
- **H3 (peak location).** The English critical layer lies in the middle third of the network (layers 8-19 of 0-27). Spanish peak within 2 layers of English (median over facts). Hindi peak the same or later than English, with lower peak magnitude.
- **H4 (editing).** Rank-1 edit at the EN traced layer succeeds in >= 80% of EN test facts. Transfer ratio satisfies 0 < HI < ES < 1 (median over facts). Edit at the random-layer control transfers less than the traced and mid-layer edits.
- **H5 (side effects).** Specificity (max |dP| over 20 control facts) below 0.05 and damage (perplexity ratio on 100 Wikipedia sentences) below 1.05 for the traced-layer edit.

## Interpretation rules (fixed in advance)
- Language-agnostic account supported: H1 holds for both pairs and transfer ratio for both languages has a bootstrap CI above 0.5.
- Language-specific account supported: overlap for both pairs is not distinguishable from the null (CI includes the null mean) and transfer ratio CI includes 0.
- Partial/mixed: anything else. In particular, a shared layer-wise profile with low edit transfer is reported as "shared locus, language-specific content", not as agreement.
- A result against these predictions is reported as such. Thresholds above are not changed after seeing data.

## Known limitations (declared now)
- n=52 and a skewed relation mix; CIs will be wide.
- Selection bias: kept facts are those the model already answers correctly in all three languages.
- The exact-match filter rejects some spelling variants (e.g. "Hanoi" vs "Hanói").
- The model weights come from an unsloth mirror, not byte-verified against Meta's repo.
- One model only; claims are limited to Llama-3.2-3B-Instruct.

## Amendments
(none)
