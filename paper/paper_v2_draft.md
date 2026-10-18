# The Babel Effect: Cross-Lingual Causal Profiles of Factual Recall in Two Small Language Models

*Draft v2 (work in progress). Sections marked **[PENDING]** wait for runs that are still executing (layer-sweep edits, phrasing robustness, non-copyable relations, in-context baseline). Every number is read from `results/manifest.json` or from a logged run (see `docs/LOG.md`). Hypotheses were pre-registered in `docs/HYPOTHESIS.md` (amendments A1-A9).*

## Abstract

Do language models store a fact once, in a language-neutral form, or separately per language? We trace the causal effect of restoring clean activations when the subject of a factual prompt is corrupted, in English (EN), Spanish (ES) and Hindi (HI), for Llama-3.2-3B-Instruct (92 facts) and Gemma-2-2B (58 facts), and compare per-layer profiles with a shuffled-fact permutation null. English and Spanish profiles of the same fact are aligned beyond chance in both models (MLP-profile cosine z = 4.22 and 4.81, p = 0.001). Hindi overlap is weaker: with a pooled null, Llama's EN-HI and ES-HI pairs are above chance (z = 2.45, 2.47) and Gemma's EN-HI marginally so (z = 2.19), but with a null restricted to the same relation only EN-ES remains significant after correction in both models. The weakness depends partly on Hindi subject length: with short Hindi subject spans the EN-HI cosine is 0.60 (Llama) against 0.20 with long spans. The layer where the fact is looked up is model-dependent (median layer 3 in Llama, 6 in Gemma). **[PENDING: edit-transfer results over every layer.]**

## 1. Introduction

Multilingual language models answer factual questions in many languages, yet it is unclear whether a fact is stored once and read out through language-specific front and back ends, or stored separately per language. The answer determines whether an edit made in one language should carry over to others. We test it with two measurements on open-weight models that fit one 8 GB GPU: overlap of causal-tracing profiles across languages, and transfer of a localised edit from English to Spanish and Hindi.

Contributions: (i) a cross-lingual causal-tracing comparison with a shuffled-fact permutation null, a relation-stratified variant of it and a Hindi tokenisation diagnostic; (ii) results on two models with different lookup layers; (iii) an edit-transfer measurement over every layer **[PENDING]**; (iv) a pre-registered design with logged deviations, all code, write-once run directories and a verifiable claim manifest.

## 2. Related work

**Locating and editing facts.** Causal tracing and rank-one editing of MLP weights follow Meng et al. (2022, ROME, arXiv:2202.05262) and its multi-edit extension MEMIT (Meng et al., 2022, arXiv:2210.07229). Geva et al. (2023, arXiv:2304.14767) describe recall as subject enrichment, relation propagation and attribute extraction. Hase et al. (2023, arXiv:2301.04213) find that the layer where tracing peaks does not reliably predict the best layer to edit, which motivates sweeping edit layers rather than relying on the traced layer. Gupta et al. (2024, arXiv:2403.07175) and Yang et al. (2024, arXiv:2406.11263) trace model collapse after ROME edits to implementation details, which is relevant to how damage is measured.

**Multilingual recall mechanisms.** Wendler et al. (2024, arXiv:2402.10588) report that Llama-2 middle layers decode towards English before language-specific output. Fierro et al. (2024, arXiv:2410.14387) use patching on XGLM, EuroLLM and mT5 and find subject enrichment to be language-independent and object extraction language-dependent. Zhang et al. (2024, arXiv:2406.16416) locate language-agnostic factual neurons mostly in middle layers. Wang et al. (2025, arXiv:2504.04264) study cross-lingual factual inconsistency in LLaMA2-7B and BLOOM-560M. Lu et al. (2025, arXiv:2505.20546) analyse the multilingual factual-recall pipeline of Llama-3.2-3B, the model used here, in English, French, Spanish, Chinese, Japanese and Korean, without editing and without Hindi. Schut et al. (2025, arXiv:2502.15603) report English-like internal decisions in multilingual models. Multilingual factual probes are provided by X-FACTR (arXiv:2010.06189) and mLAMA (arXiv:2102.00894).

**Cross-lingual knowledge editing.** Wang et al. (2023, arXiv:2309.08952) and Khandelwal et al. (2024, arXiv:2407.10275) find that parameter-based editing transfers poorly across languages while in-context editing transfers much better; Nie et al. (2024, arXiv:2406.17764) report in-context editing across 53 languages and a penalty for non-Latin scripts. Qi et al. (2023, arXiv:2310.10378) edit BLOOM-3b with ROME in English and find that the edit propagates to languages whose factual knowledge is more consistent with English, including Spanish. Banerjee et al. (2024, arXiv:2406.11139) apply ROME and MEMIT in English, Spanish, Hindi and five other languages on 7B models and report low portability across languages. Beniwal et al. (2024, arXiv:2401.10521) study cross-lingual editing in BLOOM, mBERT and XLM-RoBERTa, including Hindi. Durrani et al. (2025, arXiv:2505.14393) survey the area.

**Position of this work.** The studies above either trace or edit, mostly on 7B-class models, and none of them compares per-layer causal profiles across English, Spanish and Hindi against a shuffled-fact null and then measures edit transfer, specificity and damage over every layer on the same model. We searched for such a combination and did not find one; this is an absence in a bounded search, not proof that none exists.

## 3. Method

**Models.** Llama-3.2-3B-Instruct (28 layers) and Gemma-2-2B (26 layers, base model), bf16, TransformerLens 3.9.0 without weight processing; one 8 GB GPU (peak 7.5 GiB). Weights are byte-identical (SHA-256) to the official repositories. Gemma is a base model and Llama an Instruct model; this confound is not separable here. **[PENDING: BLOOM-1b7 as a third family, pre-registered as A10, to test whether the weak Hindi overlap is a tokenisation effect.]**

**Facts and phrasings.** Hand-written EN/ES/HI facts in five relations (capital_of, capital_country, landmark_city, invention_inventor, event_year; 408 candidates, 447 for Gemma after adding 39 events), three phrasings per fact and language. For each relation and language the three best of six answer-first templates are chosen on a 30-fact tuning set (per model). A fact is kept if the canonical phrasing begins with the answer (strict, after spelling normalisation) in all three languages and one other phrasing contains it in its first line. Kept: 93 (Llama: capital_of 27, capital_country 38, landmark_city 6, invention_inventor 0, event_year 22) and 58 (Gemma: 11, 11, 6, 0, 30).

**Causal tracing.** Gaussian noise (sigma = 3 x pooled subject-embedding std, per-fact seed) is added to the subject-token embeddings. A clean activation is restored at one (layer, prompt token) site of the residual stream or MLP output, and the effect is the change in the teacher-forced log-probability of the whole answer relative to the corrupted run (full-answer scoring; the first-token variant is stored). A pair is valid if the answer probability falls below half its clean value and the total effect is positive (92 of 93 and 58 of 58 facts).

**Overlap.** Per fact and language pair, cosine and Spearman correlation of the 28 (or 26) layer MLP-output profile at each language's own last subject token; statistic = median over facts; null = fact i in one language paired with a permuted fact in the other (1000 permutations; pooled across relations, and, as a post-hoc sensitivity analysis that becomes primary for later data, within relation); z and one-sided p from the null; 95% CIs by 2000 bootstrap resamples of facts. Six primary tests per model (3 pairs x 2 statistics), Bonferroni alpha 0.0083.

**Editing [PENDING].** Rank-1 update of the MLP down-projection (ROME-style) at every layer for 10 facts per model, English prompt only, fitted on the full new answer; transfer ratio, specificity (20 controls) and damage (perplexity ratio on 100 WikiText-2 test sentences) as in A5. In-context baseline (A9) for comparison.

## 4. Results

### 4.1 Overlap against the null

| model | pair | MLP cosine median [95% CI] | null mean | z (pooled) | z (within relation) |
|---|---|---|---|---|---|
| Llama (n=92) | EN-ES | 0.627 [0.557, 0.702] | 0.519 | 4.22 | 3.29 |
| Llama | EN-HI | 0.509 [0.335, 0.594] | 0.429 | 2.45 | 1.79 |
| Llama | ES-HI | 0.423 [0.313, 0.529] | 0.328 | 2.47 | 1.98 |
| Gemma (n=58) | EN-ES | 0.464 [0.329, 0.542] | 0.292 | 4.81 | 3.85 |
| Gemma | EN-HI | 0.226 [0.168, 0.283] | 0.136 | 2.19 | 1.68 |
| Gemma | ES-HI | 0.137 [0.014, 0.267] | 0.134 | 0.08 | -0.45 |

English-Spanish overlap is the one result that survives every check. Within relation and after correction, only EN-ES remains significant in both models. Spearman statistics agree in direction (Llama EN-ES z = 5.15; Gemma EN-ES z = 4.55). Per relation with its own null the samples are small (6-38 facts) and mostly not significant (Llama capital_country EN-ES z = 2.75, capital_of 2.08, event_year 0.91). On the 33 facts both models keep, EN-ES is above the null in both (Llama z = 1.88, Gemma z = 4.09) and EN-HI is at the null in both (z = -0.21, 0.55).

### 4.2 Where the fact is looked up

The critical MLP layer (argmax at the last subject token) has median 3 (EN), 4 (ES), 4 (HI) in Llama (28 layers; 2-11% of facts in layers 8-19) and 6, 6, 6 in Gemma (26 layers; 26-43% in layers 8-19). The median absolute layer difference between languages is 2 in Llama and 2.5-4 in Gemma. The pre-registered expectation of a peak in layers 8-19 is not met for Llama. The median hides structure in Gemma (Figure 4, exploratory): the critical layer has modes at layers 4, 6 and 14; layer 14 holds the argmax for 17 of 58 English and 13 of 58 Spanish facts but for only 2 of 58 Hindi facts, and the cluster appears in three relations (English counts 3, 5, 9 in capital_country, capital_of, event_year). Llama has no such late cluster (3%, 2%, 11% of facts in layers 8-19 for EN, ES, HI).

### 4.3 Hindi tokenisation

Hindi subjects span a median of 5 (Llama) and 8 (Gemma) tokens against 2 for English. The EN-HI profile cosine correlates negatively with Hindi subject span length (Spearman -0.35 Llama, -0.39 Gemma). Splitting at the median span, Llama's EN-HI cosine is 0.597 for short spans (z = 1.69 pooled) and 0.199 for long spans (z = -1.22). So part of the weak Hindi result is a span-length effect; it does not explain it away, because EN-HI stays below EN-ES with short spans in both models. Hindi accuracy drops mainly for answers of 7 or more tokens (Llama 0.23 against 0.40-0.42; Gemma 0.02 against 0.21).

### 4.4 Edit transfer over layers **[PENDING]**
### 4.5 Robustness: phrasings (H8), non-copyable relations (F4), in-context baseline (F3) **[PENDING]**

## 5. Discussion **[PENDING until 4.4-4.5]**

What can already be said: the two models agree on the English-Spanish alignment and differ in where the lookup happens; Hindi results depend on how the Hindi subject is tokenised and on the null that is used. Any claim of a "language-agnostic space" would need the edit-transfer and robustness results.

## 6. Deviations from the pre-registration

H6 (EN-ES z > 2 and EN-ES > EN-HI) holds in both models. H9 (EN-ES z > 2 in at least two new or extended relations) is not supported: only event_year was testable (z = 0.91). The expected layer 8-19 peak is not met for Llama. Amendments A1-A9 (docs/HYPOTHESIS.md): critical-site definition; competitor from the corrupted run (v1 only); statistics definition; edit design; v2 design; matching rule; template selection after a poor first filter; Gemma pool extension; analysis rules (stratified null as primary for new data, non-copyable set, in-context baseline). The stratified null and the Hindi span split were added after seeing results and are labelled post hoc. v1 first-token results (66 facts, Llama only) are kept in `paper/paper.md` as a pilot; in v1 the Hindi pairs were at the null, which the v2 analysis shows to be sample- and metric-dependent.

## 7. Limitations

Two models, one of them a base and one an instruct model; hand-written facts; small per-relation samples; event_year answers are digit strings that can be copied across languages and make up half of Gemma's set; Hindi is hard to trace because subjects span many tokens; invention_inventor keeps 0 facts in both models. Layer-0 edit results were not reproducible between runs (v1).

## 8. Reproducibility

Code, configs, write-once run directories, seeds, logs and a claim manifest (`results/manifest.json`, `./reproduce.sh verify`) are in the repository. End-to-end rerun status: see `docs/LOG.md`.

## References

- Meng, K., Bau, D., Andonian, A., Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. arXiv:2202.05262.
- Dubey, A. et al. (2024). The Llama 3 Herd of Models. arXiv:2407.21783.
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
