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
