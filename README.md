# The Babel Effect
Cross-lingual localization and editing of factual knowledge in small open-weight LLMs (EN / ES / HI).

**Question.** Are stored facts held in language-specific circuits or a language-agnostic conceptual space? Plan: causal tracing in three languages, circuit-overlap statistics against a permutation null, a rank-1 ROME-style MLP edit in one language, and cross-lingual transfer / specificity / damage measurements.

## Constraints
$0 API spend; local single GPU (<= 8 GB VRAM); open weights only.

## Setup (Windows, RTX 50-series needs the CUDA 12.8 wheel)
```
cd babel-effect
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt   # contains the pinned set; torch line already cu128
python code/smoke_test.py
```
Note: TransformerLens 4.x removed `HookedTransformer`; this repo pins `transformer-lens==3.9.0`.

## Model access
Llama-3.2-3B-Instruct and Gemma-2-2B are gated on Hugging Face. Accept the license on each model page and run `huggingface-cli login` (free token) before `python code/smoke_test.py`. Without access the loader falls down the ladder (Pythia-1.4B, GPT-2); every substitution is logged in `docs/LOG.md`.

## Layout
`code/` scripts · `data/` facts · `analysis/` write-once run dirs · `paper/` · `docs/` LOG, EXPERIMENT_LOG, HYPOTHESIS, NEXT

## Phases
P0 env, P1 data, P2 tracing, P3 overlap (pre-registered in docs/HYPOTHESIS.md), P4 editing, P5 transfer, P6 writing. See `docs/NEXT.md`.
