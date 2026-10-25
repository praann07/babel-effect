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
Llama-3.2-3B-Instruct is gated on Hugging Face (access was granted for this project). Accept the license on each model page and run `huggingface-cli login` (free token) before `python code/smoke_test.py`. Without access the loader falls down the ladder (Pythia-1.4B, GPT-2); every substitution is logged in `docs/LOG.md`.

## Data and model
Weights (byte-identical to Meta's repo, SHA-256 checked) and WikiText-2 are fetched with the resumable downloader (the stock Hugging Face downloader stalls on some networks):
```
python code/dl_hf.py meta-llama/Llama-3.2-3B-Instruct models/llama-3.2-3b-instruct config.json generation_config.json model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer_config.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors --token-file <file with your HF token>
python code/dl_hf.py datasets/Salesforce/wikitext data/wikitext wikitext-2-raw-v1/test-00000-of-00001.parquet wikitext-2-raw-v1/train-00000-of-00001.parquet
```

## Run order (each stage writes a write-once `analysis/run_<date>_<tag>/`)
```
set PYTHONPATH=code
python code/build_facts.py                              # P1: 289 candidates
python code/filter_facts.py --tag p1filter_v2           # P1: data/facts.json (67 facts) + accuracy table
python code/causal_trace.py --subset p2 --tag p2trace_v3 && python code/summarize_trace.py analysis/run_<date>_p2trace_v3   # P2
python code/causal_trace.py --subset all --tag p3trace  # P3: full tracing
python code/circuit_overlap.py analysis/run_<date>_p3trace --tag p3overlap
python code/knowledge_edit.py --tag p4edit              # P4
python code/eval_transfer.py analysis/run_<date>_p4edit # P5 tables
python code/make_manifest.py build                      # P6 claim manifest
./reproduce.sh verify                                   # checks the manifest against saved artifacts (passes)
./reproduce.sh cpu                                      # re-runs A12a/A13a/A13b/E2/A14 analyses from saved traces and compares (CPU, minutes)
# `./reproduce.sh full` re-runs P1-P5; only P1 was reproduced end to end (see docs/LOG.md, REPRODUCTION STATUS)
```

## Headline results (v2: Llama-3.2-3B-Instruct 92 facts, Gemma-2-2B 58 facts for tracing; details in `paper/paper_v2_draft.md`)
- EN-ES causal-profile alignment is above the shuffled-fact null in both models (MLP cosine z = 4.0 Llama, 4.8 Gemma; p = 0.0001 with 10,000 permutations). It survives independent noise per language (A13a) and, in Llama, control for name overlap and answer identity (A13b). Hindi overlap is weaker and mostly not significant after correction.
- Cross-language subject interchange (E2, A13, English-Spanish, four relations): swapping the subject-token state between languages moves the answer about as much as a same-language swap (ratio 0.90-1.17, 95% lower bounds >= 0.85) in both models; pre-registered verdict SHARED. Caveat: per-fact tracing similarity does not predict interchange success in Llama (rho -0.04). Hindi not yet tested.
- Edit transfer (10-30 facts per condition) depends on layer and model; Spanish transfer is partial and Hindi small. These samples are small and the edit results are descriptive.
- Every number is in `results/manifest.json` (1059 claims, `./reproduce.sh verify`). `./reproduce.sh cpu` regenerates the follow-up analyses from the saved traces and reproduces the saved results exactly (needs the tokenizers in `models/`).

## Layout
`code/` scripts · `data/` facts · `analysis/` write-once run dirs · `paper/` · `docs/` LOG, EXPERIMENT_LOG, HYPOTHESIS, NEXT

## Phases
P0 env, P1 data, P2 tracing, P3 overlap (pre-registered in docs/HYPOTHESIS.md), P4 editing, P5 transfer, P6 writing. See `docs/NEXT.md`.
