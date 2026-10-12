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
./reproduce.sh verify                                   # checks the manifest; `./reproduce.sh full` re-runs P1-P5
```

## Headline results (Llama-3.2-3B-Instruct, n=66 facts for tracing, 10 for editing; details in `paper/paper.md`)
- EN-ES MLP-profile overlap is above the shuffled-fact null (median cosine 0.748, z=6.06); EN-HI and ES-HI are not (z=0.39, 0.38).
- English critical MLP layer is 0-5 (median 2), not the pre-registered 8-19.
- Edit at the traced layer: EN success 9/10, transfer ratio ES 0.147, HI 0.002. Edit at layer 14: EN 10/10, ES 0.203, HI 0.244, median specificity loss 0.351. Perplexity ratio below 1.02 everywhere.

## Layout
`code/` scripts · `data/` facts · `analysis/` write-once run dirs · `paper/` · `docs/` LOG, EXPERIMENT_LOG, HYPOTHESIS, NEXT

## Phases
P0 env, P1 data, P2 tracing, P3 overlap (pre-registered in docs/HYPOTHESIS.md), P4 editing, P5 transfer, P6 writing. See `docs/NEXT.md`.
