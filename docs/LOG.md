# LOG (decisions)

## 2026-10-12 P0
- Project root: ...\reasearch\babel-effect (git toplevel verified). Sibling ../RRAP is archived: never touched.
- torch installed from the CUDA 12.8 wheel index (RTX 5060 laptop GPU, Blackwell needs cu128; default PyPI wheel is CPU-only on Windows).
- TransformerLens 4.x removed `HookedTransformer` -> pinned `transformer-lens==3.9.0` (needs re-freeze of requirements.txt).
- Model ladder substitutions (reason): Llama-3.2-3B-Instruct: gated, no HF token (GatedRepoError). Gemma-2-2B: gated, same. Pythia-1.4B: download stalled at 1.07 GB (HF throughput ~0.7 MB/s measured; Xet transfer also stalled earlier, so HF_HUB_DISABLE_XET=1 is set). GPT-2 used only as an engineering smoke test of the pipeline; it is English-only and NOT usable for the EN/ES/HI science.
- Estimated download time at ~0.7 MB/s: Llama-3.2-3B bf16 (~6.4 GB) ~2.5 h; Gemma-2-2B (~5 GB) ~2 h. Network speed may vary.
- ACTION (user): create HF account/token, accept Llama + Gemma licenses, run `hf auth login` in an external PowerShell.
- P1 part 1: built 154 candidate facts (65 capital_of, 39 landmark_city, 50 capital_country), EN/ES/HI, hand-written translations, 2 Hindi template variants (variant chosen later by whole-set accuracy, not per fact). Per-fact corruption seed stored (sha256 of id). NOT yet filtered: filtering needs the real model.
- P0 smoke test PASSED on GPT-2 (substitute, ladder rung 4): top-1 " Paris" p=0.0706, then London 0.0585, Amsterdam 0.0485, Berlin 0.0355, New 0.0229; peak VRAM 0.39 GiB (bf16). Pipeline (venv, CUDA, TransformerLens 3.9.0, hooks) verified. Gate is met only for GPT-2; re-run on Llama-3.2-3B / Gemma-2-2B once HF access exists. requirements.txt re-frozen with the transformer-lens pin.
