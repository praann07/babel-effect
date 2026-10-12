# NEXT
- [x] P0 env: repo anchored at babel-effect, venv, deps pinned, torch cu128 CUDA works
- [ ] P0 gate: probe predicts "Paris" (smoke test running; see docs/LOG.md for model used)
- [ ] BLOCKER (user action): Llama-3.2-3B-Instruct and Gemma-2-2B are gated; need free HF token + license acceptance. Pythia-1.4B/GPT-2 are English-centric, so ES/HI results would be meaningless on them.
- [ ] P1 data: /data/facts.json, 150 triples x EN/ES/HI, keep facts correct in all three, target n>=50
