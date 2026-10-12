# NEXT
- [x] P0 env: repo anchored at babel-effect, venv, deps pinned, torch cu128 CUDA works
- [x] P0 gate: Llama-3.2-3B-Instruct predicts " Paris" (p=0.613); peak VRAM 7.49 GiB (analysis/p0_smoke_llama.json)
- [ ] Verify mirror shard sha256 against meta-llama once Meta approves access (pending)
- [x] P1 data: data/facts.json n=67 (33 capital_country, 32 capital_of, 2 landmark_city); table at analysis/run_20261012_p1filter_v2/accuracy.md
- [x] docs/HYPOTHESIS.md committed before any tracing/overlap computation (pre-registration)
- [x] P2 tracing on 10 facts: pattern visible, 30/30 valid; MLP peak is layers 0-5, not the pre-registered 8-19 (H3 not met; logged)
- [~] P3: full trace of 67 facts running (analysis/run_20261012_p3trace); next: code/circuit_overlap.py per HYPOTHESIS A3
- [ ] P2 tracing, P3 overlap, P4 editing, P5 transfer, P6 writing
