# NEXT
- [x] P0 env: repo anchored at babel-effect, venv, deps pinned, torch cu128 CUDA works
- [x] P0 gate: Llama-3.2-3B-Instruct predicts " Paris" (p=0.613); peak VRAM 7.49 GiB (analysis/p0_smoke_llama.json)
- [ ] Verify mirror shard sha256 against meta-llama once Meta approves access (pending)
- [x] P1 data: data/facts.json n=52 (26 capital_country, 22 capital_of, 4 landmark_city); table at analysis/run_20261012_p1filter/accuracy.md
- [x] docs/HYPOTHESIS.md committed before any tracing/overlap computation (pre-registration)
- [ ] NEXT: code/causal_trace.py (P2, 10 facts, heatmaps; gate = mid-layer peak)
- [ ] P2 tracing, P3 overlap, P4 editing, P5 transfer, P6 writing
