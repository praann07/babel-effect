# NEXT
- [x] P0 env: repo anchored at babel-effect, venv, deps pinned, torch cu128 CUDA works
- [x] P0 gate: Llama-3.2-3B-Instruct predicts " Paris" (p=0.613); peak VRAM 7.49 GiB (analysis/p0_smoke_llama.json)
- [ ] Verify mirror shard sha256 against meta-llama once Meta approves access (pending)
- [~] P1 data: 154 candidates built (data/facts_candidates.json); next: code/filter_facts.py (pick Hindi template variant by whole-set accuracy, keep facts correct in EN+ES+HI, target n>=50, save data/facts.json + accuracy table)
- [ ] NEXT: write docs/HYPOTHESIS.md and commit it BEFORE any P3 computation; then code/causal_trace.py (P2, 10 facts)
- [ ] P2 tracing, P3 overlap, P4 editing, P5 transfer, P6 writing
