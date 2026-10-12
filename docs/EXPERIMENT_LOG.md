# EXPERIMENT LOG
| run ID | hypothesis | config | result | verdict |
|---|---|---|---|---|
| P0-smoke-gpt2 | model loads and predicts Paris | gpt2 bf16 cuda, prompt "The Eiffel Tower is located in the city of" | top1 Paris p=0.0706, VRAM 0.39 GiB | pipeline OK; not the target model |
| P0-smoke-llama3.2-3b | model loads and predicts Paris | unsloth/Llama-3.2-3B-Instruct (local), bf16 cuda, `from_pretrained_no_processing`, same prompt | top1 " Paris" p=0.6132 (then "..." 0.0688, " what" 0.0368, " which" 0.0368, "?\n" 0.0287); peak VRAM 7.49 GiB | P0 gate MET on target model; VRAM is near the 8 GiB cap, so later runs need chunked caching (artifact: analysis/p0_smoke_llama.json) |
| P1-filter-llama3.2-3b | enough facts are correct in EN+ES+HI | filter_facts.py, greedy, exact-object prefix match, HI variant per relation by whole-set accuracy | 154 candidates -> 52 kept (capital_country 26, capital_of 22, landmark_city 4); EN acc 0.95-1.00, ES 0.65-0.78, HI 0.18-0.58 (variant 0 won every relation) | P1 gate MET (n=52 >= 50, table saved: analysis/run_20261012_p1filter/accuracy.md); relation mix is skewed |
