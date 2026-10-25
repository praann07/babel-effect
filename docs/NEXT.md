# NEXT (to-do list, updated 2026-10-25)

## 2026-10-25 status
- [x] A12/A12a confound analysis, A13a independent-noise re-trace, A13b covariates, A13 E2 interchange (verdict SHARED EN-ES, both models; docs/EXPERIMENT_LOG.md). Paper section 4.7 drafted.
- [ ] GPU (needs laptop): E2 Hindi; E3 name-dissimilar facts (more relations); E4 base vs instruct. (Gemma F7 done 2026-10-25; Llama-1B pilot kept 38 facts and BLOOM-1b7 pilot 34, both below the 50 needed, so no full run.)
- [ ] CPU: related work + arXiv-verified citations; Llama in-context baseline anomaly (EN 0/10); main tables with 10,000 permutations; Fig. 4 buckets; reproduce script + clean-clone run; external pre-registration timestamp (Zenodo/OSF); full paper rewrite and abstract update.
- [ ] Open science question: tracing increment does not predict interchange per fact in Llama (rho -0.04); explain or drop the link.


Legend: [x] done, [~] running, [ ] to do. GPU items wait for the layer sweep (scripts_run_sweeps.sh). It was interrupted on 10-15 and 10-18 (laptop sleep) and resumed on 10-19 03:07 from saved rows; scripts_watchdog.sh restarts it if it dies. Estimate: Llama about 4-5 h more, Gemma about 6 h, then the queued chains.

## Done
- [x] P0-P5 on Llama-3.2-3B (v1 pilot) and paper draft v1; repo public (praann07/babel-effect)
- [x] Model research report (reports/Models for multilingual fact editing.md), Gemma-2-2B weights verified
- [x] A5-A8 pre-registered; v2 data (5 relations, 3 phrasings); full-answer tracing; Llama v2 (n=92) and Gemma-2-2B v2 (n=58) overlap; cross-model table (33 shared facts); related work (20 verified refs); manifest 356 claims

## Fixes to make the paper reviewer-proof (do one at a time, commit after each)
- [x] F1 Pre-register the relation-stratified null as the primary test for everything from now on (A9). CPU, now.
- [x] F2 Hindi diagnostic: Hindi accuracy and overlap versus Hindi token length (answers "is it a tokenizer artifact?"). CPU, now.
- [~] F3 In-context editing baseline (put the new fact in the prompt) to put the weight-edit transfer numbers in context. Code now, run after the sweep.
- [~] F4 Non-copyable facts: add relations whose answers differ across languages (landmark_country, city_country); target >= 50 non-copyable valid facts per model. Data now, GPU after the sweep.
- [~] F5 Phrasing robustness (H8): trace phrasings 1 and 2 on 30 facts per model. GPU after the sweep.
- [~] F6 Third model: forward-only pilot (Llama-3.2-1B first; Qwen3-1.7B-Base / BLOOM-1b7 next) and run the full pipeline only if it keeps >= 50 facts.
- [~] F10 BLOOM-1b7 as a third family (A10 pre-registered; weights downloading, chain scripts_run_bloom.sh queued behind scripts_run_post.sh). Tests whether weak Hindi overlap is a tokenisation effect.
- [ ] F7 More edit facts: after the sweep, rerun the best layers on about 30 facts per model.
- [ ] F8 Paper rewrite for v2 (tables, figures, honest verdicts on H6-H9), HYPOTHESIS verdicts, README.
- [~] F9 reproduce.sh: `./reproduce.sh check` (tests + manifest + figures) done; the full v2 end-to-end rerun from a clean clone is still to do.
- [x] CPU extras done 2026-10-19: paper figures (paper/figures), unit tests (tests/), watchdog.

## Hypotheses status (docs/HYPOTHESIS.md, Verdicts section)
- H6 supported (both models). H7 supported in Gemma, contradicted for ES in Llama. H8 supported (both). H9 not supported. F4 met for Llama, not Gemma. F3/F7 descriptive (Llama F7 done).
- B1-B3 (A10, BLOOM-1b7): pending.

## Left to run (GPU; start with Desktop "Start Experiments.bat" or the two nohup commands in the queue2 script header)
1. Gemma F7 (20 new facts; partial dir analysis/run_20261020_p4f7_gemma2 is kept unused, the queue starts a fresh one)
2. BLOOM-1b7 pilot (A10; TransformerLens BLOOM path untested)
3. Llama-3.2-1B pilot (full pipeline only if >= 50 facts)

## Left to write (CPU)
- Add Gemma F7 and the pilots to paper section 4.5/5 once run; related-work additions after verifying 2411.08745; README refresh; reproduce.sh for the v2 pipeline + one clean-clone rerun; fix Start Experiments.bat.

## Waiting on the user
- Venue choice; optional Hindi-speaker spot check; keep the laptop plugged in while the GPU queue runs.
