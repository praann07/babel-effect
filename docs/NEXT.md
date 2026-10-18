# NEXT (to-do list, updated 2026-10-19)

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

## Hypotheses status (docs/HYPOTHESIS.md A5)
- H6 (EN-ES z>2; EN-ES > EN-HI): supported for Llama (z=4.22) and Gemma (z=4.81)
- H7 (mid-layer edits transfer more than layers 0-5): pending the sweep
- H8 (phrasing robustness): pending F5 (analysis queued in scripts_run_post.sh)
- B1-B3 (A10, BLOOM-1b7): pending
- H9 (EN-ES z>2 in >= 2 new/extended relations): not supported (only event_year testable, z=0.91)

## Waiting on the user
- Nothing now. Keep the laptop awake and plugged in while the sweep runs.

## Queue order (all GPU, one at a time; each waits on the previous log's ALL_DONE)
scripts_run_sweeps.sh (Llama then Gemma sweep; scripts_watchdog.sh restarts it) -> scripts_run_extra.sh -> scripts_run_incontext.sh -> scripts_run_llama1b.sh -> scripts_run_post.sh (H8 analysis, F7) -> scripts_run_bloom.sh. Logs: analysis/_logs/*_chain.log.
