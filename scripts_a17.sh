#!/bin/bash
# A17: leakage-excluded CPU re-analysis (no GPU). Run from the repo root.
set -e
export PYTHONPATH=code PYTHONIOENCODING=utf8
PY=${PY:-.venv/Scripts/python}
S_L=analysis/run_20261024_p3seeds_10240109_llama; S_G=analysis/run_20261024_p3seeds_10240109_gemma2
$PY code/seeds_analysis.py --llama-run $S_L --gemma-run $S_G --perm 10000 --exclude-leak --tag seeds_A13a_A17
$PY code/seeds_covariates.py --llama-run $S_L --gemma-run $S_G --exclude-leak --tag seeds_covariates_A13b_A17
E2L=$(ls -d analysis/run_*_e2full_A_llama | tail -1); E2G=$(ls -d analysis/run_*_e2full_A_gemma2 | tail -1)
$PY code/e2_analysis.py --llama-run $E2L --gemma-run $E2G --exclude-leak --tag e2_analysis_A13_A17
HL=$(ls -d analysis/run_*_e2h2full_A_llama | tail -1); HG=$(ls -d analysis/run_*_e2h2full_A_gemma2 | tail -1)
$PY code/e2_analysis.py --llama-run $HL --gemma-run $HG --langs en,hi --exclude-leak --tag e2h2_analysis_A16_A17
echo A17 CPU done
