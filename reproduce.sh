#!/usr/bin/env bash
# Usage:
#   ./reproduce.sh verify        check results/manifest.json against the saved run artifacts (no GPU, seconds)
#   ./reproduce.sh test          unit tests for matching rules and overlap statistics (CPU, seconds)
#   ./reproduce.sh figures       regenerate paper/figures from the saved run artifacts (CPU, seconds)
#   ./reproduce.sh check         test + verify + figures: the one-command CPU check of the paper's numbers
#   ./reproduce.sh cpu [TAG]     re-run the CPU analyses of the v2 follow-up (A12a confound, A13a noise re-trace, A13b covariates,
#                                A13 E2 interchange, A14 overlap at 10,000 permutations) from the saved raw traces and E2 effects into new
#                                write-once dirs tagged TAG (default "repro"), then compare every numeric leaf with the saved results (minutes, no GPU)
#   ./reproduce.sh full [TAG]    re-run P1-P5 into new write-once run dirs tagged TAG (default "repro"), then compare
#                                (about 1.5-2 h on an 8 GB GPU). Needs models/llama-3.2-3b-instruct/ and data/wikitext/
#                                (see README "Data and model" for the download commands).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/Scripts/python}
[ -x "$PY" ] || PY=.venv/bin/python
export PYTHONPATH=code PYTHONIOENCODING=utf8
newest() { ls -d analysis/run_*_"$1" | tail -1; }

case "${1:-verify}" in
  verify)
    $PY code/make_manifest.py verify
    ;;
  test)
    $PY -m unittest discover -s tests
    ;;
  figures)
    $PY code/make_figures.py
    $PY code/make_tables.py
    ;;
  check)
    $PY -m unittest discover -s tests
    $PY code/make_manifest.py verify
    $PY code/make_figures.py
    $PY code/make_tables.py
    ;;
  cpu)
    TAG="${2:-repro}"
    L=analysis/run_20261024_p3seeds_10240109_llama; G=analysis/run_20261024_p3seeds_10240109_gemma2
    EL=analysis/run_20261025_e2full_A_llama; EG=analysis/run_20261025_e2full_A_gemma2
    $PY code/confound_overlap.py --perm 10000 --tag "confound_$TAG"
    $PY code/seeds_analysis.py --llama-run $L --gemma-run $G --tag "seeds_$TAG"
    $PY code/seeds_covariates.py --llama-run $L --gemma-run $G --tag "seedscov_$TAG"
    $PY code/e2_analysis.py --llama-run $EL --gemma-run $EG --tag "e2_$TAG"
    $PY code/circuit_overlap.py analysis/run_20261015_p3trace_v2_llama --perm 10000 --tag "ovl_$TAG"
    $PY code/circuit_overlap.py analysis/run_20261015_p3trace_v2c_gemma2 --perm 10000 --tag "ovg_$TAG"
    rc=0
    cmpres() { $PY code/compare_results.py "$1/results.json" "$(newest "$2")/results.json" "${@:3}" || rc=1; }
    cmpres analysis/run_20261024_confound_A12a "confound_$TAG"
    cmpres analysis/run_20261024_seeds_A13a_10240109 "seeds_$TAG"
    cmpres analysis/run_20261024_seeds_covariates_A13b "seedscov_$TAG"
    cmpres analysis/run_20261025_e2_analysis_A13 "e2_$TAG"
    cmpres analysis/run_20261025_p3overlap_v2_perm10k_llama "ovl_$TAG"
    cmpres analysis/run_20261025_p3overlap_v2_perm10k_gemma2 "ovg_$TAG"
    exit $rc
    ;;
  full)
    TAG="${2:-repro}"
    mkdir -p data/repro
    $PY code/build_facts.py
    $PY code/filter_facts.py --tag "p1filter_$TAG" --out data/repro/facts.json
    $PY code/causal_trace.py --facts data/repro/facts.json --subset all --tag "p3trace_$TAG"
    $PY code/circuit_overlap.py "$(newest "p3trace_$TAG")" --tag "p3overlap_$TAG"
    $PY code/knowledge_edit.py --facts data/repro/facts.json --trace-run "$(newest "p3trace_$TAG")" --tag "p4edit_$TAG"
    $PY code/eval_transfer.py "$(newest "p4edit_$TAG")"
    $PY code/compare_repro.py "$TAG"
    ;;
  *) echo "usage: $0 verify | test | figures | check | cpu [TAG] | full [TAG]"; exit 2;;
esac
