#!/usr/bin/env bash
# Usage:
#   ./reproduce.sh verify        check results/manifest.json against the saved run artifacts (no GPU, seconds)
#   ./reproduce.sh test          unit tests for matching rules and overlap statistics (CPU, seconds)
#   ./reproduce.sh figures       regenerate paper/figures from the saved run artifacts (CPU, seconds)
#   ./reproduce.sh check         test + verify + figures: the one-command CPU check of the paper's numbers
#   ./reproduce.sh full [TAG]    re-run P1-P5 into new write-once run dirs tagged TAG (default "repro"), then compare
#                                (about 1.5-2 h on an 8 GB GPU). Needs models/llama-3.2-3b-instruct/ and data/wikitext/
#                                (see README "Data and model" for the download commands).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python
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
    ;;
  check)
    $PY -m unittest discover -s tests
    $PY code/make_manifest.py verify
    $PY code/make_figures.py
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
  *) echo "usage: $0 verify | test | figures | check | full [TAG]"; exit 2;;
esac
