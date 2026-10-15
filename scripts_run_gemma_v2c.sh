#!/usr/bin/env bash
# Gemma-2-2B final v2 pool (A8): own templates + extended events -> filter -> span check -> full-answer trace -> overlap.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python
export PYTHONPATH=code PYTHONIOENCODING=utf8
cp -n data/v2/facts_gemma2.json data/v2/facts_gemma2_own_templates_45.json || true
echo "[1/5] build"; $PY code/build_facts_v2.py --templates data/v2/templates_gemma2.json --with-events2 --out data/v2/facts_candidates_gemma2.json
echo "[2/5] filter"; $PY code/filter_facts_v2.py --model gemma2 --candidates data/v2/facts_candidates_gemma2.json --tag p1v2c_gemma2 > analysis/_logs/filter_v2c_gemma2.log 2>&1
echo "[3/5] spans"; $PY code/check_spans.py --model gemma2 > analysis/_logs/spans_gemma2c.log 2>&1
echo "[4/5] trace"; $PY code/causal_trace_v2.py --model gemma2 --subset all --tag p3trace_v2c > analysis/_logs/trace_v2c_gemma2.log 2>&1
echo "[5/5] overlap"; $PY code/circuit_overlap.py "$(ls -d analysis/run_*_p3trace_v2c_gemma2 | tail -1)" --tag p3overlap_v2c_gemma2 > analysis/_logs/overlap_v2c_gemma2.log 2>&1
echo ALL_DONE
