#!/usr/bin/env bash
# Gemma-2-2B v2 pipeline: filter -> span check -> full-answer trace. Stops at the first failing stage.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python
export PYTHONPATH=code PYTHONIOENCODING=utf8
echo "[1/3] filter"; $PY code/filter_facts_v2.py --model gemma2 > analysis/_logs/filter_v2_gemma2.log 2>&1
echo "[2/3] spans";  $PY code/check_spans.py --model gemma2 > analysis/_logs/spans_gemma2.log 2>&1
echo "[3/3] trace";  $PY code/causal_trace_v2.py --model gemma2 --subset all --tag p3trace_v2 > analysis/_logs/trace_v2_gemma2.log 2>&1
echo ALL_DONE
