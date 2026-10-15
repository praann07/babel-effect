#!/usr/bin/env bash
# Gemma-2-2B with its own selected templates: select -> rebuild candidates -> filter -> span check. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python
export PYTHONPATH=code PYTHONIOENCODING=utf8
cp -n data/v2/facts_gemma2.json data/v2/facts_gemma2_llama_templates.json || true
echo "[1/4] select templates"; $PY code/select_templates.py --model gemma2 > analysis/_logs/select_templates_gemma2.log 2>&1
echo "[2/4] build candidates"; $PY code/build_facts_v2.py --templates data/v2/templates_gemma2.json --out data/v2/facts_candidates_gemma2.json
echo "[3/4] filter"; $PY code/filter_facts_v2.py --model gemma2 --candidates data/v2/facts_candidates_gemma2.json --tag p1v2b_gemma2 > analysis/_logs/filter_v2b_gemma2.log 2>&1
echo "[4/4] spans"; $PY code/check_spans.py --model gemma2 > analysis/_logs/spans_gemma2b.log 2>&1
echo ALL_DONE
