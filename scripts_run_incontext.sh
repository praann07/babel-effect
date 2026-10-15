#!/usr/bin/env bash
# Waits for the sweeps and the F4/F5 chain, then runs the in-context baseline (F3) for both models.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python
export PYTHONPATH=code PYTHONIOENCODING=utf8
until grep -q ALL_DONE analysis/_logs/extra_chain.log 2>/dev/null; do sleep 60; done
for M in llama gemma2; do
  SW=$(ls -d analysis/run_*_p4sweep_$M | tail -1)
  $PY code/incontext_baseline.py --model $M --sweep-run "$SW" > analysis/_logs/incontext_$M.log 2>&1
done
echo ALL_DONE
