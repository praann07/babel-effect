#!/usr/bin/env bash
# Post-run analyses: H8 (after the extra chain), then F7 (30 edit facts at the A9-rule layers) once the whole queue is done.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python
export PYTHONPATH=code PYTHONIOENCODING=utf8
declare -A TR=( [llama]=analysis/run_20261015_p3trace_v2_llama [gemma2]=analysis/run_20261015_p3trace_v2c_gemma2 )
until grep -q ALL_DONE analysis/_logs/extra_chain.log 2>/dev/null; do sleep 60; done
for M in llama gemma2; do
  H=$(ls -d analysis/run_*_h8trace_$M 2>/dev/null | tail -1)
  [ -n "$H" ] && $PY code/h8_analysis.py --main-run ${TR[$M]} --h8-run "$H" --model $M --tag h8 > analysis/_logs/h8_$M.log 2>&1
done
echo "h8 done"
until grep -q ALL_DONE analysis/_logs/llama1b_chain.log 2>/dev/null; do sleep 60; done
for M in llama gemma2; do
  SW=$(ls -d analysis/run_*_p4sweep_$M | head -1)
  L=$($PY code/select_f7_layers.py "$SW") || { echo "[$M] layer rule failed"; continue; }
  echo "[$M] F7 layer $L (+ traced, L//2)"
  $PY code/knowledge_edit_v2.py --model $M --trace-run ${TR[$M]} --n-edit 30 --skip-first 10 --layers $L --add-traced --tag p4f7 > analysis/_logs/f7_$M.log 2>&1
  F=$(ls -d analysis/run_*_p4f7_$M | tail -1)
  $PY code/eval_sweep.py "$F" > analysis/_logs/eval_f7_$M.log 2>&1
done
echo ALL_DONE
