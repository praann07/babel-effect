#!/usr/bin/env bash
# Layer-sweep edits (A5): Llama then Gemma, 10 facts x every layer each. Resume a crashed stage with
#   RESUME_LLAMA=<run dir> or RESUME_GEMMA=<run dir> ./scripts_run_sweeps.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python
export PYTHONPATH=code PYTHONIOENCODING=utf8
LT=analysis/run_20261015_p3trace_v2_llama
GT=analysis/run_20261015_p3trace_v2c_gemma2
if [ -n "${RESUME_LLAMA:-}" ]; then
  $PY code/knowledge_edit_v2.py --model llama --trace-run $LT --resume "$RESUME_LLAMA" > analysis/_logs/sweep_llama_resume.log 2>&1
  LRUN="$RESUME_LLAMA"
else
  $PY code/knowledge_edit_v2.py --model llama --trace-run $LT --tag p4sweep > analysis/_logs/sweep_llama.log 2>&1
  LRUN=$(ls -d analysis/run_*_p4sweep_llama | tail -1)
fi
echo "llama sweep done: $LRUN"; $PY code/eval_sweep.py "$LRUN" > analysis/_logs/eval_sweep_llama.log 2>&1
if [ -n "${RESUME_GEMMA:-}" ]; then
  $PY code/knowledge_edit_v2.py --model gemma2 --trace-run $GT --resume "$RESUME_GEMMA" > analysis/_logs/sweep_gemma_resume.log 2>&1
  GRUN="$RESUME_GEMMA"
else
  $PY code/knowledge_edit_v2.py --model gemma2 --trace-run $GT --tag p4sweep > analysis/_logs/sweep_gemma.log 2>&1
  GRUN=$(ls -d analysis/run_*_p4sweep_gemma2 | tail -1)
fi
echo "gemma sweep done: $GRUN"; $PY code/eval_sweep.py "$GRUN" > analysis/_logs/eval_sweep_gemma.log 2>&1
echo ALL_DONE
