#!/usr/bin/env bash
# A13 (E2) GPU run: Gemma pilot (excluded), Llama full, Gemma full. Restartable via done markers; write-once run dirs use time-stamped tags.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python; export PYTHONPATH=code PYTHONIOENCODING=utf8
LOGD=analysis/_logs; DONE=$LOGD/night_e2.done; touch "$DONE"; TS=$(date +%m%d%H%M)
step() { local name=$1; shift; grep -qx "$name" "$DONE" && return 0
  echo "[$(date +%F' '%T)] START $name"
  if "$@"; then echo "$name" >> "$DONE"; echo "[$(date +%F' '%T)] OK $name"; return 0; fi
  echo "[$(date +%F' '%T)] FAIL $name"; return 1; }
py() { local log=$1; shift; $PY "$@" > "$LOGD/$log" 2>&1; }
step e2_pilot_gemma2 py e2_pilot_gemma2.log code/interchange.py --model gemma2 --pilot 10 --tag e2pilot_$TS
step e2_full_llama py e2_full_llama.log code/interchange.py --model llama --tag e2full_A
step e2_full_gemma2 py e2_full_gemma2.log code/interchange.py --model gemma2 --tag e2full_A
LL=$(ls -d analysis/run_*_e2full_A_llama | sort | tail -1); GG=$(ls -d analysis/run_*_e2full_A_gemma2 | sort | tail -1)
step e2_analysis py e2_analysis.log code/e2_analysis.py --llama-run $LL --gemma-run $GG --tag e2_analysis_A13
echo "[$(date +%F' '%T)] e2 script finished"
