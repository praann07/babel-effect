#!/usr/bin/env bash
# A15 (EN-HI interchange) GPU run: filter (Llama, Gemma), unit tests, full interchange (Llama, Gemma), analysis. Restartable via done markers.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python; export PYTHONPATH=code PYTHONIOENCODING=utf8
LOGD=analysis/_logs; DONE=$LOGD/night_e2h.done; touch "$DONE"
step() { local name=$1; shift; grep -qx "$name" "$DONE" && return 0
  echo "[$(date +%F' '%T)] START $name"
  if "$@"; then echo "$name" >> "$DONE"; echo "[$(date +%F' '%T)] OK $name"; return 0; fi
  echo "[$(date +%F' '%T)] FAIL $name"; return 1; }
py() { local log=$1; shift; $PY "$@" > "$LOGD/$log" 2>&1; }
step e2h_filter_llama py e2h_filter_llama.log code/e2_filter_hi.py --model llama
step e2h_filter_gemma2 py e2h_filter_gemma2.log code/e2_filter_hi.py --model gemma2
step e2h_full_llama py e2h_full_llama.log code/interchange.py --model llama --langs en,hi --max-pairs 200 --tag e2hfull_A
step e2h_full_gemma2 py e2h_full_gemma2.log code/interchange.py --model gemma2 --langs en,hi --max-pairs 200 --tag e2hfull_A
LL=$(ls -d analysis/run_*_e2hfull_A_llama | sort | tail -1); GG=$(ls -d analysis/run_*_e2hfull_A_gemma2 | sort | tail -1)
step e2h_analysis py e2h_analysis.log code/e2_analysis.py --langs en,hi --llama-run $LL --gemma-run $GG --tag e2h_analysis_A15
echo "[$(date +%F' '%T)] e2h script finished"
