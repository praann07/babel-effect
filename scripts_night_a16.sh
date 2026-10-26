#!/usr/bin/env bash
# A16 (HI->EN interchange, enlarged Hindi fact set) GPU run: filters, full runs (Llama, Gemma), analysis. Restartable via done markers.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python; export PYTHONPATH=code PYTHONIOENCODING=utf8
LOGD=analysis/_logs; DONE=$LOGD/night_a16.done; touch "$DONE"
step() { local name=$1; shift; grep -qx "$name" "$DONE" && return 0
  echo "[$(date +%F' '%T)] START $name"
  if "$@"; then echo "$name" >> "$DONE"; echo "[$(date +%F' '%T)] OK $name"; return 0; fi
  echo "[$(date +%F' '%T)] FAIL $name"; return 1; }
py() { local log=$1; shift; $PY "$@" > "$LOGD/$log" 2>&1; }
step a16_filter_llama py a16_filter_llama.log code/e2_filter_hi.py --model llama --a16 --tag e2h2filter
step a16_filter_gemma2 py a16_filter_gemma2.log code/e2_filter_hi.py --model gemma2 --a16 --tag e2h2filter
for m in llama gemma2; do
  step a16_full_$m py a16_full_$m.log code/interchange.py --model $m --langs en,hi --facts data/v2/e2h2_facts_$m.json --directions hi2en --k-src 4 --shuffle-pairs --max-pairs 300 --tag e2h2full_A
done
LL=$(ls -d analysis/run_*_e2h2full_A_llama | sort | tail -1); GG=$(ls -d analysis/run_*_e2h2full_A_gemma2 | sort | tail -1)
step a16_analysis py a16_analysis.log code/e2_analysis.py --langs en,hi --llama-run $LL --gemma-run $GG --tag e2h2_analysis_A16
echo "[$(date +%F' '%T)] a16 script finished"
