#!/usr/bin/env bash
# A13a night: smoke test, Llama and Gemma independent-noise re-trace, then the CPU analysis. Restartable (done markers); write-once run dirs use time-stamped tags.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python
export PYTHONPATH=code PYTHONIOENCODING=utf8
LOGD=analysis/_logs; DONE=$LOGD/night_a13a.done; touch "$DONE"
TS=$(date +%m%d%H%M)
step() { local name=$1; shift; grep -qx "$name" "$DONE" && return 0
  echo "[$(date +%F' '%T)] START $name"
  if "$@"; then echo "$name" >> "$DONE"; echo "[$(date +%F' '%T)] OK $name"; return 0; fi
  echo "[$(date +%F' '%T)] FAIL $name"; return 1; }
py() { local log=$1; shift; $PY "$@" > "$LOGD/$log" 2>&1; }
latest() { local d; for d in $(ls -d $1 2>/dev/null | sort -r); do [ -f "$d/records.json" ] && { echo "$d"; return 0; }; done; return 1; }
step smoke_llama py a13a_smoke_llama.log code/trace_seeds.py --model llama --limit 2 --tag p3seedssmoke_$TS || { echo "smoke failed; stopping"; exit 1; }
step trace_llama py a13a_llama.log code/trace_seeds.py --model llama --tag p3seeds_$TS
step trace_gemma2 py a13a_gemma2.log code/trace_seeds.py --model gemma2 --tag p3seeds_$TS
LL=$(latest "analysis/run_*_p3seeds_*_llama" | grep -v smoke); GG=$(latest "analysis/run_*_p3seeds_*_gemma2" | grep -v smoke)
[ -n "${LL:-}" ] && [ -n "${GG:-}" ] && step analysis_a13a py a13a_analysis.log code/seeds_analysis.py --llama-run "$LL" --gemma-run "$GG" --tag seeds_A13a_$TS
echo "[$(date +%F' '%T)] night script finished"
