#!/usr/bin/env bash
# Kaggle version of scripts_a23.sh (new fact types). No 06:30 stop; resumable via done markers.
cd "$(dirname "$0")/.."
export PYTHONPATH=code PYTHONIOENCODING=utf8
PY=${PY:-python}
LOG=analysis/_logs/run2_new_facts.log; DONE=analysis/_logs/run2_new_facts.done; mkdir -p analysis/_logs; touch $DONE
step() { local n=$1; shift; grep -qx "$n" $DONE && return 0; echo "[$(date +%F' '%H:%M)] START $n" >> $LOG
  if "$@" >> $LOG 2>&1; then echo "$n" >> $DONE; echo "[$(date +%F' '%H:%M)] OK $n" >> $LOG; else echo "[$(date +%F' '%H:%M)] FAIL $n" >> $LOG; exit 1; fi; }
M=llama
step filter_es $PY code/e2_filter.py --model $M --set a23 --tag e2a23filter
step filter_hi $PY code/e2_filter_hi.py --model $M --set a23 --tag e2ha23filter
step es_run $PY code/interchange.py --model $M --langs en,es --facts data/v2/e2a23_facts_$M.json --k-src 2 --shuffle-pairs --max-pairs 200 --tag e2a23_A
step hi_run $PY code/interchange.py --model $M --langs en,hi --facts data/v2/e2ha23_facts_$M.json --span end --directions en2hi,hi2en --k-src 4 --shuffle-pairs --max-pairs 200 --tag e2ha23_A
ES=$(ls -d analysis/run_*_e2a23_A_$M | tail -1); HI=$(ls -d analysis/run_*_e2ha23_A_$M 2>/dev/null | tail -1)
step an_es $PY code/e2_analysis.py --models $M --langs en,es --llama-run "$ES" --tag e2a23_analysis
step an_es_lf $PY code/e2_analysis.py --models $M --langs en,es --exclude-leak --llama-run "$ES" --tag e2a23_analysis_leakfree
[ -n "$HI" ] && step an_hi $PY code/e2_analysis.py --models $M --langs en,hi --sites A --llama-run "$HI" --tag e2ha23_analysis
[ -n "$HI" ] && step an_hi_lf $PY code/e2_analysis.py --models $M --langs en,hi --sites A --exclude-leak --llama-run "$HI" --tag e2ha23_analysis_leakfree
echo "[$(date +%F' '%H:%M)] run2 finished" >> $LOG
