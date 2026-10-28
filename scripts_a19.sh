#!/usr/bin/env bash
# A19: Llama-3.2-3B base (docs/HYPOTHESIS.md A19). Resumable via done markers; run dirs are write-once so each step runs once.
cd "$(dirname "$0")"
export PYTHONPATH=code PYTHONIOENCODING=utf8
PY=${PY:-.venv/Scripts/python}
LOG=analysis/_logs/a19.log; DONE=analysis/_logs/a19.done; mkdir -p analysis/_logs; touch $DONE
step() { local n=$1; shift; grep -qx "$n" $DONE && return 0; echo "[$(date +%F' '%H:%M)] START $n" >> $LOG
  if "$@" >> $LOG 2>&1; then echo "$n" >> $DONE; echo "[$(date +%F' '%H:%M)] OK $n" >> $LOG; else echo "[$(date +%F' '%H:%M)] FAIL $n" >> $LOG; exit 1; fi; }
# GPU steps: 10 min rest after each one so the laptop cools; no new GPU step starts after 06:30 (window ends 07:30). Re-run the script to resume.
gstep() { local h=$(date +%H%M); if [ "$h" -ge 0630 ] && [ "$h" -lt 1200 ]; then echo "[$(date +%F' '%H:%M)] past 06:30, not starting $1" >> $LOG; exit 0; fi
  grep -qx "$1" $DONE && return 0; step "$@"; echo "[$(date +%F' '%H:%M)] resting 10 min" >> $LOG; sleep 600; }
M=llama_base
gstep filter_es $PY code/e2_filter.py --model $M --tag e2bfilter
gstep filter_hi $PY code/e2_filter_hi.py --model $M --a16 --tag e2hbfilter
gstep es_run $PY code/interchange.py --model $M --langs en,es --k-src 2 --shuffle-pairs --max-pairs 200 --tag e2b_A
gstep hi_run $PY code/interchange.py --model $M --langs en,hi --facts data/v2/e2h2_facts_$M.json --span end --directions en2hi,hi2en --k-src 4 --shuffle-pairs --max-pairs 200 --tag e2hb_A
ES=$(ls -d analysis/run_*_e2b_A_$M | tail -1); HI=$(ls -d analysis/run_*_e2hb_A_$M | tail -1)
step an_es $PY code/e2_analysis.py --models $M --llama-label $M --langs en,es --llama-run "$ES" --tag e2b_analysis_A19
step an_es_lf $PY code/e2_analysis.py --models $M --llama-label $M --langs en,es --exclude-leak --llama-run "$ES" --tag e2b_analysis_A19_leakfree
step an_hi $PY code/e2_analysis.py --models $M --llama-label $M --langs en,hi --sites A --llama-run "$HI" --tag e2hb_analysis_A19
step an_hi_lf $PY code/e2_analysis.py --models $M --llama-label $M --langs en,hi --sites A --exclude-leak --llama-run "$HI" --tag e2hb_analysis_A19_leakfree
step cluster $PY code/cluster_sensitivity.py --set a19b
echo "[$(date +%F' '%H:%M)] a19 finished" >> $LOG
