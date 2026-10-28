#!/usr/bin/env bash
# A20: 20 more Llama edit facts (docs/HYPOTHESIS.md A20). Waits for the A19 GPU steps, stops at 06:30; resume = rerun.
cd "$(dirname "$0")"
export PYTHONPATH=code PYTHONIOENCODING=utf8
PY=${PY:-.venv/Scripts/python}; LOG=analysis/_logs/a20.log
until grep -qx hi_run analysis/_logs/a19.done 2>/dev/null; do
  [ "$(date +%H%M)" -ge 0630 ] && [ "$(date +%H%M)" -lt 1200 ] && { echo "[$(date +%F' '%H:%M)] past 06:30 before A19 finished" >> $LOG; exit 0; }
  sleep 30; done
now=$(date +%s); end=$(date -d "$(date +%F) 06:30" +%s); [ $end -le $now ] && { echo "[$(date)] no time left" >> $LOG; exit 0; }
D=$(ls -d analysis/run_*_p4a20_llama 2>/dev/null | tail -1)
echo "[$(date +%F' '%H:%M)] START a20 ${D:+resume $D}" >> $LOG
timeout $((end-now)) $PY code/knowledge_edit_v2.py --model llama --trace-run analysis/run_20261015_p3trace_v2_llama --n-edit 50 --skip-first 30 --layers 21 --add-traced ${D:+--resume $D} --tag p4a20 >> $LOG 2>&1
echo "[$(date +%F' '%H:%M)] a20 exit $?" >> $LOG
