#!/usr/bin/env bash
# A24: shared-key edit (docs/HYPOTHESIS.md A24). Run AFTER A21 (needs its run dir as the English-key baseline). Stops at 06:30; rerun to resume.
cd "$(dirname "$0")"
export PYTHONPATH=code PYTHONIOENCODING=utf8
PY=${PY:-.venv/Scripts/python}; LOG=analysis/_logs/a24.log
BASE=$(ls -d analysis/run_*_p4a21_llama 2>/dev/null | tail -1)
[ -z "$BASE" ] && { echo "A24 needs the A21 run (analysis/run_*_p4a21_llama) first" | tee -a $LOG; exit 1; }
now=$(date +%s); end=$(date -d "$(date +%F) 06:30" +%s); [ $end -le $now ] && end=$(date -d "tomorrow 06:30" +%s)
D=$(ls -d analysis/run_*_p4a24_llama 2>/dev/null | tail -1)
echo "[$(date +%F' '%H:%M)] START a24 ${D:+resume $D} base $BASE" >> $LOG
timeout $((end-now)) $PY code/knowledge_edit_v2.py --model llama --trace-run analysis/run_20261015_p3trace_v2_llama --n-edit 40 --skip-first 10 --layers 14 --add-traced --para --key-modes mean_all,joint_all ${D:+--resume $D} --tag p4a24 >> $LOG 2>&1
echo "[$(date +%F' '%H:%M)] a24 exit $?" >> $LOG
D=$(ls -d analysis/run_*_p4a24_llama | tail -1)
$PY code/keymode_analysis.py "$D" --base "$BASE" >> $LOG 2>&1
