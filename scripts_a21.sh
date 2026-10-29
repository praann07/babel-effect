#!/usr/bin/env bash
# A21: English-paraphrase control for edit transfer (docs/HYPOTHESIS.md A21). Stops at 06:30; rerun to resume.
cd "$(dirname "$0")"
export PYTHONPATH=code PYTHONIOENCODING=utf8
PY=${PY:-.venv/Scripts/python}; LOG=analysis/_logs/a21.log
now=$(date +%s); end=$(date -d "$(date +%F) 06:30" +%s); [ $end -le $now ] && end=$(date -d "tomorrow 06:30" +%s)
D=$(ls -d analysis/run_*_p4a21_llama 2>/dev/null | tail -1)
echo "[$(date +%F' '%H:%M)] START a21 ${D:+resume $D}" >> $LOG
timeout $((end-now)) $PY code/knowledge_edit_v2.py --model llama --trace-run analysis/run_20261015_p3trace_v2_llama --n-edit 40 --skip-first 10 --layers 14 --add-traced --para ${D:+--resume $D} --tag p4a21 >> $LOG 2>&1
echo "[$(date +%F' '%H:%M)] a21 exit $?" >> $LOG
D=$(ls -d analysis/run_*_p4a21_llama | tail -1)
$PY code/para_analysis.py "$D" --ref analysis/run_20261020_p4f7_llama --ref "$(ls -d analysis/run_*_p4a20_llama | tail -1)" >> $LOG 2>&1
