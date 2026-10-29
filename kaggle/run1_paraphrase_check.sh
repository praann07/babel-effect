#!/usr/bin/env bash
# Kaggle version of scripts_a21.sh (paraphrase check). No 06:30 stop; stops cleanly after HOURS (default 10.5) so the session can save.
cd "$(dirname "$0")/.."
export PYTHONPATH=code PYTHONIOENCODING=utf8
PY=${PY:-python}; LOG=analysis/_logs/run1_paraphrase.log; HOURS=${HOURS:-10.5}
mkdir -p analysis/_logs
D=$(ls -d analysis/run_*_p4a21_llama 2>/dev/null | tail -1)
echo "[$(date +%F' '%H:%M)] START run1 ${D:+resume $D}" >> $LOG
timeout $(python -c "print(int($HOURS*3600))") $PY code/knowledge_edit_v2.py --model llama --trace-run analysis/run_20261015_p3trace_v2_llama --n-edit 40 --skip-first 10 --layers 14 --add-traced --para ${D:+--resume $D} --tag p4a21 >> $LOG 2>&1
echo "[$(date +%F' '%H:%M)] run1 exit $?" >> $LOG
D=$(ls -d analysis/run_*_p4a21_llama | tail -1)
$PY code/para_analysis.py "$D" --ref analysis/run_20261020_p4f7_llama --ref "$(ls -d analysis/run_*_p4a20_llama | tail -1)" >> $LOG 2>&1
