#!/usr/bin/env bash
# Kaggle version of scripts_a24.sh (shared-key edit). Needs the paraphrase-check run directory (run1). Stops cleanly after HOURS (default 10.5).
cd "$(dirname "$0")/.."
export PYTHONPATH=code PYTHONIOENCODING=utf8
PY=${PY:-python}; LOG=analysis/_logs/run3_shared_edit.log; HOURS=${HOURS:-10.5}
mkdir -p analysis/_logs
BASE=$(ls -d analysis/run_*_p4a21_llama 2>/dev/null | tail -1)
[ -z "$BASE" ] && { echo "run3 needs the run1 directory (analysis/run_*_p4a21_llama) first" | tee -a $LOG; exit 1; }
D=$(ls -d analysis/run_*_p4a24_llama 2>/dev/null | tail -1)
echo "[$(date +%F' '%H:%M)] START run3 ${D:+resume $D} base $BASE" >> $LOG
timeout $(python -c "print(int($HOURS*3600))") $PY code/knowledge_edit_v2.py --model llama --trace-run analysis/run_20261015_p3trace_v2_llama --n-edit 40 --skip-first 10 --layers 14 --add-traced --para --key-modes mean_all,joint_all ${D:+--resume $D} --tag p4a24 >> $LOG 2>&1
echo "[$(date +%F' '%H:%M)] run3 exit $?" >> $LOG
D=$(ls -d analysis/run_*_p4a24_llama | tail -1)
$PY code/keymode_analysis.py "$D" --base "$BASE" >> $LOG 2>&1
