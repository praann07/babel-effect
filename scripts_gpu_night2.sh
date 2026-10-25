#!/usr/bin/env bash
# 2026-10-25 afternoon/night: pre-registered pending GPU steps (Gemma F7, Llama-3.2-1B pilot, BLOOM-1b7 pilot). Hard stop at 07:40 (scripts_hardstop.sh).
cd "$(dirname "$0")"; LOG=analysis/_logs/after_e2.log
for s in scripts_run_queue2.sh scripts_run_llama1b.sh scripts_run_bloom.sh; do
  echo "[$(date +%F' '%T)] START $s" >> $LOG; bash ./$s >> analysis/_logs/after_$s.log 2>&1; echo "[$(date +%F' '%T)] END $s rc=$?" >> $LOG
done
echo "[$(date +%F' '%T)] gpu_night2 finished" >> $LOG
