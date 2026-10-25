#!/usr/bin/env bash
# After the E2 run: Gemma F7 (queue2), Llama-3.2-1B pilot (F6), BLOOM-1b7 pilot (A10). No new step starts after 06:00; hard stop 07:40.
cd "$(dirname "$0")"; LOG=analysis/_logs/after_e2.log
until grep -q "e2 script finished" analysis/_logs/night_e2.log; do sleep 60; done
for s in scripts_run_queue2.sh scripts_run_llama1b.sh scripts_run_bloom.sh; do
  h=$(date +%H); [ "$h" -ge 6 ] && [ "$h" -lt 12 ] && { echo "[$(date +%T)] skip $s (too late)" >> $LOG; continue; }
  echo "[$(date +%F' '%T)] START $s" >> $LOG; bash ./$s >> analysis/_logs/after_$s.log 2>&1; echo "[$(date +%F' '%T)] END $s rc=$?" >> $LOG
done
echo "[$(date +%F' '%T)] after_e2 finished" >> $LOG
