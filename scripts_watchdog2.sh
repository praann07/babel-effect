#!/usr/bin/env bash
# Restarts scripts_run_queue2.sh (idempotent, skips finished steps) if it dies before writing ALL_DONE.
set -uo pipefail
cd "$(dirname "$0")"
LOG=analysis/_logs/watchdog2.log
alive() { powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -like '*scripts_run_queue2.sh*' -and \$_.CommandLine -notlike '*shell-snapshots*' }) { exit 0 } else { exit 1 }"; }
while true; do
  if grep -q ALL_DONE analysis/_logs/queue2.log 2>/dev/null; then echo "$(date '+%F %T') queue2 ALL_DONE, watchdog exits" >> $LOG; exit 0; fi
  if ! alive; then
    sleep 30; alive && continue
    echo "$(date '+%F %T') queue2 not running; (re)launching" >> $LOG
    (nohup ./scripts_run_queue2.sh >> analysis/_logs/queue2.log 2>&1 &)
    sleep 180
  fi
  sleep 120
done
