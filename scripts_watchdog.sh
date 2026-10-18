#!/usr/bin/env bash
# Restarts the layer-sweep chain (resuming from saved rows) if it dies before writing ALL_DONE, e.g. after the laptop sleeps.
# The later chains (extra, incontext, llama1b) only wait on logs and are not restarted here.
set -uo pipefail
cd "$(dirname "$0")"
LOG=analysis/_logs/watchdog.log
alive() {  # is a bash process running ./scripts_run_sweeps.sh ?
  powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -like '*scripts_run_sweeps.sh*' }) { exit 0 } else { exit 1 }"
}
while true; do
  if grep -q ALL_DONE analysis/_logs/sweeps_chain.log 2>/dev/null; then echo "$(date '+%F %T') sweeps ALL_DONE, watchdog exits" >> $LOG; exit 0; fi
  if ! alive; then
    sleep 20; alive && continue   # avoid racing a chain that is starting
    export RESUME_LLAMA=$(ls -d analysis/run_*_p4sweep_llama | head -1)
    G=$(ls -d analysis/run_*_p4sweep_gemma2 2>/dev/null | head -1 || true)
    if [ -n "${G:-}" ]; then export RESUME_GEMMA="$G"; fi
    echo "$(date '+%F %T') sweep chain not running; relaunching (llama=$RESUME_LLAMA gemma=${RESUME_GEMMA:-new})" >> $LOG
    cp analysis/_logs/sweeps_chain.log "analysis/_logs/sweeps_chain_prev_$(date +%H%M%S).log" 2>/dev/null
    (nohup ./scripts_run_sweeps.sh > analysis/_logs/sweeps_chain.log 2>&1 &)
    sleep 120
  fi
  sleep 120
done
