#!/usr/bin/env bash
# Kills the GPU jobs at 07:40 so the laptop is free at 08:00.
until [ "$(date +%H%M)" -ge 0740 ] && [ "$(date +%H%M)" -lt 1200 ]; do sleep 60; done
powershell -NoProfile -Command "Get-CimInstance Win32_Process | ? { \$_.CommandLine -match 'babel-effect' -and \$_.CommandLine -match 'scripts_|interchange|causal_trace|knowledge_edit|circuit_overlap|code/' -and \$_.ProcessId -ne \$PID } | % { Stop-Process -Id \$_.ProcessId -Force }"
echo "[$(date +%F' '%T)] HARD STOP" >> analysis/_logs/after_e2.log
