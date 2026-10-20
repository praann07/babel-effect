# Stop the watchdog, the queue and any running experiment. Finished steps stay saved; the step in progress restarts from zero later.
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'scripts_watchdog2' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'scripts_run_queue2' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'code[\\/]' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep 4
$left = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'scripts_run_queue2|scripts_watchdog2|code[\\/]\w+\.py' }
if ($left) { "Some processes are still alive - run this again." } else { "Everything stopped. GPU is free." }
