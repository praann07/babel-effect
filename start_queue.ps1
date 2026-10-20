# Start (or resume) the experiment queue + watchdog as hidden, independent processes. Finished steps are skipped.
$root = "C:\Users\Som Praneeth\Desktop\reasearch\babel-effect"
$bash = "C:\Program Files\Git\bin\bash.exe"
function Running($pat) { [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $pat -and $_.Name -match 'bash' }) }
if (Running 'scripts_run_queue2') { "Queue is already running - nothing to do." }
else {
    Start-Process -WindowStyle Hidden -FilePath $bash -WorkingDirectory $root -ArgumentList '-lc', 'cd "/c/Users/Som Praneeth/Desktop/reasearch/babel-effect" && ./scripts_run_queue2.sh >> analysis/_logs/queue2.log 2>&1'
    "Queue started."
}
if (Running 'scripts_watchdog2') { "Watchdog is already running." }
else {
    Start-Process -WindowStyle Hidden -FilePath $bash -WorkingDirectory $root -ArgumentList '-lc', 'cd "/c/Users/Som Praneeth/Desktop/reasearch/babel-effect" && ./scripts_watchdog2.sh > analysis/_logs/watchdog2_stdout.log 2>&1'
    "Watchdog started."
}
Start-Sleep 4
"Check: " + $(if (Running 'scripts_run_queue2') { "queue process is alive." } else { "queue process NOT found - tell Claude." })
