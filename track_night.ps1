# Live tracker for the A13a night run (read-only). Refreshes every 10 s. Close the window to stop watching; the run keeps going.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$logs = Join-Path $root "analysis\_logs"
$steps = @(
  @{n="smoke_llama";  d="2-fact test run of the new tracer (checks it reproduces the old numbers)"; log="a13a_smoke_llama.log"},
  @{n="trace_llama";  d="Llama: re-trace 126 facts x EN/ES with fresh noise (16 draws each)";      log="a13a_llama.log"; total=126},
  @{n="trace_gemma2"; d="Gemma: re-trace 59 facts x EN/ES with fresh noise (16 draws each)";       log="a13a_gemma2.log"; total=59},
  @{n="analysis_a13a";d="CPU analysis: does the alignment survive independent noise?";             log="a13a_analysis.log"}
)
while ($true) {
  Clear-Host
  $done = @(); $f = Join-Path $logs "night_a13a.done"; if (Test-Path $f) { $done = Get-Content $f }
  Write-Host "=== A13a NIGHT RUN  ($(Get-Date -Format 'ddd HH:mm:ss')) ===" -ForegroundColor Cyan
  Write-Host "Question: does the English-Spanish match survive when each language gets its OWN random noise?`n"
  $running = $false
  foreach ($s in $steps) {
    $lp = Join-Path $logs $s.log; $state = "waiting"; $extra = ""
    if ($done -contains $s.n) { $state = "DONE" }
    elseif (Test-Path $lp) {
      $age = ((Get-Date) - (Get-Item $lp).LastWriteTime).TotalMinutes
      $state = if ($age -lt 5) { "RUNNING" } else { "stalled/failed?" }
      $last = (Get-Content $lp -Tail 400 | Select-String -Pattern '^\d+ / \d+' | Select-Object -Last 1)
      if ($last) { $extra = " progress " + $last.Line.Split(" ")[0..2] -join " " }
      if ($state -eq "RUNNING") { $running = $true }
    }
    $col = switch ($state) { "DONE" {"Green"} "RUNNING" {"Yellow"} default {"Gray"} }
    Write-Host ("[{0,-15}] {1}" -f $state, $s.n) -ForegroundColor $col
    Write-Host ("                  " + $s.d + $extra)
  }
  Write-Host "`n--- GPU ---"
  try { & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader } catch { Write-Host "nvidia-smi not available" }
  Write-Host "`n--- Latest log lines ---"
  $newest = Get-ChildItem $logs -Filter "a13a_*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($newest) { Write-Host ("(" + $newest.Name + ")"); Get-Content $newest.FullName -Tail 6 }
  else { Write-Host "No log yet. The run has not started." }
  Write-Host "`nWhen it says DONE for analysis_a13a, results are in analysis\run_*_seeds_A13a_*\results.json"
  Start-Sleep -Seconds 10
}
