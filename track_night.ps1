# Live tracker for the A13a night run (read-only). Refreshes every 10 s. Close the window to stop watching; the run keeps going.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$logs = Join-Path $root "analysis\_logs"
$steps = @(
  @{n="e2_pilot_gemma2"; d="Gemma 10-pair debug pilot of the swap test (excluded from analysis)"; log="e2_pilot_gemma2.log"},
  @{n="e2_full_llama"; d="Llama: swap English subject representations into Spanish prompts (and back), all layers"; log="e2_full_llama.log"; total=1},
  @{n="e2_full_gemma2"; d="Gemma: same swap test, all layers"; log="e2_full_gemma2.log"}
)
while ($true) {
  Clear-Host
  $done = @(); $f = Join-Path $logs "night_e2.done"; if (Test-Path $f) { $done = Get-Content $f }
  Write-Host "=== E2 SWAP TEST  ($(Get-Date -Format 'ddd HH:mm:ss')) ===" -ForegroundColor Cyan
  Write-Host "Question: can an English subject representation stand in for the Spanish one (and back)?`n"
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
  $newest = Get-ChildItem $logs -Filter "e2_*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($newest) { Write-Host ("(" + $newest.Name + ")"); Get-Content $newest.FullName -Tail 6 }
  else { Write-Host "No log yet. The run has not started." }
  Write-Host "`nWhen both DONE, tell Claude; results are in analysis\run_*_e2full_*"
  Start-Sleep -Seconds 10
}
