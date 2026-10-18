# Holds the system awake (no settings changed) until this process is stopped.
Add-Type -Namespace W -Name K -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.K]::SetThreadExecutionState(0x80000001) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
while ($true) { Start-Sleep -Seconds 60; [W.K]::SetThreadExecutionState(0x80000001) | Out-Null }
