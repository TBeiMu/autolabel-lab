param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskUrl = "http://127.0.0.1:$Port"
$taskRuntime = Join-Path $taskRoot 'runtime.local.json'
$taskPython = if ($env:ALAB_PYTHON) { $env:ALAB_PYTHON } elseif (Test-Path -LiteralPath $taskRuntime) {
    (Get-Content -LiteralPath $taskRuntime -Raw -Encoding UTF8 | ConvertFrom-Json).python
} else { Join-Path $taskRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run scripts/setup.ps1 -Training or set ALAB_PYTHON first.' }
try {
    $taskExisting = Invoke-RestMethod -Uri "$taskUrl/api/bootstrap" -TimeoutSec 2
    if ($taskExisting.app -eq 'autolabel-lab' -and $taskExisting.workspace -eq $taskRoot) {
        Start-Process $taskUrl
        Write-Output "Opened existing workbench: $taskUrl"
        exit 0
    }
    throw "Port $Port belongs to another service. Use -Port with a free port."
} catch {
    if ($_.Exception.Message -like 'Port *belongs*') { throw }
}
$taskLogs = Join-Path $taskRoot '.gui'
New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
$taskOldPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $taskRoot 'src'
    $taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-u', '-m', 'autolabel_lab.web', '--workspace', ('"' + $taskRoot + '"'), '--port', "$Port", '--no-browser') -WindowStyle Hidden -PassThru -WorkingDirectory $taskRoot -RedirectStandardOutput (Join-Path $taskLogs 'server.log') -RedirectStandardError (Join-Path $taskLogs 'server-error.log')
} finally {
    $env:PYTHONPATH = $taskOldPythonPath
}
for ($taskAttempt=0; $taskAttempt -lt 30; $taskAttempt++) {
    try {
        $taskReady = Invoke-RestMethod -Uri "$taskUrl/api/bootstrap" -TimeoutSec 1
        if ($taskReady.app -eq 'autolabel-lab') {
            $taskProcess.Id | Set-Content -LiteralPath (Join-Path $taskLogs 'server.pid')
            Start-Process $taskUrl
            Write-Output "Workbench ready: $taskUrl"
            exit 0
        }
    } catch { }
    if ($taskProcess.HasExited) { throw "Web server exited. See $taskLogs\server-error.log" }
    Start-Sleep -Milliseconds 300
}
throw "Web server did not respond. See $taskLogs\server-error.log"
