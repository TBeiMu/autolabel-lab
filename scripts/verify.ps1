$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
& $taskPython -m pytest (Join-Path $taskRoot 'tests') -q
exit $LASTEXITCODE
