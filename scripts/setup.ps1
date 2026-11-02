param([switch]$Training, [string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskVenv = Join-Path $taskRoot '.venv'
if (-not (Test-Path -LiteralPath (Join-Path $taskVenv 'Scripts\python.exe'))) {
    & $Python -m venv $taskVenv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment.' }
}
$taskPython = Join-Path $taskVenv 'Scripts\python.exe'
$taskPackage = if ($Training) { $taskRoot + '[train,test]' } else { $taskRoot + '[test]' }
& $taskPython -m pip install --disable-pip-version-check -e $taskPackage
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Output "Ready: $taskPython"
