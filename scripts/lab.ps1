# Run from any directory. Pass CLI arguments unchanged after this script.
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskRuntimePath = Join-Path $taskRoot 'runtime.local.json'
$taskPython = if ($env:ALAB_PYTHON) {
    $env:ALAB_PYTHON
} elseif (Test-Path -LiteralPath $taskRuntimePath) {
    (Get-Content -LiteralPath $taskRuntimePath -Raw -Encoding UTF8 | ConvertFrom-Json).python
} else {
    Join-Path $taskRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Python environment not found. Run scripts/setup.ps1 first, or set ALAB_PYTHON.'
}
$taskOldPythonPath = $env:PYTHONPATH
$taskCliArgs = foreach ($taskArgument in $args) {
    # PowerShell parses unquoted comma lists as arrays. Keep each such list one CLI value.
    if ($taskArgument -is [Array]) { $taskArgument -join ',' } else { $taskArgument }
}
try {
    $env:PYTHONPATH = (Join-Path $taskRoot 'src') + $(if ($taskOldPythonPath) { [IO.Path]::PathSeparator + $taskOldPythonPath } else { '' })
    & $taskPython -m autolabel_lab @taskCliArgs
    $taskExitCode = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $taskOldPythonPath
}
exit $taskExitCode
