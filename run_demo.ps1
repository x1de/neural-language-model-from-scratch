$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $project '.venv\Scripts\python.exe'
$bundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (Test-Path $venv) {
    $pythonExe = $venv
} elseif (Test-Path $bundled) {
    $pythonExe = $bundled
} else {
    $pythonExe = 'python'
}
Set-Location $project
if ($args.Count -eq 0) {
    & $pythonExe (Join-Path $project 'tui.py')
} else {
    & $pythonExe (Join-Path $project 'demo.py') @args
}
