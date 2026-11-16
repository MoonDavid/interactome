param(
    [int]$Port = 8000,
    [switch]$OpenBrowser
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$normaRoot = Join-Path $projectRoot 'tools/NORMA'
$serverUrl = "http://localhost:$Port"
function Get-NormaHealth {
    try {
        $body = & curl.exe --silent --noproxy '*' --max-time 2 "http://127.0.0.1:$Port/api/health"
        if ($LASTEXITCODE -eq 0 -and $body) { return ($body | ConvertFrom-Json) }
    } catch { }
    return $null
}
if (-not (Test-Path -LiteralPath (Join-Path $normaRoot 'frontend/dist/norma.html'))) {
    throw 'NORMA is not built. In tools/NORMA/frontend run npm.cmd ci and npm.cmd run build.'
}
try {
    $health = Get-NormaHealth
} catch { $health = $null }
if ($health -and $health.status -eq 'ok' -and $health.api) {
    Write-Output "NORMA is already running: $serverUrl"
    if ($OpenBrowser) { Start-Process $serverUrl }
    exit 0
}
# The project's venv launcher may reference a Python no longer installed on Windows.
# Prefer its bundled standalone interpreter, then check ordinary Python launchers.
$candidates = @()
$tooling = Join-Path $projectRoot '.venv/.tooling/python'
if (Test-Path -LiteralPath $tooling) {
    $candidates += Get-ChildItem -LiteralPath $tooling -Directory |
        ForEach-Object { Join-Path $_.FullName 'python.exe' }
}
$candidates += Join-Path $projectRoot '.venv/Scripts/python.exe'
$command = Get-Command python.exe -ErrorAction SilentlyContinue
if ($command) { $candidates += $command.Source }
$pythonExe = $null
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate) {
        & $candidate --version 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $pythonExe = $candidate; break }
    }
}
if (-not $pythonExe) { throw 'No working Python interpreter found. NORMA needs Python 3.9+.' }
$logRoot = Join-Path $projectRoot 'data/analysis_runs/norma_server'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logRoot "$stamp.stdout.log"
$stderr = Join-Path $logRoot "$stamp.stderr.log"
$process = Start-Process -FilePath $pythonExe -ArgumentList @(
    '-u', 'backend/server.py', '--mode', 'local', '--host', '127.0.0.1', '--port', "$Port", '--no-open'
) -WorkingDirectory $normaRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
@{ pid=$process.Id; url=$serverUrl; python=$pythonExe; stdout=$stdout; stderr=$stderr;
   started=(Get-Date -Format o) } | ConvertTo-Json | Set-Content (Join-Path $logRoot 'job.json')
for ($attempt=0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Milliseconds 300
    try {
        $health = Get-NormaHealth
        if ($health.status -eq 'ok' -and $health.api) {
            Write-Output "NORMA ready: $serverUrl (PID $($process.Id))"
            Write-Output "Server logs: $logRoot"
            if ($OpenBrowser) { Start-Process $serverUrl }
            exit 0
        }
    } catch { }
    $process.Refresh()
    if ($process.HasExited) { break }
}
if (Test-Path -LiteralPath $stderr) { Get-Content -LiteralPath $stderr -Tail 15 }
throw "NORMA did not start. See $logRoot"
