# VoiceForge launcher — creates the server venv on first run, starts the API+UI.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[VoiceForge] uv not found - installing it now..." -ForegroundColor Cyan
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id astral-sh.uv -e --accept-package-agreements --accept-source-agreements
    } else {
        try { irm https://astral.sh/uv/install.ps1 | iex } catch {}
    }
    # fresh installs don't refresh PATH in this session
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    foreach ($dir in @((Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links"), (Join-Path $env:USERPROFILE ".local\bin"))) {
        if ((Test-Path (Join-Path $dir "uv.exe")) -and ($env:Path -notlike "*$dir*")) {
            $env:Path = "$dir;$env:Path"
        }
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "Could not install uv automatically. Install it with:  winget install astral-sh.uv" -ForegroundColor Red
        exit 1
    }
    Write-Host "[VoiceForge] uv installed" -ForegroundColor Green
}

$ServerPy = Join-Path $Root "envs\server\Scripts\python.exe"
if (-not (Test-Path $ServerPy)) {
    Write-Host "[VoiceForge] first run - creating server environment..." -ForegroundColor Cyan
    uv venv (Join-Path $Root "envs\server") --python 3.12
    uv pip install -r (Join-Path $Root "server\requirements.txt") --python $ServerPy
}

$Port = 8484
$CfgPath = Join-Path $Root "config.json"
if (Test-Path $CfgPath) {
    try { $cfg = Get-Content $CfgPath -Raw | ConvertFrom-Json; if ($cfg.port) { $Port = $cfg.port } } catch {}
}

Write-Host "[VoiceForge] starting on http://127.0.0.1:$Port" -ForegroundColor Green
Start-Job -ScriptBlock {
    param($u) Start-Sleep -Seconds 2; Start-Process $u
} -ArgumentList "http://127.0.0.1:$Port" | Out-Null

& $ServerPy -m uvicorn app:app --host 127.0.0.1 --port $Port --app-dir (Join-Path $Root "server")
