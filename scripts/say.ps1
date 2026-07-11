# say.ps1 — speak text through VoiceForge from any script or agent.
#   say.cmd "hello there"
#   say.cmd -Voice my-voice -NoWait "build finished"
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]]$Text,
    [string]$Voice = "",
    [switch]$NoWait
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Port = 8484
$CfgPath = Join-Path $Root "config.json"
if (Test-Path $CfgPath) {
    try { $cfg = Get-Content $CfgPath -Raw | ConvertFrom-Json; if ($cfg.port) { $Port = $cfg.port } } catch {}
}
$Body = @{ text = ($Text -join " "); wait = (-not $NoWait) }
if ($Voice) { $Body.voice = $Voice }
try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/speak" -Method Post `
        -ContentType "application/json; charset=utf-8" `
        -Body ([System.Text.Encoding]::UTF8.GetBytes(($Body | ConvertTo-Json))) `
        -TimeoutSec 600
    Write-Host "spoken: $($r.file)"
} catch {
    if ($_.Exception -is [System.Net.WebException] -and $null -eq $_.Exception.Response) {
        Write-Host "VoiceForge server is not running - start it with VoiceForge.bat" -ForegroundColor Red
        exit 2
    }
    Write-Host "speak failed: $($_.ErrorDetails.Message)" -ForegroundColor Red
    exit 1
}
