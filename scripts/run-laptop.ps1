# Run change-detect on a Windows laptop (secondary path; the HQ Mac is primary).
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run-laptop.ps1 [--simulate]
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Repo = (Get-Location).Path
$Venv = Join-Path $Repo ".venv-laptop"
$Py = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path "laptop.env")) {
    Copy-Item "deploy\laptop.env.example" "laptop.env"
    Write-Host ">> Created laptop.env: set MQTT_PASS (the isrkit password from the Pi)."
}
foreach ($line in Get-Content "laptop.env") {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$') {
        if (-not [Environment]::GetEnvironmentVariable($Matches[1], "Process")) {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim(), "Process")
        }
    }
}
if (-not $env:CHANGE_DETECT_DATA_DIR) { $env:CHANGE_DETECT_DATA_DIR = Join-Path $HOME "change-detect-data" }
if (-not $env:CHANGE_DETECT_PROFILE) { $env:CHANGE_DETECT_PROFILE = "laptop" }

if (-not (Test-Path $Py)) {
    Write-Host ">> Creating $Venv"
    $made = $false
    foreach ($v in "3.12", "3.11", "3.13") {
        & py "-$v" -m venv $Venv 2>$null
        if ($LASTEXITCODE -eq 0) { $made = $true; break }
    }
    if (-not $made) { & python -m venv $Venv }
}
$stamp = Join-Path $Venv ".deps-installed"
if (-not (Test-Path $stamp)) {
    & $Py -m pip install -q --upgrade pip
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Write-Host ">> NVIDIA GPU found: installing CUDA PyTorch"
        & $Py -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
    }
    & $Py -m pip install -q -r requirements-desktop.txt
    New-Item -ItemType File $stamp | Out-Null
}
$accel = & $Py -c "from changedetect.labeler import detect_accelerator as d; print(d())"
Write-Host ">> Accelerator: $accel | TAK $($env:TAK_HOST):$($env:TAK_PORT) | MQTT $($env:MQTT_USER)@$($env:MQTT_HOST)"
Write-Host ">> Web UI: http://localhost:$(if ($env:WEB_PORT) { $env:WEB_PORT } else { 8130 })/"
& $Py -m changedetect --profile $env:CHANGE_DETECT_PROFILE @args
