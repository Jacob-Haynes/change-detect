# Run change-detect on a Windows laptop (secondary path; the Mac stations are primary).
# Reads configuration from the ENVIRONMENT only. Foreground until Ctrl+C.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run-laptop.ps1 [--simulate]
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Repo = (Get-Location).Path
$Venv = Join-Path $Repo ".venv-laptop"
$Py = Join-Path $Venv "Scripts\python.exe"

$defaults = @{ TAK_HOST = "isr-pi.local"; TAK_PORT = "8087"; MQTT_HOST = "isr-pi.local"; MQTT_PORT = "1883";
               MQTT_USER = "isrkit"; WEB_PORT = "8130"; VLM_URL = "http://localhost:11434"; VLM_MODEL = "qwen3-vl:8b-instruct";
               CHANGE_DETECT_PROFILE = "laptop"; CHANGE_DETECT_DATA_DIR = (Join-Path $HOME "change-detect-data") }
foreach ($k in $defaults.Keys) {
    if (-not [Environment]::GetEnvironmentVariable($k, "Process")) { [Environment]::SetEnvironmentVariable($k, $defaults[$k], "Process") }
}

if (-not (Test-Path $Py)) {
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
        Write-Host "[change-detect] NVIDIA GPU found: installing CUDA PyTorch"
        & $Py -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
    }
    & $Py -m pip install -q -r requirements-desktop.txt
    New-Item -ItemType File $stamp | Out-Null
}
if (-not $env:MQTT_PASS) { Write-Host "[change-detect] WARNING: MQTT_PASS is empty" }
& $Py -m changedetect --host 0.0.0.0 --port $env:WEB_PORT --profile $env:CHANGE_DETECT_PROFILE @args
