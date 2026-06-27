# Native Windows fix for: "Application Control policy has blocked" numpy/torch DLLs
# Run this script as Administrator AFTER turning Smart App Control OFF (see below).

$ErrorActionPreference = "Stop"
$BackendRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$VenvPath = Join-Path $BackendRoot ".venv"
$NumpyCore = Join-Path $VenvPath "Lib\site-packages\numpy\_core"

Write-Host ""
Write-Host "ChatMemory — Windows ML stack fix (no WSL)" -ForegroundColor Cyan
Write-Host "Backend: $BackendRoot"
Write-Host ""

# Smart App Control blocks .pyd DLLs; Defender folder exclusions do NOT bypass it.
try {
    $sac = (Get-MpComputerStatus).SmartAppControlState
    Write-Host "Smart App Control: $sac"
    if ($sac -eq "On") {
        Write-Host ""
        Write-Host "ACTION REQUIRED (manual):" -ForegroundColor Yellow
        Write-Host "  1. Open Windows Security"
        Write-Host "  2. App and browser control -> Smart App Control"
        Write-Host "  3. Set to OFF (may require restart)"
        Write-Host ""
        Write-Host "Folder exclusions alone will NOT fix this while SAC is On." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Could not read Smart App Control state (continue if SAC is already off)."
}

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)

if ($isAdmin) {
    Write-Host "Adding Defender exclusions..." -ForegroundColor Green
    Add-MpPreference -ExclusionPath $BackendRoot -ErrorAction SilentlyContinue
    Add-MpPreference -ExclusionPath $VenvPath -ErrorAction SilentlyContinue
    if (Test-Path $NumpyCore) {
        Add-MpPreference -ExclusionPath $NumpyCore -ErrorAction SilentlyContinue
    }
    Write-Host "Exclusions added."
} else {
    Write-Host "Tip: Re-run this script as Administrator to add Defender exclusions." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Verifying imports..." -ForegroundColor Cyan
Push-Location $BackendRoot
try {
    Write-Host "Installing CUDA PyTorch (cu124) for NVIDIA GPU..." -ForegroundColor Cyan
    & uv pip uninstall torch 2>$null
    & uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA not available — check NVIDIA driver or Smart App Control'"
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Success. Restart the backend and try Build persona again." -ForegroundColor Green
    } else {
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
