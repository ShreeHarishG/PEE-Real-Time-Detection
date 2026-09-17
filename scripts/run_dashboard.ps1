<#
.SYNOPSIS
Runs the EdgeVision Dashboard.

.DESCRIPTION
Activates the Python virtual environment and executes the Streamlit dashboard.
#>

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvPath = Join-Path (Split-Path -Parent $ProjectRoot) "ppe-env\Scripts\Activate.ps1"

if (Test-Path $EnvPath) {
    Write-Host "Activating virtual environment..."
    . $EnvPath
} else {
    Write-Warning "Virtual environment not found at $EnvPath. Running with global python."
}

Write-Host "Starting EdgeVision Dashboard..."
Set-Location $ProjectRoot
streamlit run dashboard\dashboard.py
