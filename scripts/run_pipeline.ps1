<#
.SYNOPSIS
Runs the EdgeVision Inference Pipeline.

.DESCRIPTION
Activates the Python virtual environment and executes the main pipeline script.
#>

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvPath = Join-Path (Split-Path -Parent $ProjectRoot) "ppe-env\Scripts\Activate.ps1"

if (Test-Path $EnvPath) {
    Write-Host "Activating virtual environment..."
    . $EnvPath
} else {
    Write-Warning "Virtual environment not found at $EnvPath. Running with global python."
}

Write-Host "Starting EdgeVision Pipeline..."
python "$ProjectRoot\src\pipeline.py"
