# Execute V3 Experiment 1 Dataset Setup
Write-Host "Activating environment..."
$PYTHON_PATH = "W:\3 projects\Building\Tfrenzy\ppe-env\Scripts\python.exe"

Set-Location "w:\3 projects\Building\Tfrenzy\notebook"

Write-Host "`n=========================================="
Write-Host "V3 EXPERIMENT 1: DATASET SETUP"
Write-Host "=========================================="
& $PYTHON_PATH "scripts\exp1_dataset_setup.py"

Write-Host "`nDone. Check outputs/v3/ for audit and manifest."
