# Execute Phase 3A: Threshold Calibration and Spatial Association

Write-Host "Activating environment..."
# Assuming standard python activation, or just using the specific python path
$PYTHON_PATH = "W:\3 projects\Building\Tfrenzy\ppe-env\Scripts\python.exe"

Write-Host "Running Phase 3A..."
Set-Location "w:\3 projects\Building\Tfrenzy\notebook"
& $PYTHON_PATH "scripts\phase3a_threshold_calibration.py"

Write-Host "Done. Check the outputs directory for CSV reports."
