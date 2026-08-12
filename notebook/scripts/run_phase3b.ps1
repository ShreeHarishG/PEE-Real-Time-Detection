# Execute Phase 3B: Threshold Grid Sweep

Write-Host "Activating environment..."
$PYTHON_PATH = "W:\3 projects\Building\Tfrenzy\ppe-env\Scripts\python.exe"

Write-Host "Running Phase 3B..."
Set-Location "w:\3 projects\Building\Tfrenzy\notebook"
& $PYTHON_PATH "scripts\phase3b_threshold_grid.py"

Write-Host "Done. Check the outputs directory for phase3b_threshold_grid.csv."
