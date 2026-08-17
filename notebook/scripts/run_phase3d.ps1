# Execute Phase 3D: Final Clean Validation

Write-Host "Activating environment..."
$PYTHON_PATH = "W:\3 projects\Building\Tfrenzy\ppe-env\Scripts\python.exe"

Write-Host "Running Phase 3D Validation..."
Set-Location "w:\3 projects\Building\Tfrenzy\notebook"
& $PYTHON_PATH "scripts\phase3d_clean_validation.py"

Write-Host "Done. Check the outputs/phase3d_validation directory."
