# Execute Phase 3E: Positive PPE Validation

Write-Host "Activating environment..."
$PYTHON_PATH = "W:\3 projects\Building\Tfrenzy\ppe-env\Scripts\python.exe"

Write-Host "Running Phase 3E Validation..."
Set-Location "w:\3 projects\Building\Tfrenzy\notebook"
& $PYTHON_PATH "scripts\phase3e_positive_validation.py"

Write-Host "Done. Check the outputs/phase3e_positive_validation directory."
