# Execute Phase 3I: Association Validation

Write-Host "Activating environment..."
$PYTHON_PATH = "W:\3 projects\Building\Tfrenzy\ppe-env\Scripts\python.exe"

Write-Host "Running Phase 3I Validation..."
Set-Location "w:\3 projects\Building\Tfrenzy\notebook"
& $PYTHON_PATH "scripts\phase3i_association_validation.py"

Write-Host "Done. Check the outputs/phase3i_association_validation directory."
