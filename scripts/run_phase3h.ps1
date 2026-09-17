# Execute Phase 3H: Association Audit

Write-Host "Activating environment..."
$PYTHON_PATH = "W:\3 projects\Building\Tfrenzy\ppe-env\Scripts\python.exe"

Write-Host "Running Phase 3H Association Audit..."
Set-Location "w:\3 projects\Building\Tfrenzy\notebook"
& $PYTHON_PATH "scripts\phase3h_association_audit.py"

Write-Host "Done. Check the outputs/phase3h_association_audit directory."
