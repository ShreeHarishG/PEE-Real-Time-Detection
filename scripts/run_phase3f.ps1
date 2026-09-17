# Execute Phase 3F: Violation Root-Cause Audit

Write-Host "Activating environment..."
$PYTHON_PATH = "W:\3 projects\Building\Tfrenzy\ppe-env\Scripts\python.exe"

Write-Host "Running Phase 3F Audit..."
Set-Location "w:\3 projects\Building\Tfrenzy\notebook"
& $PYTHON_PATH "scripts\phase3f_violation_audit.py"

Write-Host "Done. Check the outputs/phase3f_violation_audit directory."
