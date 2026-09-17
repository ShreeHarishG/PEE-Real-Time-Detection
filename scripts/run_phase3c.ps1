# Execute Phase 3C: Violation Evidence Audit

Write-Host "Activating environment..."
$PYTHON_PATH = "W:\3 projects\Building\Tfrenzy\ppe-env\Scripts\python.exe"

Write-Host "Running Phase 3C Audit..."
Set-Location "w:\3 projects\Building\Tfrenzy\notebook"
& $PYTHON_PATH "scripts\phase3c_audit_report.py"
