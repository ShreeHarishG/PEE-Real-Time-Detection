# Execute Phase 4: Cleanup + Dataset Audit

Write-Host "Activating environment..."
$PYTHON_PATH = "W:\3 projects\Building\Tfrenzy\ppe-env\Scripts\python.exe"

Set-Location "w:\3 projects\Building\Tfrenzy\notebook"

Write-Host "`n=========================================="
Write-Host "STEP 1: PROJECT CLEANUP"
Write-Host "=========================================="
& $PYTHON_PATH "scripts\phase4_cleanup.py"

Write-Host "`n=========================================="
Write-Host "STEP 2: DATASET AUDIT"
Write-Host "=========================================="
& $PYTHON_PATH "scripts\phase4_dataset_audit.py"

Write-Host "`nDone. Check outputs/project_audit/ and outputs/v3/ directories."
