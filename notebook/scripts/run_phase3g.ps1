# Execute Phase 3G: Association Improvement

Write-Host "Activating environment..."
$PYTHON_PATH = "W:\3 projects\Building\Tfrenzy\ppe-env\Scripts\python.exe"

Write-Host "Running Phase 3G Improvement Test..."
Set-Location "w:\3 projects\Building\Tfrenzy\notebook"
& $PYTHON_PATH "scripts\phase3g_association_improvement.py"

Write-Host "Done. Check the outputs/phase3g_association directory."
