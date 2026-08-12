# Execute V3 Experiment 1: Train and Evaluate
Write-Host "Activating environment..."
$PYTHON_PATH = "W:\3 projects\Building\Tfrenzy\ppe-env\Scripts\python.exe"

Set-Location "w:\3 projects\Building\Tfrenzy\notebook"

Write-Host "`n=========================================="
Write-Host "V3 EXPERIMENT 1: TRAINING V3-HN"
Write-Host "=========================================="
& $PYTHON_PATH "scripts\exp1_train.py"

Write-Host "`n=========================================="
Write-Host "V3 EXPERIMENT 1: EVALUATION"
Write-Host "=========================================="
& $PYTHON_PATH "scripts\exp1_evaluate.py"

Write-Host "`nDone. Check outputs/v3/exp1_results.csv."
