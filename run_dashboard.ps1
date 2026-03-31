$pythonExe = Join-Path $PSScriptRoot ".venv_std\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
}
$appFile = Join-Path $PSScriptRoot "dashboard\app.py"
$requirementsFile = Join-Path $PSScriptRoot "requirements.txt"

if (-not (Test-Path $pythonExe)) {
    Write-Host "Python not found at '$pythonExe'." -ForegroundColor Red
    Write-Host "Create .venv_std or update `$pythonExe in run_dashboard.ps1 if your Python is installed elsewhere."
    exit 1
}

& $pythonExe -m streamlit --version *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Streamlit is not installed for '$pythonExe'." -ForegroundColor Yellow
    Write-Host "Install dependencies with:"
    Write-Host "  & `"$pythonExe`" -m pip install -r `"$requirementsFile`""
    exit 1
}

& $pythonExe -c "import joblib, pandas, numpy, sklearn, matplotlib, seaborn, plotly, openpyxl, xgboost, reportlab" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python dependencies for the dashboard are missing or incomplete in '$pythonExe'." -ForegroundColor Yellow
    Write-Host "Install dependencies with:"
    Write-Host "  & `"$pythonExe`" -m pip install -r `"$requirementsFile`""
    exit 1
}

Write-Host "Launching RSPM dashboard..." -ForegroundColor Cyan
& $pythonExe -m streamlit run $appFile
