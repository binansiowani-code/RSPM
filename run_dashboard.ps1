$pythonExe = "C:\Program Files\Python313\python.exe"
$appFile = Join-Path $PSScriptRoot "dashboard\app.py"
$requirementsFile = Join-Path $PSScriptRoot "requirements.txt"

if (-not (Test-Path $pythonExe)) {
    Write-Host "Python not found at '$pythonExe'." -ForegroundColor Red
    Write-Host "Update `$pythonExe in run_dashboard.ps1 if your Python is installed elsewhere."
    exit 1
}

& $pythonExe -m streamlit --version *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Streamlit is not installed for '$pythonExe'." -ForegroundColor Yellow
    Write-Host "Install dependencies with:"
    Write-Host "  & `"$pythonExe`" -m pip install -r `"$requirementsFile`""
    exit 1
}

Write-Host "Launching RSPM dashboard..." -ForegroundColor Cyan
& $pythonExe -m streamlit run $appFile
