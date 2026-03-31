@echo off
setlocal

set "PYTHON_EXE=%~dp0.venv_std\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "APP_FILE=%~dp0dashboard\app.py"
set "REQ_FILE=%~dp0requirements.txt"
set "DEP_CHECK_FILE=%TEMP%\rspm_dep_check.txt"

if not exist "%PYTHON_EXE%" (
    echo Python not found at "%PYTHON_EXE%".
    echo Update PYTHON_EXE inside run_dashboard.bat if your Python is installed elsewhere.
    pause
    exit /b 1
)

if not exist "%APP_FILE%" (
    echo Dashboard app not found at "%APP_FILE%".
    pause
    exit /b 1
)

"%PYTHON_EXE%" -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo Streamlit is not installed for "%PYTHON_EXE%".
    echo Install dependencies with:
    echo   "%PYTHON_EXE%" -m pip install -r "%REQ_FILE%"
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import joblib, pandas, numpy, sklearn, matplotlib, seaborn, plotly, openpyxl, xgboost, reportlab" >nul 2>"%DEP_CHECK_FILE%"
if errorlevel 1 (
    echo Python dependencies for the dashboard are missing or incomplete in "%PYTHON_EXE%".
    type "%DEP_CHECK_FILE%"
    echo.
    echo Install dependencies with:
    echo   "%PYTHON_EXE%" -m pip install -r "%REQ_FILE%"
    pause
    exit /b 1
)

if exist "%DEP_CHECK_FILE%" del "%DEP_CHECK_FILE%"

echo Launching RSPM dashboard...
"%PYTHON_EXE%" -m streamlit run "%APP_FILE%"

endlocal
