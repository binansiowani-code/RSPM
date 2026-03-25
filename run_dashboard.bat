@echo off
setlocal

set "PYTHON_EXE=C:\Program Files\Python313\python.exe"
set "APP_FILE=%~dp0dashboard\app.py"
set "REQ_FILE=%~dp0requirements.txt"

if not exist "%PYTHON_EXE%" (
    echo Python not found at "%PYTHON_EXE%".
    echo Update PYTHON_EXE inside run_dashboard.bat if your Python is installed elsewhere.
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

echo Launching RSPM dashboard...
"%PYTHON_EXE%" -m streamlit run "%APP_FILE%"

endlocal
