@echo off
title CryptoAudit ML
cd /d "%~dp0"

:: Find actual Python executable path (not just 'py' launcher)
set PYTHON=
for /f "tokens=*" %%i in ('py -c "import sys;print(sys.executable)" 2^>nul') do set PYTHON=%%i
if "%PYTHON%"=="" for /f "tokens=*" %%i in ('python -c "import sys;print(sys.executable)" 2^>nul') do set PYTHON=%%i

if "%PYTHON%"=="" (
    echo Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

echo Using: %PYTHON%
"%PYTHON%" --version
echo.

:: Install deps
echo Installing dependencies...
"%PYTHON%" -m pip install numpy scikit-learn scipy joblib --quiet 2>nul
"%PYTHON%" -m pip install scapy --quiet 2>nul

:: Verify
"%PYTHON%" -c "import numpy,sklearn,scipy,joblib;print('Dependencies OK')"
if errorlevel 1 (
    echo ERROR: Dependencies failed. Try:
    echo   "%PYTHON%" -m pip install numpy scikit-learn scipy joblib
    pause
    exit /b 1
)
echo.

:: Launch using the actual executable, not the py launcher
echo Starting CryptoAudit...
echo.
"%PYTHON%" crypto_audit.py %*
echo.
echo CryptoAudit closed. Exit code: %errorlevel%
pause
