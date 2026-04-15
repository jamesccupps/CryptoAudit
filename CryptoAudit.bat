@echo off
title CryptoAudit ML
cd /d "%~dp0"

:: Find the right Python — prefer 'py' launcher, fall back to 'python'
set PYTHON=
where py >nul 2>&1
if not errorlevel 1 (
    set PYTHON=py
) else (
    where python >nul 2>&1
    if not errorlevel 1 (
        :: Make sure it's not Inkscape/bundled Python
        python -m pip --version >nul 2>&1
        if not errorlevel 1 (
            set PYTHON=python
        )
    )
)

if "%PYTHON%"=="" (
    echo Python not found. Install Python 3.10+ from python.org
    echo Make sure "Add Python to PATH" is checked during install.
    pause
    exit /b 1
)

echo Using:
%PYTHON% --version
%PYTHON% -c "import sys; print(sys.executable)"
echo.

:: Install missing deps
%PYTHON% -c "import numpy" 2>nul || %PYTHON% -m pip install numpy --quiet
%PYTHON% -c "import sklearn" 2>nul || %PYTHON% -m pip install scikit-learn --quiet
%PYTHON% -c "import scipy" 2>nul || %PYTHON% -m pip install scipy --quiet
%PYTHON% -c "import joblib" 2>nul || %PYTHON% -m pip install joblib --quiet
%PYTHON% -c "import scapy" 2>nul || %PYTHON% -m pip install scapy --quiet 2>nul

:: Verify
%PYTHON% -c "import numpy, sklearn, scipy, joblib" 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Dependencies failed to install.
    echo Try: %PYTHON% -m pip install numpy scikit-learn scipy joblib
    pause
    exit /b 1
)

:: Launch
echo Starting CryptoAudit...
echo.
%PYTHON% crypto_audit.py %*

echo.
echo CryptoAudit closed. Exit code: %errorlevel%
pause
