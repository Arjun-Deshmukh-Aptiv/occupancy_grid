@echo off
setlocal EnableExtensions

set "VENV_DIR=.venv"
set "REQ_FILE=requirements.txt"

REM Find Python

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY=py"
    goto :python_found
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY=python"
    goto :python_found
)



echo Python not found. Install Python 3 and rerun.
exit /b 1

:python_found

REM Create requirements.txt if missing
if not exist "%REQ_FILE%" (
    (
        echo # Requirements for this project
        echo # Add packages below, one per line. Example:
        echo # requests==2.31.0
    ) > "%REQ_FILE%"

    echo Created %REQ_FILE% ^(template^).
)

REM Create virtual environment if needed
if exist "%VENV_DIR%" (
    echo Virtualenv already exists at %VENV_DIR%
) else (
    echo Creating virtual environment at %VENV_DIR%...
    %PY% -m venv "%VENV_DIR%"
    if errorlevel 1 exit /b 1
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo Virtualenv python not found at %VENV_PY%
    exit /b 1
)

echo Installing requirements...
"%VENV_PY%" -m pip install -r "%REQ_FILE%"
if errorlevel 1 exit /b 1

echo.
echo Done. Activate the venv with:
echo   %VENV_DIR%\Scripts\activate.bat

endlocal