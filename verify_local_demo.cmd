@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "ROOT=%~dp0"
set "PY=%ROOT%backend\.venv\Scripts\python.exe"
set "STRICT="

if /I "%~1"=="--require-models" (
    set "STRICT=--require-models"
) else if not "%~1"=="" (
    echo Usage: verify_local_demo.cmd [--require-models]
    exit /b 2
)
if not "%~2"=="" (
    echo Usage: verify_local_demo.cmd [--require-models]
    exit /b 2
)

if not exist "%PY%" (
    echo ERROR: Backend virtual environment is missing: "%PY%"
    echo Create it in backend and install requirements-dev.txt and requirements-ml.txt.
    exit /b 2
)
if not exist "%ROOT%frontend\node_modules" (
    echo ERROR: Frontend dependencies are missing.
    echo From frontend run: npm.cmd ci
    exit /b 2
)
where npm.cmd >nul 2>&1
if errorlevel 1 (
    echo ERROR: npm.cmd is not on PATH. Install Node.js and reopen CMD.
    exit /b 2
)

echo.
echo [1/4] Backend automated tests
cd /d "%ROOT%backend"
if errorlevel 1 goto failed
"%PY%" -m pytest -q
if errorlevel 1 goto failed

echo.
echo [2/4] In-process HTTP smoke test
"%PY%" scripts\final_acceptance.py %STRICT%
if errorlevel 1 goto failed

echo.
echo [3/4] Frontend production build
cd /d "%ROOT%frontend"
if errorlevel 1 goto failed
call npm.cmd run build
if errorlevel 1 goto failed

echo.
echo [4/4] Frontend lint
call npm.cmd run lint
if errorlevel 1 goto failed

echo.
echo PASS: AcousticSpace local demo software verification completed.
echo NOTE: This does not establish scientific deepfake or blind-RIR accuracy.
echo See docs\FINAL_HANDOFF.md for launching and demonstrating the project.
exit /b 0

:failed
echo.
echo FAIL: A verification step failed. Review the output above.
exit /b 1
