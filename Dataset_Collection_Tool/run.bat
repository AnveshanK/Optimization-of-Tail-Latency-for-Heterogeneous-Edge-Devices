@echo off
rem Windows:   run.bat [options]      (same as: py run.py [options])
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 run.py %*
    exit /b %errorlevel%
)
where python >nul 2>nul
if %errorlevel%==0 (
    python run.py %*
    exit /b %errorlevel%
)
echo Python 3.9 - 3.13 is needed (3.12 recommended): https://www.python.org/downloads/
exit /b 1
