@echo off
setlocal
cd /d "%~dp0"
echo ==== leadcut setup ====
echo.

set "PYCMD="
py -3.12 --version >nul 2>nul && set "PYCMD=py -3.12"
if not defined PYCMD py -3.11 --version >nul 2>nul && set "PYCMD=py -3.11"
if not defined PYCMD py -3.10 --version >nul 2>nul && set "PYCMD=py -3.10"
if not defined PYCMD python --version >nul 2>nul && set "PYCMD=python"
if not defined PYCMD goto nopython

echo Using Python: %PYCMD%
if not exist ".venv\Scripts\python.exe" %PYCMD% -m venv .venv
if not exist ".venv\Scripts\python.exe" goto novenv

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -e ".[cpu]"
if errorlevel 1 goto pipfail

echo.
python -m leadcut doctor --fix
echo.
echo When the line above says "ALL GOOD", double-click start_windows.bat
pause
exit /b 0

:nopython
echo Python was not found. Install Python 3.12 from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH"), then run this file again.
pause
exit /b 1

:novenv
echo Could not create the virtual environment (.venv).
pause
exit /b 1

:pipfail
echo.
echo The install failed. Scroll up and read the last red lines, then send them to get help.
pause
exit /b 1
