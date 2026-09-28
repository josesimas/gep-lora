@echo off
cd /d "%~dp0"
py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
echo.
echo Environment created. This window stays open with .venv active.
cmd /k
