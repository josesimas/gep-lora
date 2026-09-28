@echo off
cd /d "%~dp0"
call .venv\Scripts\deactivate.bat 2>nul
rmdir /s /q .venv
echo Environment deleted.
