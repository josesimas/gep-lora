@echo off
rem Runs the async API server: http://127.0.0.1:8780 -- start at /guide.html;
rem /runs.html, /settings.html and /console.html are the other pages.
rem Uses the venv one level up, which has torch and unsloth -- inference loads
rem the model in this process. Extra arguments are passed on, e.g.
rem     run_api_server.bat --port 9000 --host 0.0.0.0
rem Jobs only run while a worker is up too: python -m gep_lora.service.worker

cd /d "%~dp0"
set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Cannot find %PYTHON%
    echo The API needs the project venv one level up. See README.md.
    pause
    exit /b 1
)

"%PYTHON%" -m gep_lora.apps.web.server %*
pause
