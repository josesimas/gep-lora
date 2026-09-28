@echo off
rem Runs the async API worker: takes queued jobs in arrival order and runs each
rem one through main.py. Run one worker per jobs folder, beside run_api_server.bat.
rem Uses the venv one level up, which has torch and unsloth -- main.py launches
rem every generated script with this same interpreter. Extra arguments are
rem passed on, e.g.
rem     run_api_worker.bat --once
rem Stop it with Ctrl+C; a job it is running is stopped and marked failed.

cd /d "%~dp0"
set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Cannot find %PYTHON%
    echo The worker needs the project venv one level up. See README.md.
    pause
    exit /b 1
)

"%PYTHON%" -m gep_lora.service.worker %*
pause
