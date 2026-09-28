@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" goto no_venv

call .venv\Scripts\activate.bat

if exist "%~dp0wheels\*.whl" goto offline

echo No local wheels found. Downloading PyTorch CUDA 12.8 from the internet...
echo Tip: run download_wheels.bat once to cache these for next time.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
goto after_torch

:offline
echo Installing PyTorch from local wheelhouse .\wheels offline...
pip install --no-index --find-links "%~dp0wheels" torch torchvision torchaudio

:after_torch
if errorlevel 1 goto torch_fail

echo.
echo Installing Unsloth, using .\wheels_unsloth cache if present...
pip install -q --find-links "%~dp0wheels_unsloth" unsloth
if errorlevel 1 goto unsloth_fail

echo.
echo Verifying torch + CUDA + Unsloth...
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
python -c "import unsloth; print('unsloth:', unsloth.__version__)"

echo.
echo Done. Window stays open with .venv active.
cmd /k
goto :eof

:no_venv
echo ERROR: .venv not found. Run create_env.bat first.
pause
exit /b 1

:torch_fail
echo.
echo ERROR: PyTorch install failed.
cmd /k
exit /b 1

:unsloth_fail
echo.
echo ERROR: Unsloth install failed.
cmd /k
exit /b 1
