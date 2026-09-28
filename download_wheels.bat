@echo off
cd /d "%~dp0"

echo Downloading PyTorch CUDA 12.8 wheels into .\wheels (one time)...
py -3.11 -m pip download torch torchvision torchaudio ^
    --index-url https://download.pytorch.org/whl/cu128 ^
    -d "%~dp0wheels"

if errorlevel 1 (
    echo.
    echo ERROR: torch download failed.
    pause
    exit /b 1
)

echo.
echo Downloading Unsloth + dependencies into .\wheels_unsloth (one time)...
echo (kept separate from .\wheels so it cannot disturb the cu128 torch install)
py -3.11 -m pip download unsloth ^
    --find-links "%~dp0wheels" ^
    -d "%~dp0wheels_unsloth"

if errorlevel 1 (
    echo.
    echo ERROR: unsloth download failed.
    pause
    exit /b 1
)

echo.
echo Done. torch -^> .\wheels, unsloth -^> .\wheels_unsloth. setup_torch.bat installs offline.
pause
