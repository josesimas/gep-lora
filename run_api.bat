@echo off
rem Starts -- or restarts -- the async API server and its worker, each in its
rem own window, with one command:
rem     run_api.bat            stop whatever is running, then start both
rem     run_api.bat stop       stop both
rem     run_api.bat status     say which are running
rem The running ones are found by command line (python -m gep_lora.apps.web.server /
rem gep_lora.service.worker, or the async_api.* names they had before), so a server or worker started by hand with
rem run_api_server.bat / run_api_worker.bat is restarted too, window and all.
rem Stopping is a hard kill: a job the worker was running is marked failed by
rem the next worker to start, as it would be after the worker's own Ctrl+C.

setlocal
cd /d "%~dp0"

set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=restart"
if /i "%ACTION%"=="start"   goto restart
if /i "%ACTION%"=="restart" goto restart
if /i "%ACTION%"=="stop"    goto stop
if /i "%ACTION%"=="status"  goto status
echo Usage: run_api.bat [restart^|stop^|status]
exit /b 1

:restart
call :ps stop
rem let the old server let go of its port before the new one binds it
ping -n 3 127.0.0.1 >nul
start "GEP API server" /min cmd /c ""%~dp0run_api_server.bat""
start "GEP API worker" /min cmd /c ""%~dp0run_api_worker.bat""
echo Started the server and the worker: http://127.0.0.1:8780/guide.html
exit /b 0

:stop
call :ps stop
exit /b 0

:status
call :ps status
exit /b 0

rem -- One PowerShell pass per action. For each of server and worker: the
rem    python processes running that module, and the cmd.exe window running
rem    the .bat around each (killing that takes the python and the window's
rem    "pause" with it).
:ps
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$all = Get-CimInstance Win32_Process;" ^
  "$mods = @{ server = 'gep_lora\.apps\.web\.server'; worker = 'gep_lora\.service\.worker' };" ^
  "foreach ($name in 'server','worker') {" ^
  "  $py = @($all | Where-Object { $_.CommandLine -match ('-m\s+(' + $mods[$name] + '|async_api\.' + $name + ')(\s|$)') });" ^
  "  if (-not $py) { Write-Host ('API ' + $name + ': not running'); continue }" ^
  "  if ('%~1' -eq 'status') { Write-Host ('API ' + $name + ': running, pid ' + ($py.ProcessId -join ', ')); continue }" ^
  "  foreach ($p in $py) {" ^
  "    $top = $p.ProcessId;" ^
  "    $parent = $all | Where-Object { $_.ProcessId -eq $p.ParentProcessId -and $_.Name -eq 'cmd.exe' -and $_.CommandLine -match 'run_api_' };" ^
  "    if ($parent) { $top = $parent.ProcessId }" ^
  "    taskkill /pid $top /t /f 2>$null | Out-Null" ^
  "  }" ^
  "  Write-Host ('API ' + $name + ': stopped')" ^
  "}"
exit /b 0
