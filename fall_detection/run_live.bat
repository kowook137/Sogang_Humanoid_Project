@echo off
setlocal
cd /d "%~dp0.."
if defined CONDA_PREFIX (
  "%CONDA_PREFIX%\python.exe" fall_detection\live_detect.py %*
) else (
  python fall_detection\live_detect.py %*
)
if errorlevel 1 (
  echo.
  echo Live detection failed. Activate the fall-detection environment and try again.
  pause
)
endlocal
