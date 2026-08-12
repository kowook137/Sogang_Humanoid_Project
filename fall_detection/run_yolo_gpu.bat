@echo off
setlocal
pushd "%~dp0.." || exit /b 1

if defined CONDA_PREFIX (
  set "FALL_PYTHON=%CONDA_PREFIX%\python.exe"
) else (
  set "FALL_PYTHON=%USERPROFILE%\miniconda3\envs\fall-detection\python.exe"
)

if not exist "%FALL_PYTHON%" (
  echo Python was not found: %FALL_PYTHON%
  echo Activate the fall-detection Conda environment and try again.
  popd
  pause
  exit /b 1
)

"%FALL_PYTHON%" fall_detection\yolo_pose.py --source 0 --device 0 %*
set "FALL_EXIT=%ERRORLEVEL%"
popd

if not "%FALL_EXIT%"=="0" (
  echo.
  echo GPU fall detection failed. Review the error above.
  pause
)
exit /b %FALL_EXIT%
