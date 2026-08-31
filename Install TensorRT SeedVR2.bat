@echo off
setlocal
title Install TensorRT SeedVR2 for ComfyUI
cd /d "%~dp0"
echo Installing TensorRT SeedVR2 for ComfyUI and its GPU runtime...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" %*
set "seedvr_exit_code=%errorlevel%"
echo.
if not "%seedvr_exit_code%"=="0" (
  echo Installation failed. See the message above and outputs\install.log.
) else (
  echo Installation complete.
  echo TensorRT acceleration is now active for SeedVR2 in ComfyUI.
)
pause
exit /b %seedvr_exit_code%
