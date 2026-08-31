# Wrapper for building TensorRT engines directly in ComfyUI environment
$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$NodeRoot = Split-Path -Parent $ScriptDir
$TargetPy = Join-Path $NodeRoot '..\..\..\python_embeded\python.exe'
if (-not (Test-Path $TargetPy)) { $TargetPy = 'python' }
& $TargetPy (Join-Path $ScriptDir 'prepare_tensorrt.py')
