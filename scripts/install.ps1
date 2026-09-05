param(
    [switch]$SkipModels,
    [switch]$SkipTensorRT,
    [switch]$Repair,
    [string]$PythonPath
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

$NodeRoot = Split-Path -Parent $PSScriptRoot
$Outputs = Join-Path $NodeRoot 'outputs'
$Log = Join-Path $Outputs 'install.log'
$Marker = Join-Path $NodeRoot '.seedvr2-trt-installed'
Push-Location -LiteralPath $NodeRoot

New-Item -ItemType Directory -Force $Outputs | Out-Null
try { Start-Transcript -Path $Log -Append | Out-Null } catch {}

function Write-Step([string]$Message) {
    Write-Host ''
    Write-Host ('== ' + $Message + ' ==') -ForegroundColor Cyan
}

function Refresh-Path {
    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = $machinePath + ';' + $userPath

    $candidatePaths = @(
        'C:\Program Files\ffmpeg\bin',
        'C:\Program Files\ffmpeg',
        'C:\Program Files (x86)\ffmpeg\bin',
        'C:\ffmpeg\bin',
        'D:\ffmpeg\bin',
        (Join-Path $NodeRoot 'bin\ffmpeg\bin'),
        (Join-Path $NodeRoot 'bin')
    )
    foreach ($p in $candidatePaths) {
        if ((Test-Path (Join-Path $p 'ffmpeg.exe')) -and (Test-Path (Join-Path $p 'ffprobe.exe'))) {
            $env:Path = $p + ';' + $env:Path
            break
        }
    }
}

function Find-ComfyPython {
    if ($PythonPath -and (Test-Path -LiteralPath $PythonPath)) {
        return (Resolve-Path $PythonPath).Path
    }

    # 1. ComfyUI python_embeded
    $comfyEmbeded = @(
        'D:\USERFILES\ComfyUI\python_embeded\python.exe',
        'C:\ComfyUI\python_embeded\python.exe',
        (Join-Path $NodeRoot '..\..\..\python_embeded\python.exe'),
        (Join-Path $NodeRoot '..\..\python_embeded\python.exe')
    )
    foreach ($p in $comfyEmbeded) {
        if (Test-Path -LiteralPath $p) { return (Resolve-Path $p).Path }
    }

    # 2. Active Virtual Environment
    if ($env:VIRTUAL_ENV) {
        $activePy = Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe'
        if (Test-Path -LiteralPath $activePy) { return $activePy }
    }

    # 3. PATH Python
    $pythonCmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCmd) { return $pythonCmd.Source }

    return $null
}

function Ensure-FFmpeg {
    Refresh-Path
    if ((Get-Command ffmpeg.exe -ErrorAction SilentlyContinue) -and (Get-Command ffprobe.exe -ErrorAction SilentlyContinue)) {
        return
    }

    $candidatePaths = @(
        'C:\Program Files\ffmpeg\bin',
        'C:\Program Files\ffmpeg',
        'C:\Program Files (x86)\ffmpeg\bin',
        'C:\ffmpeg\bin',
        'D:\ffmpeg\bin'
    )
    foreach ($candidate in $candidatePaths) {
        if ((Test-Path -LiteralPath (Join-Path $candidate 'ffmpeg.exe')) -and (Test-Path -LiteralPath (Join-Path $candidate 'ffprobe.exe'))) {
            $env:Path = $candidate + ';' + $env:Path
            return
        }
    }
}

try {
    Write-Host 'ComfyUI SeedVR2 Video Upscaler — TensorRT & Fast Attention Installer' -ForegroundColor Green
    Write-Host 'Installs TensorRT RTX, FlashAttention 2, SageAttention 2, models, and VAE engines into ComfyUI environment.'

    Ensure-FFmpeg

    $TargetPython = Find-ComfyPython
    if (-not $TargetPython) {
        throw 'ComfyUI Python environment was not found. Please specify -PythonPath "path\to\python.exe".'
    }
    Write-Host "Target ComfyUI Python: $TargetPython" -ForegroundColor Cyan

    function Invoke-TargetPip([string[]]$Arguments) {
        & $TargetPython -m pip @Arguments
        if ($LASTEXITCODE -ne 0) { throw "pip failed: $($Arguments -join ' ')" }
    }

    Write-Step 'Verifying ComfyUI PyTorch and CUDA'
    $torchInfo = & $TargetPython -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $torchInfo) {
        throw "Could not execute PyTorch in $TargetPython. Ensure ComfyUI environment is working."
    }
    Write-Host $torchInfo -ForegroundColor Green

    Write-Step 'Installing TensorRT RTX & ONNX tools'
    $reqFile = Join-Path $NodeRoot 'requirements-windows-cu132.txt'
    if (Test-Path $reqFile) {
        Invoke-TargetPip @('install', '--requirement', $reqFile, '--no-deps', '--index-url', 'https://pypi.org/simple')
    }

    function Install-CachedWheel([string]$ModuleName, [string]$PackageName, [string]$Url) {
        $check = & $TargetPython -c "import $ModuleName" 2>$null
        if ($LASTEXITCODE -eq 0 -and -not $Repair) {
            Write-Host "$PackageName is already installed." -ForegroundColor Green
            return
        }

        $wheelsDir = Join-Path $NodeRoot 'wheels'
        New-Item -ItemType Directory -Force $wheelsDir | Out-Null
        $filename = [System.IO.Path]::GetFileName($Url.Split('?')[0])
        $filename = [System.Uri]::UnescapeDataString($filename)
        $cachedPath = Join-Path $wheelsDir $filename

        if (-not (Test-Path -LiteralPath $cachedPath)) {
            Write-Step "Downloading $PackageName wheel"
            Invoke-WebRequest -Uri $Url -OutFile $cachedPath -UseBasicParsing
        }

        Write-Step "Installing $PackageName"
        Invoke-TargetPip @('install', $cachedPath, '--no-deps')
    }

    Write-Step 'Installing FlashAttention 2 and SageAttention 2'
    $pyTag = (& $TargetPython -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')" 2>$null).Trim()
    Write-Host "Detected ComfyUI Python tag: $pyTag" -ForegroundColor Cyan

    $flashWheels = @{
        'cp311' = 'https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.8.3%2Bcu130torch2.9.1cxx11abiTRUE-cp311-cp311-win_amd64.whl'
        'cp312' = 'https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.9.1%2Bcu132torch2.13.0cxx11abiTRUE-cp312-cp312-win_amd64.whl'
        'cp313' = 'https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.9.1%2Bcu132torch2.13.0cxx11abiTRUE-cp313-cp313-win_amd64.whl'
        'cp314' = 'https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.9.1%2Bcu132torch2.13.0cxx11abiTRUE-cp314-cp314-win_amd64.whl'
    }

    $sageWheels = @{
        'cp312' = 'https://huggingface.co/ussoewwin/Sage-Attention-for-Windows/resolve/main/sageattention-2.2.0.post6%2Bcu132torch2.13.0-cp312-cp312-win_amd64.whl'
        'cp313' = 'https://huggingface.co/ussoewwin/Sage-Attention-for-Windows/resolve/main/sageattention-2.2.0.post6%2Bcu132torch2.13.0-cp313-cp313-win_amd64.whl'
        'cp314' = 'https://huggingface.co/ussoewwin/Sage-Attention-for-Windows/resolve/main/sageattention-2.2.0.post6%2Bcu132torch2.13.0-cp314-cp314-win_amd64.whl'
    }

    if ($flashWheels.ContainsKey($pyTag)) {
        try { Install-CachedWheel 'flash_attn' 'FlashAttention 2' $flashWheels[$pyTag] } catch { Write-Warning "FlashAttention 2 installation skipped: $_" }
    } else {
        Write-Host "No prebuilt FlashAttention 2 wheel for $pyTag." -ForegroundColor Yellow
    }

    if ($sageWheels.ContainsKey($pyTag)) {
        try { Install-CachedWheel 'sageattention' 'SageAttention 2' $sageWheels[$pyTag] } catch { Write-Warning "SageAttention 2 installation skipped: $_" }
    } else {
        Write-Host "No prebuilt SageAttention 2 wheel for $pyTag." -ForegroundColor Yellow
    }

    Write-Step 'Checking Attention & CUDA optimization'
    & $TargetPython -c "import sys, torch; sys.path.insert(0, '.'); from src.optimization.compatibility import SAGE_ATTN_2_AVAILABLE, FLASH_ATTN_2_AVAILABLE; print('SageAttention 2:', 'ready' if SAGE_ATTN_2_AVAILABLE else 'not available'); print('FlashAttention 2:', 'ready' if FLASH_ATTN_2_AVAILABLE else 'not available')"
    if ($LASTEXITCODE -ne 0) { Write-Warning 'Attention kernel verification reported a warning.' }

    if (-not $SkipModels) {
        Write-Step 'Ensuring default SeedVR2 3B FP8 model and VAE'
        & $TargetPython (Join-Path $PSScriptRoot 'download_models.py')
        if ($LASTEXITCODE -ne 0) { Write-Warning 'Model download can be resumed when executing the node.' }
    }

    if (-not $SkipTensorRT) {
        Write-Step 'Building TensorRT RTX VAE engines'
        & $TargetPython (Join-Path $PSScriptRoot 'prepare_tensorrt.py')
        if ($LASTEXITCODE -ne 0) { throw 'TensorRT engine preparation failed.' }
    }

    Write-Step 'Final readiness check'
    Refresh-Path
    & $TargetPython (Join-Path $PSScriptRoot 'verify_install.py')
    if ($LASTEXITCODE -ne 0) { throw 'The final installation check failed.' }
    Set-Content -LiteralPath $Marker -Value (Get-Date -Format o) -Encoding ascii

    Write-Host ''
    Write-Host 'ComfyUI SeedVR2 Video Upscaler (TensorRT) setup is complete.' -ForegroundColor Green
}
catch {
    Write-Host ''
    Write-Host ('INSTALLATION FAILED: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host ('Detailed log: ' + $Log) -ForegroundColor Yellow
    exit 1
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
    try { Pop-Location } catch {}
}
