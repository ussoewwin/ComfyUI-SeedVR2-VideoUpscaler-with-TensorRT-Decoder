"""
ComfyUI-SeedVR2_VideoUpscaler Automated Installer
Installs requirements.txt, TensorRT RTX stack, and prepares VAE engines.
Attention backends (SageAttention / FlashAttention) are optional; if not installed, PyTorch SDPA is used.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "tensorrt_backend" / "artifacts"

ENGINES = [
    "vae_encoder_5f_tile512.rtxplan",
    "vae_encoder_21f_tile512.rtxplan",
    "vae_decoder_tile_512_5f.rtxplan",
    "vae_decoder_tile_256_21f.rtxplan",
]


def log_step(message: str) -> None:
    print(f"\n[SeedVR2 TensorRT Installer] == {message} ==")


def ensure_ffmpeg_path() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    candidate_dirs = [
        Path(r"C:\Program Files\ffmpeg\bin"),
        Path(r"C:\Program Files\ffmpeg"),
        Path(r"C:\Program Files (x86)\ffmpeg\bin"),
        Path(r"C:\ffmpeg\bin"),
        Path(r"D:\ffmpeg\bin"),
        ROOT / "bin" / "ffmpeg" / "bin",
        ROOT / "bin",
    ]
    for d in candidate_dirs:
        if (d / "ffmpeg.exe").exists() and (d / "ffprobe.exe").exists():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            return

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe and Path(ffmpeg_exe).exists():
            ffmpeg_dir = str(Path(ffmpeg_exe).parent)
            if ffmpeg_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def pip_install(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", *args]
    print(f"> {' '.join(cmd)}")
    subprocess.check_call(cmd)


def install_requirements() -> None:
    req_file = ROOT / "requirements.txt"
    if not req_file.exists():
        return
    log_step("Installing base requirements from requirements.txt")
    try:
        pip_install(["-r", str(req_file)])
    except Exception as exc:
        print(f"[SeedVR2] Warning: Failed to install some requirements.txt packages: {exc}")


def ensure_package(import_name: str, install_name: str | None = None, no_deps: bool = True) -> None:
    try:
        __import__(import_name)
        return
    except ImportError:
        pass
    pkg = install_name or import_name
    log_step(f"Installing {pkg}")
    args = [pkg]
    if no_deps:
        args.append("--no-deps")
    try:
        pip_install(args)
    except Exception as exc:
        print(f"[SeedVR2] Warning: Could not install {pkg}: {exc}")


def sync_or_build_engines() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    all_present = True
    for engine_name in ENGINES:
        local_path = ARTIFACTS_DIR / engine_name
        if local_path.exists() and local_path.stat().st_size > 1_000_000:
            print(f"[SeedVR2] Engine ready: {engine_name}")
            continue
        all_present = False

    if not all_present:
        log_step("Building missing TensorRT RTX VAE engines (5f / 21f)")
        prepare_script = ROOT / "scripts" / "prepare_tensorrt.py"
        if prepare_script.exists():
            try:
                subprocess.check_call([sys.executable, str(prepare_script)])
            except Exception as exc:
                print(f"[SeedVR2] Warning: TensorRT engine preparation skipped or failed: {exc}")
                print(f"[SeedVR2] Engines can be built on demand in ComfyUI using the 'SeedVR2 Build TensorRT VAE Engines' node.")


def download_default_models() -> None:
    download_script = ROOT / "scripts" / "download_models.py"
    if download_script.exists():
        try:
            subprocess.check_call([sys.executable, str(download_script)])
        except Exception as exc:
            print(f"[SeedVR2] Warning: model pre-download failed: {exc} (will download on first run)")


def main() -> int:
    print("=" * 80)
    print("SeedVR2 Video Upscaler — ComfyUI TensorRT Auto-Installer")
    print(f"Python: {sys.executable}")
    print("=" * 80)

    # 0. Ensure FFmpeg is on PATH
    ensure_ffmpeg_path()

    # 1. Base requirements from requirements.txt
    install_requirements()

    # 2. TensorRT RTX & ONNX stack
    ensure_package("tensorrt_rtx", "tensorrt-rtx==1.6.1.120", no_deps=True)
    ensure_package("triton", "triton-windows==3.5.1.post24", no_deps=True)
    ensure_package("onnx", "onnx==1.22.0", no_deps=True)
    ensure_package("onnxscript", "onnxscript==0.7.1", no_deps=True)
    ensure_package("polygraphy", "polygraphy==0.53.4", no_deps=True)

    # 3. TensorRT RTX VAE Engines
    sync_or_build_engines()

    # 4. Default models
    download_default_models()

    # 5. Verify
    verify_script = ROOT / "scripts" / "verify_install.py"
    if verify_script.exists():
        try:
            subprocess.run([sys.executable, str(verify_script)], check=False)
        except Exception as exc:
            print(f"[SeedVR2] Verification check warning: {exc}")

    print("\n" + "=" * 80)
    print("SeedVR2 Video Upscaler (TensorRT) installation complete.")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
