"""
ComfyUI-SeedVR2_VideoUpscaler Automated Installer
Installs TensorRT RTX, FlashAttention 2, SageAttention 2, and prepares VAE engines.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WHEELS_DIR = ROOT / "wheels"
ARTIFACTS_DIR = ROOT / "tensorrt_backend" / "artifacts"

WHEELS = [
    {
        "module": "flash_attn",
        "name": "FlashAttention 2",
        "url": "https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.8.4%2Bcu132torch2.13.0cxx11abiTRUE-cp314-cp314-win_amd64.whl"
    },
    {
        "module": "sageattention",
        "name": "SageAttention 2",
        "url": "https://huggingface.co/ussoewwin/Sage-Attention-for-Windows/resolve/main/sageattention-2.2.0.post6%2Bcu132torch2.13.0-cp314-cp314-win_amd64.whl"
    }
]

ENGINES = [
    "vae_encoder_5f_tile512.rtxplan",
    "vae_encoder_21f_tile512.rtxplan",
    "vae_decoder_tile_512_5f.rtxplan",
    "vae_decoder_tile_256_21f.rtxplan"
]


def log_step(message: str) -> None:
    print(f"\n[SeedVR2 TensorRT Installer] == {message} ==")


def pip_install(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", *args]
    print(f"> {' '.join(cmd)}")
    subprocess.check_call(cmd)


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
    pip_install(args)


def install_wheels() -> None:
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)
    for entry in WHEELS:
        try:
            __import__(entry["module"])
            print(f"[SeedVR2] {entry['name']} is already installed.")
            continue
        except ImportError:
            pass

        url = entry["url"]
        filename = urllib.parse.unquote(Path(urllib.parse.urlparse(url).path).name)
        cached_file = WHEELS_DIR / filename
        if not cached_file.exists() or cached_file.stat().st_size < 1000:
            log_step(f"Downloading {entry['name']} wheel")
            print(f"URL: {url}")
            urllib.request.urlretrieve(url, cached_file)

        log_step(f"Installing {entry['name']}")
        pip_install([str(cached_file), "--no-deps"])


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
        log_step("Building missing TensorRT RTX VAE engines")
        prepare_script = ROOT / "scripts" / "prepare_tensorrt.py"
        if prepare_script.exists():
            subprocess.check_call([sys.executable, str(prepare_script)])


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

    # 1. Base requirements
    ensure_package("tensorrt_rtx", "tensorrt-rtx==1.6.1.120", no_deps=True)
    ensure_package("triton", "triton-windows==3.5.1.post24", no_deps=True)
    ensure_package("onnx", "onnx==1.22.0", no_deps=True)
    ensure_package("onnxscript", "onnxscript==0.7.1", no_deps=True)
    ensure_package("polygraphy", "polygraphy==0.53.4", no_deps=True)

    # 2. FlashAttention 2 & SageAttention 2 wheels
    install_wheels()

    # 3. TensorRT RTX VAE Engines
    sync_or_build_engines()

    # 4. Default models
    download_default_models()

    # 5. Verify
    verify_script = ROOT / "scripts" / "verify_install.py"
    if verify_script.exists():
        subprocess.check_call([sys.executable, str(verify_script)])

    print("\n" + "=" * 80)
    print("SeedVR2 Video Upscaler (TensorRT) installation complete.")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
