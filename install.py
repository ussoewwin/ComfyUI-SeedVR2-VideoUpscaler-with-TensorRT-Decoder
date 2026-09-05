"""
ComfyUI-SeedVR2_VideoUpscaler Automated Installer
Installs requirements.txt, TensorRT RTX stack, Fast Attention wheels (FlashAttention 2, SageAttention 2), and verifies setup.
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

# Known wheels by Python version from ussoewwin Hugging Face repos
FLASH_ATTN_WHEELS = {
    "cp311": "https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.8.3%2Bcu130torch2.9.1cxx11abiTRUE-cp311-cp311-win_amd64.whl",
    "cp312": "https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.9.1%2Bcu132torch2.13.0cxx11abiTRUE-cp312-cp312-win_amd64.whl",
    "cp313": "https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.9.1%2Bcu132torch2.13.0cxx11abiTRUE-cp313-cp313-win_amd64.whl",
    "cp314": "https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.9.1%2Bcu132torch2.13.0cxx11abiTRUE-cp314-cp314-win_amd64.whl",
}

SAGE_ATTN_WHEELS = {
    "cp312": "https://huggingface.co/ussoewwin/Sage-Attention-for-Windows/resolve/main/sageattention-2.2.0.post6%2Bcu132torch2.13.0-cp312-cp312-win_amd64.whl",
    "cp313": "https://huggingface.co/ussoewwin/Sage-Attention-for-Windows/resolve/main/sageattention-2.2.0.post6%2Bcu132torch2.13.0-cp313-cp313-win_amd64.whl",
    "cp314": "https://huggingface.co/ussoewwin/Sage-Attention-for-Windows/resolve/main/sageattention-2.2.0.post6%2Bcu132torch2.13.0-cp314-cp314-win_amd64.whl",
}

ENGINES = [
    "vae_encoder_5f_tile512.rtxplan",
    "vae_encoder_21f_tile512.rtxplan",
    "vae_decoder_tile_512_5f.rtxplan",
    "vae_decoder_tile_256_21f.rtxplan",
]


def log_step(message: str) -> None:
    print(f"\n[SeedVR2 TensorRT Installer] == {message} ==")


def ensure_ffmpeg_path() -> None:
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
            break


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


def install_wheels() -> None:
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    print(f"[SeedVR2] Detected Python version: {sys.version.split()[0]} ({py_tag})")

    wheel_entries = []
    if py_tag in FLASH_ATTN_WHEELS:
        wheel_entries.append(("flash_attn", "FlashAttention 2", FLASH_ATTN_WHEELS[py_tag]))
    else:
        print(f"[SeedVR2] No prebuilt FlashAttention 2 wheel for {py_tag}. SeedVR2 will use standard attention.")

    if py_tag in SAGE_ATTN_WHEELS:
        wheel_entries.append(("sageattention", "SageAttention 2", SAGE_ATTN_WHEELS[py_tag]))
    else:
        print(f"[SeedVR2] No prebuilt SageAttention 2 wheel for {py_tag}. SeedVR2 will use standard attention.")

    for mod, name, url in wheel_entries:
        try:
            __import__(mod)
            print(f"[SeedVR2] {name} is already installed.")
            continue
        except ImportError:
            pass

        filename = urllib.parse.unquote(Path(urllib.parse.urlparse(url).path).name)
        cached_file = WHEELS_DIR / filename
        if not cached_file.exists() or cached_file.stat().st_size < 1000:
            log_step(f"Downloading {name} wheel ({py_tag})")
            print(f"URL: {url}")
            try:
                urllib.request.urlretrieve(url, cached_file)
            except Exception as exc:
                print(f"[SeedVR2] Warning: Failed to download {name} wheel: {exc}")
                continue

        log_step(f"Installing {name}")
        try:
            pip_install([str(cached_file), "--no-deps"])
            print(f"[SeedVR2] Successfully installed {name}.")
        except Exception as exc:
            print(f"[SeedVR2] Warning: Failed to install {name} wheel: {exc}")
            print(f"[SeedVR2] SeedVR2 will safely fall back to standard attention.")


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

    # 3. FlashAttention 2 & SageAttention 2 wheels (version-matched)
    install_wheels()

    # 4. TensorRT RTX VAE Engines
    sync_or_build_engines()

    # 5. Default models
    download_default_models()

    # 6. Verify
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
