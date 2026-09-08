"""Fast, non-rendering readiness check for ComfyUI SeedVR2 TensorRT."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _ensure_ffmpeg_path():
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


_ensure_ffmpeg_path()

REQUIRED_ENGINES = (
    "vae_decoder_tile_256_21f.rtxplan",
    "vae_decoder_tile_512_5f.rtxplan",
    "vae_encoder_5f_tile512.rtxplan",
    "vae_encoder_21f_tile512.rtxplan",
)

SEARCH_DIRS = [
    ROOT / "tensorrt_backend" / "artifacts",
    ROOT.parents[1] / "models" / "tensorrt" / "seedvr2",
]


def main() -> int:
    _ensure_ffmpeg_path()
    failures: list[str] = []

    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            failures.append(f"{executable} is not on PATH")

    for module in ("torch", "tensorrt_rtx", "onnx"):
        try:
            importlib.import_module(module)
        except Exception as exc:
            failures.append(f"cannot import {module}: {exc}")

    try:
        import torch
        if not torch.cuda.is_available():
            failures.append("PyTorch cannot access an NVIDIA CUDA GPU")
        else:
            print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    try:
        from src.optimization.compatibility import SAGE_ATTN_2_AVAILABLE, FLASH_ATTN_2_AVAILABLE
        print(f"SageAttention 2: {'ready' if SAGE_ATTN_2_AVAILABLE else 'not available (using SDPA)'}")
        print(f"FlashAttention 2: {'ready' if FLASH_ATTN_2_AVAILABLE else 'not available (using SDPA)'}")
    except Exception:
        pass

    missing_engines = []
    for name in REQUIRED_ENGINES:
        found = False
        for s_dir in SEARCH_DIRS:
            p = s_dir / name
            if p.exists() and p.stat().st_size > 1_000_000:
                found = True
                break
        if not found:
            missing_engines.append(name)

    if missing_engines:
        print("\nNote: Some TensorRT RTX VAE engines are not yet present:")
        for name in missing_engines:
            print(f" - {name}")
        print("Engines can be built on demand in ComfyUI using the 'SeedVR2 Build TensorRT VAE Engines' node.")

    if failures:
        print("\nInstallation is incomplete:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("\nComfyUI SeedVR2 TensorRT installation is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
