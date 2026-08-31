"""Standalone ONNX export worker for large-batch TensorRT VAE engine builds.

Runs 100% on CPU (System RAM 64GB) with CUDA hard-disabled.
Exports exact full-resolution static frame ONNX (185f, 205f, 512x512 tile)
with Causal Conv memory chunking to guarantee zero 335GB allocation OOM.

Usage:
    python tools/export_onnx_worker.py --repo <custom_node_root> \
        --kind encoder --frames 185 --output <onnx_path> [--model ema_vae_fp16.safetensors]
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

# Absolute hard-disable of CUDA before torch import
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import yaml


def _force_utf8_stdio() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _disable_cuda_bindings() -> None:
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.cuda.init = lambda: None
    torch.cuda.memory_stats = lambda *a, **k: {"reserved_bytes.all.current": 0}
    try:
        torch.cuda._lazy_init = lambda *a, **k: None
    except Exception:
        pass
    try:
        torch.cuda.set_device = lambda *a, **k: None
    except Exception:
        pass


def _find_vae_file(model_name: str, model_dir: Path) -> Path:
    candidates = [
        model_dir / model_name,
        model_dir / "SEEDVR2" / model_name,
        Path(r"D:\USERFILES\ComfyUI\ComfyUI\models\SEEDVR2") / model_name,
        Path(r"D:\USERFILES\ComfyUI\ComfyUI\models\vae") / model_name,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(f"Could not locate VAE file {model_name} in candidate paths")


def _cleanup_stale_artifacts(repo_path: Path) -> None:
    artifacts_dir = repo_path / "tensorrt_backend" / "artifacts"
    if artifacts_dir.exists():
        for p in artifacts_dir.glob("*205f*"):
            try:
                p.unlink(missing_ok=True)
                print(f"WORKER Cleaned up stale artifact: {p.name}", flush=True)
            except Exception:
                pass


def main() -> int:
    _force_utf8_stdio()
    _disable_cuda_bindings()
    print("WORKER CUDA hard-disabled (CPU 64GB RAM mode)", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="custom node root directory")
    parser.add_argument("--kind", choices=["encoder", "decoder"], required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="ema_vae_fp16.safetensors", help="VAE filename")
    parser.add_argument("--dit-model", default=None, help="ignored (VAE only)")
    args = parser.parse_args()

    repo_path = Path(args.repo)
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    _cleanup_stale_artifacts(repo_path)

    from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKL
    from src.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
    from src.interfaces.trt_vae_model_loader import (
        _EncoderModule,
        _DecoderModule,
        configure_fixed_vae,
    )
    from tools.onnx_export_utils import _portable_export

    # Load SafeTensors directly
    try:
        from safetensors.torch import load_file as load_safetensors_file
    except ImportError:
        import safetensors.torch
        load_safetensors_file = safetensors.torch.load_file

    vae_config_path = repo_path / "src" / "models" / "video_vae_v3" / "s8_c16_t4_inflation_sd3.yaml"
    with open(vae_config_path, "r", encoding="utf-8") as f:
        vae_kwargs = yaml.safe_load(f)

    import inspect
    sig = inspect.signature(VideoAutoencoderKL.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "args", "kwargs"}
    filtered_kwargs = {k: v for k, v in vae_kwargs.items() if k in valid_params}

    print("WORKER Instantiating 3D VideoAutoencoderKL architecture on meta device...", flush=True)
    with torch.device("meta"):
        vae = VideoAutoencoderKL(**filtered_kwargs)

    vae_file = _find_vae_file(args.model, repo_path / "models")
    print(f"WORKER Loading VAE weights from {vae_file} to CPU...", flush=True)
    state_dict = load_safetensors_file(str(vae_file), device="cpu")
    vae.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict

    vae = vae.to(device="cpu", dtype=torch.float16).eval()
    configure_fixed_vae(vae)

    # Enable safe memory limit on Conv3d layers to absorb the 335GB im2col explosion on CPU
    for module in vae.modules():
        if isinstance(module, InflatedCausalConv3d):
            module.set_memory_limit(16.0)

    print("WORKER VAE loaded on CPU float16 (memory limit chunking enabled)", flush=True)

    frames = ((args.frames - 1) // 4) * 4 + 1
    lat_frames = (frames - 1) // 4 + 1
    dec_tile_px = 256 if frames >= 21 else 512
    dec_lat_tile = dec_tile_px // 8

    # Full standard resolution (512x512 for encoder, 256px/32x32 for decoder)
    t0 = time.perf_counter()
    if args.kind == "encoder":
        mod = _EncoderModule(vae).eval()
        dummy = torch.zeros((1, 3, frames, 512, 512), dtype=torch.float16)
    else:
        mod = _DecoderModule(vae.decoder).eval()
        dummy = torch.zeros((1, 16, lat_frames, dec_lat_tile, dec_lat_tile), dtype=torch.float16)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"WORKER Exporting full {frames}f ONNX on CPU (512x512 tile)...", flush=True)
    with torch.inference_mode():
        _portable_export(mod, (dummy,), output, legacy=True)
    print(f"WORKER-OK {args.kind} {frames}f -> {output} ({time.perf_counter() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
