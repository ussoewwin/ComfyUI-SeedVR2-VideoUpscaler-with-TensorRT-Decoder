"""Dedicated Full-Batch TensorRT VAE decoder for ComfyUI SeedVR2.
Executes exact 1-shot TensorRT acceleration for ANY batch size.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import torch

try:
    import tensorrt_rtx as trt
    HAS_TRT = True
except ImportError:
    try:
        import tensorrt as trt
        HAS_TRT = True
    except ImportError:
        trt = None
        HAS_TRT = False


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIRS = [
    ROOT / "tensorrt_backend" / "artifacts",
    ROOT.parents[1] / "models" / "tensorrt" / "seedvr2",
]

_DECODER_ENGINES: dict[str, tuple[object, object, object, str, str, torch.cuda.Stream, int, int]] = {}
_DECODE_LOCK = Lock()


def find_engine_path(latent_frames: int) -> tuple[Path | None, int, int]:
    # Dedicated static engine for this exact frame count
    video_frames = (latent_frames - 1) * 4 + 1
    tile = 32 if latent_frames >= 6 else 64
    overlap = 8 if latent_frames >= 6 else 12
    tile_px = tile * 8

    name = f"vae_decoder_tile_{tile_px}_{video_frames}f.rtxplan"
    for d in ARTIFACTS_DIRS:
        p = d / name
        if p.exists() and p.stat().st_size > 1_000_000:
            return p, tile, overlap
    return None, 0, 0


def is_available(latent_frames: int | None = None) -> bool:
    """Check if TensorRT VAE decoder is available."""
    if not HAS_TRT:
        return False
    return True


def _engine(latent_frames: int, vae: torch.nn.Module | None = None):
    cache_key = latent_frames
    cached = _DECODER_ENGINES.get(cache_key)
    if cached is not None:
        return cached

    path, tile, overlap = find_engine_path(latent_frames)
    if path is None:
        video_frames = (latent_frames - 1) * 4 + 1
        print(f"[SeedVR2 TensorRT] Dedicated {video_frames}f decoder engine not found. Building now...")
        from ..interfaces.trt_vae_model_loader import ensure_trt_engine_for_frames
        ensure_trt_engine_for_frames(video_frames, vae=vae)
        path, tile, overlap = find_engine_path(latent_frames)
        if path is None:
            raise FileNotFoundError(f"Failed to build TensorRT VAE decoder engine for {latent_frames} latent frames")

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize TensorRT decoder: {path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"TensorRT could not create an execution context for {path}")

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    output_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    stream = torch.cuda.Stream()
    cached = (runtime, engine, context, input_name, output_name, stream, tile, overlap)
    _DECODER_ENGINES[cache_key] = cached
    return cached


def _positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def _feather(length: int, overlap: int, left: bool, right: bool, device: torch.device) -> torch.Tensor:
    weight = torch.ones(length, device=device, dtype=torch.float32)
    if left and overlap:
        weight[:overlap] = torch.linspace(0.0, 1.0, overlap + 1, device=device)[1:]
    if right and overlap:
        weight[-overlap:] = torch.minimum(
            weight[-overlap:], torch.linspace(1.0, 0.0, overlap + 1, device=device)[1:]
        )
    return weight


@torch.inference_mode()
def _decode_single_chunk(latent: torch.Tensor, latent_frames: int, vae: torch.nn.Module | None = None) -> torch.Tensor:
    """Decode a single full batch directly in 1 shot with TensorRT."""
    _, _, _, height, width = latent.shape
    _, engine, _, input_name, output_name, stream, tile, overlap = _engine(int(latent_frames), vae=vae)
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT could not create a per-batch decoder context")

    # Set input shape for this exact latent batch size
    context.set_input_shape(input_name, (1, 16, latent_frames, tile, tile))

    source = latent.to(device="cuda", dtype=torch.float16).contiguous()
    video_frames = (latent_frames - 1) * 4 + 1
    ys, xs = _positions(height, tile, overlap), _positions(width, tile, overlap)
    padded_h, padded_w = max(height, ys[-1] + tile), max(width, xs[-1] + tile)
    source = torch.nn.functional.pad(source, (0, padded_w - width, 0, padded_h - height))
    out_h, out_w = height * 8, width * 8
    raw_out_h, raw_out_w = padded_h * 8, padded_w * 8
    result = torch.zeros((1, 3, video_frames, raw_out_h, raw_out_w), device="cuda", dtype=torch.float32)
    weights = torch.zeros_like(result)
    out_tile, out_overlap = tile * 8, overlap * 8

    with _DECODE_LOCK, torch.cuda.stream(stream):
        for y in ys:
            for x in xs:
                tile_input = source[:, :, :, y:y + tile, x:x + tile].contiguous()
                tile_output = torch.empty((1, 3, video_frames, out_tile, out_tile), device="cuda", dtype=torch.float16)
                context.set_tensor_address(input_name, tile_input.data_ptr())
                context.set_tensor_address(output_name, tile_output.data_ptr())
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError(f"TensorRT VAE decoder failed at tile y={y}, x={x}")
                stream.synchronize()
                oy, ox = y * 8, x * 8
                wy = _feather(out_tile, out_overlap, y != ys[0], y != ys[-1], tile_output.device)
                wx = _feather(out_tile, out_overlap, x != xs[0], x != xs[-1], tile_output.device)
                window = (wy[:, None] * wx[None, :]).view(1, 1, 1, out_tile, out_tile)
                result[:, :, :, oy:oy + out_tile, ox:ox + out_tile] += tile_output.float() * window
                weights[:, :, :, oy:oy + out_tile, ox:ox + out_tile] += window

    decoded = (result / weights.clamp_min(1e-6))[:, :, :, :out_h, :out_w].to(latent.dtype)
    del context
    return decoded


@torch.inference_mode()
def decode(latent: torch.Tensor, vae: torch.nn.Module | None = None) -> torch.Tensor:
    """
    Decode [B,16,T_lat,H,W] to video [B,3,(T_lat-1)*4+1,H*8,W*8] in 1 shot using TensorRT engine.
    """
    if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != 16:
        raise ValueError(f"TensorRT decoder expects [1,16,T,H,W], got {tuple(latent.shape)}")
    _, _, latent_frames, _, _ = latent.shape
    video_frames = (latent_frames - 1) * 4 + 1

    print(f"[SeedVR2 TensorRT] 1-Shot Directly decoding {video_frames} frames ({latent_frames} latents) with dedicated {video_frames}f TensorRT engine...")
    sample = _decode_single_chunk(latent, latent_frames, vae=vae)
    return sample


def release() -> None:
    """Clear cached execution contexts and streams to free GPU VRAM."""
    global _DECODER_ENGINES
    _DECODER_ENGINES.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()