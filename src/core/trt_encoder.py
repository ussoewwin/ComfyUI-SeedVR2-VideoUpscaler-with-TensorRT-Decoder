"""Dedicated Full-Batch TensorRT VAE encoder for ComfyUI SeedVR2.
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

_ENGINES: dict[str, tuple[object, object, object, str, str, torch.cuda.Stream]] = {}
_ENCODE_LOCK = Lock()


def find_engine_path(frames: int) -> Path | None:
    # Dedicated static engine for this exact frame count
    name = f"vae_encoder_{frames}f_tile512.rtxplan"
    for d in ARTIFACTS_DIRS:
        p = d / name
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    return None


def is_available(frames: int | None = None) -> bool:
    """Check if TensorRT VAE encoder is available."""
    if not HAS_TRT:
        return False
    return True


def _engine(frames: int, vae: torch.nn.Module | None = None, dit_model: str | None = None):
    cache_key = frames
    cached = _ENGINES.get(cache_key)
    if cached is not None:
        return cached

    path = find_engine_path(frames)
    if path is None:
        print(f"[SeedVR2 TensorRT] Dedicated {frames}f encoder engine not found. Building now...")
        from ..interfaces.trt_vae_model_loader import ensure_trt_engine_for_frames
        ensure_trt_engine_for_frames(frames, vae=vae)
        path = find_engine_path(frames)
        if path is None:
            raise FileNotFoundError(f"Failed to build TensorRT VAE encoder engine for {frames} frames")

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize TensorRT encoder: {path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"TensorRT could not create an execution context for {path}")

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    output_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    stream = torch.cuda.Stream()
    cached = (runtime, engine, context, input_name, output_name, stream)
    _ENGINES[cache_key] = cached
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
def _encode_single_chunk(sample: torch.Tensor, frames: int, vae: torch.nn.Module | None = None, dit_model: str | None = None) -> torch.Tensor:
    """Encode a single full batch directly in 1 shot with TensorRT."""
    _, _, _, height, width = sample.shape
    _, engine, _, input_name, output_name, stream = _engine(int(frames), vae=vae, dit_model=dit_model)
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT could not create a per-batch encoder context")

    # Set input shape for this exact batch size
    context.set_input_shape(input_name, (1, 3, frames, 512, 512))

    source = sample.to(device="cuda", dtype=torch.float16).contiguous()
    tile, overlap = 512, 96
    ys, xs = _positions(height, tile, overlap), _positions(width, tile, overlap)
    padded_h, padded_w = max(height, ys[-1] + tile), max(width, xs[-1] + tile)
    source = torch.nn.functional.pad(source, (0, padded_w - width, 0, padded_h - height))
    latent_frames = (frames - 1) // 4 + 1
    latent_h, latent_w = height // 8, width // 8
    raw_h, raw_w = padded_h // 8, padded_w // 8
    result = torch.zeros((1, 32, latent_frames, raw_h, raw_w), device="cuda", dtype=torch.float32)
    weights = torch.zeros_like(result)
    overlap_latent = overlap // 8

    with _ENCODE_LOCK, torch.cuda.stream(stream):
        for y in ys:
            for x in xs:
                tile_input = source[:, :, :, y:y + tile, x:x + tile].contiguous()
                tile_output = torch.empty((1, 32, latent_frames, 64, 64), device="cuda", dtype=torch.float16)
                context.set_tensor_address(input_name, tile_input.data_ptr())
                context.set_tensor_address(output_name, tile_output.data_ptr())
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError(f"TensorRT VAE encoder failed at tile y={y}, x={x}")
                stream.synchronize()
                ly, lx = y // 8, x // 8
                wy = _feather(64, overlap_latent, y != ys[0], y != ys[-1], tile_output.device)
                wx = _feather(64, overlap_latent, x != xs[0], x != xs[-1], tile_output.device)
                window = (wy[:, None] * wx[None, :]).view(1, 1, 1, 64, 64)
                result[:, :, :, ly:ly + 64, lx:lx + 64] += tile_output.float() * window
                weights[:, :, :, ly:ly + 64, lx:lx + 64] += window

    encoded = (result / weights.clamp_min(1e-6))[:, :16, :, :latent_h, :latent_w].to(sample.dtype)
    del context
    return encoded


def _pick_engine_frames(total_frames: int, preferred: str = "auto") -> int | None:
    """Pick the engine frame size. preferred (from the loader dropdown) wins if its engine exists."""
    if preferred != "auto":
        try:
            cand = int(preferred)
            if find_engine_path(cand) is not None:
                return cand
        except ValueError:
            pass
    for cand in (total_frames, 29, 21, 5):
        if find_engine_path(cand) is not None:
            return cand
    return None


@torch.inference_mode()
def _encode_chunked(sample: torch.Tensor, total_frames: int, engine_frames: int, vae: torch.nn.Module | None = None, dit_model: str | None = None) -> torch.Tensor:
    """Encode a long clip by splitting it into engine_frames chunks with 4-frame temporal overlap."""
    _, _, _, height, width = sample.shape
    lat_total = (total_frames - 1) // 4 + 1
    lat_engine = (engine_frames - 1) // 4 + 1
    stride = engine_frames - 4  # 4-frame overlap -> 1 latent-frame overlap
    lat_h, lat_w = height // 8, width // 8
    result = torch.zeros((1, 16, lat_total, lat_h, lat_w), device="cuda", dtype=sample.dtype)
    starts = list(range(0, total_frames - engine_frames + 1, stride))
    if starts[-1] != total_frames - engine_frames:
        starts.append(total_frames - engine_frames)
    for start in starts:
        chunk = sample[:, :, start:start + engine_frames]
        lat = _encode_single_chunk(chunk, engine_frames, vae=vae, dit_model=dit_model)
        lat_start = start // 4
        result[:, :, lat_start:lat_start + lat_engine] = lat
    return result


@torch.inference_mode()
def encode(sample: torch.Tensor, vae: torch.nn.Module | None = None, dit_model: str | None = None, engine_frames: str = "auto") -> torch.Tensor:
    """
    Encode [B,3,T,H,W] to posterior mean [B,16,(T-1)/4+1,H/8,W/8] in 1 shot using TensorRT engine.
    """
    if sample.ndim != 5 or sample.shape[0] != 1 or sample.shape[1] != 3:
        raise ValueError(f"TensorRT encoder expects [1,3,T,H,W], got {tuple(sample.shape)}")
    _, _, total_frames, height, width = sample.shape
    if height % 8 or width % 8:
        raise ValueError("TensorRT encoder input dimensions must be divisible by 8")

    # Ensure 4n+1
    req_frames = ((total_frames - 1) // 4) * 4 + 1
    if total_frames != req_frames:
        pad_len = req_frames - total_frames
        last_frame = sample[:, :, -1:, :, :].repeat(1, 1, pad_len, 1, 1)
        sample = torch.cat([sample, last_frame], dim=2)
        total_frames = req_frames

    engine_frames = _pick_engine_frames(total_frames, engine_frames)
    if engine_frames is None:
        raise FileNotFoundError("No TensorRT VAE encoder engine found (need vae_encoder_{5,21,29}f_tile512.rtxplan)")
    if engine_frames == total_frames:
        print(f"[SeedVR2 TensorRT] 1-Shot Directly encoding {total_frames} frames with dedicated {total_frames}f TensorRT engine...")
        return _encode_single_chunk(sample, total_frames, vae=vae, dit_model=dit_model)
    n_chunks = (total_frames + engine_frames - 5) // (engine_frames - 4)
    print(f"[SeedVR2 TensorRT] Encoding {total_frames} frames in {n_chunks} chunks of {engine_frames}f (4-frame temporal overlap)...")
    return _encode_chunked(sample, total_frames, engine_frames, vae=vae, dit_model=dit_model)


def release() -> None:
    """Clear cached execution contexts and streams to free GPU VRAM."""
    global _ENGINES
    _ENGINES.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()