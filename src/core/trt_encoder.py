"""Dedicated Full-Batch TensorRT VAE encoder for ComfyUI SeedVR2.
Executes exact 1-shot TensorRT acceleration for ANY batch size.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import torch

# --- TRT encode debug hooks (SEEDVR2_TRT_DEBUG=1 to enable; default off) ---
_TRT_DEBUG = os.environ.get("SEEDVR2_TRT_DEBUG", "0") == "1"
_TRT_DEBUG_DIR = os.environ.get("SEEDVR2_TRT_DEBUG_DIR", "") or None

def _trt_dbg_log(msg):
    if _TRT_DEBUG:
        print(f"[TRT-DEBUG] {msg}", flush=True)

def _trt_dbg_stats(tag, t):
    if not _TRT_DEBUG:
        return
    tt = t.detach().float()
    _trt_dbg_log(f"{tag}: shape={tuple(tt.shape)} dtype={t.dtype} "
                f"min={float(tt.min()):.4f} max={float(tt.max()):.4f} "
                f"mean={float(tt.mean()):.4f} std={float(tt.std()):.4f} "
                f"NaN={bool(torch.isnan(tt).any())} Inf={bool(torch.isinf(tt).any())}")
    if _TRT_DEBUG_DIR:
        try:
            import os as _os
            torch.save(tt.cpu(), _os.path.join(_TRT_DEBUG_DIR, tag.replace(' ', '_') + '.pt'))
        except Exception as e:
            _trt_dbg_log(f"save {tag} failed: {e}")

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


def find_engine_path(frames: int) -> tuple[Path | None, int]:
    """Return (engine_path, tile_px). Prefers the 256px-tile engine, then 512px."""
    for tile_px in (256, 512):
        name = f"vae_encoder_{frames}f_tile{tile_px}.rtxplan"
        for d in ARTIFACTS_DIRS:
            p = d / name
            if p.exists() and p.stat().st_size > 1_000_000:
                return p, tile_px
    return None, 0


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

    path, tile_px = find_engine_path(frames)
    if path is None:
        # No auto-build: engines are created explicitly via the build scripts/node.
        # Without an engine we fall back to the standard PyTorch VAE.
        raise FileNotFoundError(
            f"TensorRT VAE encoder engine for {frames} frames not found. "
            f"Build it first with tools/cloud_export_gpu.py + tools/cloud_build_engine.py "
            f"or the SeedVR2 Build TensorRT VAE Engines node."
        )

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
    cached = (runtime, engine, context, input_name, output_name, stream, tile_px)
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
    if overlap:
        t = torch.linspace(0.0, 1.0, overlap + 1, device=device)[1:]
        ramp = (1.0 - torch.cos(t * 3.141592653589793)) / 2.0  # cosine ease
        if left:
            weight[:overlap] = ramp
        if right:
            weight[-overlap:] = torch.minimum(weight[-overlap:], torch.flip(ramp, dims=[0]))
    return weight


@torch.inference_mode()
def _encode_single_chunk(sample: torch.Tensor, frames: int, vae: torch.nn.Module | None = None, dit_model: str | None = None) -> torch.Tensor:
    """Encode a single full batch directly in 1 shot with TensorRT."""
    _, _, _, height, width = sample.shape
    _, _, context, input_name, output_name, stream, tile_px = _engine(int(frames), vae=vae, dit_model=dit_model)
    if context is None:
        raise RuntimeError("TensorRT could not create a per-batch encoder context")

    # Set input shape to the engine's tile size (256px or 512px)
    context.set_input_shape(input_name, (1, 3, frames, tile_px, tile_px))
    # Barrier: TRT may (de)allocate internal buffers asynchronously after
    # set_input_shape. Without a sync, the FIRST tile of a batch intermittently
    # reads a half-initialized buffer -> NaN (always the first tile y=0/x=0).
    torch.cuda.synchronize()

    source = sample.to(device="cuda", dtype=torch.float16).contiguous()
    # Wide overlap (96px on 256px tiles = 37.5%) to keep the tile-edge zero-padding
    # influence out of the blended region. Small tiles make the receptive-field
    # edge effect proportionally larger, so 256px needs a wider overlap than 512px.
    tile, overlap = tile_px, tile_px * 3 // 8  # 37.5% tile-to-tile overlap (96px@256, 192px@512)
    # Outer pad = half a tile: places each image corner at the CENTER of its
    # corner tile, so the receptive-field-poor tile edges and the replicated
    # padding stay away from real image content (fixes the top-left blur/noise).
    pad = tile_px // 2
    source = torch.nn.functional.pad(source, (pad, pad, pad, pad, 0, 0), mode="replicate")
    height_p, width_p = height + 2 * pad, width + 2 * pad
    ys, xs = _positions(height_p, tile, overlap), _positions(width_p, tile, overlap)
    padded_h, padded_w = max(height_p, ys[-1] + tile), max(width_p, xs[-1] + tile)
    source = torch.nn.functional.pad(source, (0, padded_w - width_p, 0, padded_h - height_p))
    latent_frames = (frames - 1) // 4 + 1
    latent_h, latent_w = height // 8, width // 8
    raw_h, raw_w = padded_h // 8, padded_w // 8
    result = torch.zeros((1, 32, latent_frames, raw_h, raw_w), device="cuda", dtype=torch.float32)
    weights = torch.zeros_like(result)
    dc_result = torch.zeros((1, 32, latent_frames, raw_h, raw_w), device="cuda", dtype=torch.float32)
    overlap_latent = overlap // 8
    offset_latent = pad // 8

    with _ENCODE_LOCK, torch.cuda.stream(stream):
        for y in ys:
            for x in xs:
                tile_input = source[:, :, :, y:y + tile, x:x + tile].contiguous()
                tile_lat = tile_px // 8
                tile_output = torch.zeros((1, 32, latent_frames, tile_lat, tile_lat), device="cuda", dtype=torch.float16)
                context.set_tensor_address(input_name, tile_input.data_ptr())
                context.set_tensor_address(output_name, tile_output.data_ptr())
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError(f"TensorRT VAE encoder failed at tile y={y}, x={x}")
                stream.synchronize()
                if _TRT_DEBUG:
                    _dbg_tv = tile_output.float()
                    _dbg_sd = float(_dbg_tv.std())
                    _trt_dbg_log(f"enc tile y={y} x={x} (ly={y // 8},lx={x // 8}) "
                                f"min={float(_dbg_tv.min()):.4f} max={float(_dbg_tv.max()):.4f} "
                                f"std={_dbg_sd:.5f}" + ("  <<< BLACK?" if _dbg_sd < 0.05 else ""))
                ly, lx = y // 8, x // 8
                # DC offset correction: estimate the tile's true DC from its
                # accurate center (inside the receptive-field-poor edge ring),
                # subtract it, and restore it later as a weighted average.
                edge = overlap_latent // 2
                center = tile_output[:, :, :, edge:tile_lat - edge, edge:tile_lat - edge]
                dc = center.mean(dim=(3, 4), keepdim=True)
                corrected = tile_output.float() - dc.float()
                wy = _feather(tile_lat, overlap_latent, y != ys[0], y != ys[-1], tile_output.device)
                wx = _feather(tile_lat, overlap_latent, x != xs[0], x != xs[-1], tile_output.device)
                window = (wy[:, None] * wx[None, :]).view(1, 1, 1, tile_lat, tile_lat)
                result[:, :, :, ly:ly + tile_lat, lx:lx + tile_lat] += corrected * window
                dc_result[:, :, :, ly:ly + tile_lat, lx:lx + tile_lat] += dc.float() * window
                weights[:, :, :, ly:ly + tile_lat, lx:lx + tile_lat] += window

    restored = (result + dc_result) / weights.clamp_min(1e-6)
    encoded = restored[:, :16, :, offset_latent:offset_latent + latent_h, offset_latent:offset_latent + latent_w].to(sample.dtype)
    if _TRT_DEBUG:
        _trt_dbg_stats(f"enc_chunk_out_{frames}f", encoded)
    return encoded


def _pick_engine_frames(total_frames: int, preferred: str = "auto") -> int | None:
    """Pick the engine frame size. preferred (from the loader dropdown) wins if its engine exists."""
    if preferred != "auto":
        try:
            cand = int(preferred)
            if find_engine_path(cand)[0] is not None:
                return cand
        except ValueError:
            pass
    for cand in (total_frames, 29, 21, 5):
        if find_engine_path(cand)[0] is not None:
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

    # Release cached-but-unused VRAM from previous batches/other nodes to avoid
    # allocator pressure during 512px-tile engine execution (NaN source).
    torch.cuda.empty_cache()

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
        print(f"[SeedVR2 TensorRT] Encoding {engine_frames}f in 1 shot with dedicated {engine_frames}f TensorRT engine...")
        return _encode_single_chunk(sample, total_frames, vae=vae, dit_model=dit_model)
    n_chunks = (total_frames + engine_frames - 5) // (engine_frames - 4)
    print(f"[SeedVR2 TensorRT] Encoding {n_chunks} chunks of {engine_frames}f with TensorRT engine (4-frame temporal overlap)...")
    return _encode_chunked(sample, total_frames, engine_frames, vae=vae, dit_model=dit_model)


def resolve_engine_frames(preferred: str = "auto") -> int | None:
    """Return the largest available encoder engine frame size (for chunking)."""
    return _pick_engine_frames(29, preferred)


def release() -> None:
    """Clear cached execution contexts and streams to free GPU VRAM."""
    global _ENGINES
    _ENGINES.clear()
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _gc.collect()
        torch.cuda.empty_cache()
