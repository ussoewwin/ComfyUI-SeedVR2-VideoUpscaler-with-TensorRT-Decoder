"""Dedicated Full-Batch TensorRT VAE decoder for ComfyUI SeedVR2.
Executes exact 1-shot TensorRT acceleration for ANY batch size.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from threading import Lock

import torch

# --- TRT decode debug hooks (SEEDVR2_TRT_DEBUG=1 to enable; default off) ---
_TRT_DEBUG = os.environ.get("SEEDVR2_TRT_DEBUG", "0") == "1"
_TRT_DEBUG_DIR = os.environ.get("SEEDVR2_TRT_DEBUG_DIR", "") or None

def _trt_dbg_log(msg):
    if _TRT_DEBUG:
        print(f"[TRT-DEBUG] {msg}", flush=True)

def _trt_dbg_stats(tag, t):
    """Log + save tensor stats (input latents / outputs)."""
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

_DECODER_ENGINES: dict[str, tuple[object, object, object, str, str, torch.cuda.Stream, int, int]] = {}
_DECODE_LOCK = Lock()


def find_engine_path(latent_frames: int) -> tuple[Path | None, int, int]:
    # Prefer the 512px-tile decoder (matches the encoder tile), then 256px.
    video_frames = (latent_frames - 1) * 4 + 1
    # Studio-compatible overlaps: 512px tile -> 24 latent px, 256px tile -> 12 latent px.
    for tile in (64, 32):
        overlap = 24 if tile == 64 else 12
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


def _engine(latent_frames: int, vae: torch.nn.Module | None = None, dit_model: str | None = None):
    cache_key = latent_frames
    cached = _DECODER_ENGINES.get(cache_key)
    if cached is not None:
        return cached

    path, tile, overlap = find_engine_path(latent_frames)
    if path is None:
        # No auto-build: engines are created explicitly via the build scripts/node.
        video_frames = (latent_frames - 1) * 4 + 1
        raise FileNotFoundError(
            f"TensorRT VAE decoder engine for {video_frames} frames not found. "
            f"Build it first with tools/cloud_export_gpu.py + tools/cloud_build_engine.py "
            f"or the SeedVR2 Build TensorRT VAE Engines node."
        )

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
def _decode_single_chunk(latent: torch.Tensor, latent_frames: int, vae: torch.nn.Module | None = None, dit_model: str | None = None) -> torch.Tensor:
    """Decode a single full batch directly in 1 shot with TensorRT."""
    _, _, _, height, width = latent.shape
    _, _, context, input_name, output_name, stream, tile, overlap = _engine(int(latent_frames), vae=vae, dit_model=dit_model)
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
                tile_output = torch.zeros((1, 3, video_frames, out_tile, out_tile), device="cuda", dtype=torch.float16)
                context.set_tensor_address(input_name, tile_input.data_ptr())
                context.set_tensor_address(output_name, tile_output.data_ptr())
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError(f"TensorRT VAE decoder failed at tile y={y}, x={x}")
                stream.synchronize()
                if _TRT_DEBUG:
                    _dbg_tv = tile_output.float()
                    _dbg_sd = float(_dbg_tv.std())
                    _trt_dbg_log(f"tile y={y} x={x} (oy={y * 8},ox={x * 8}) "
                                f"min={float(_dbg_tv.min()):.4f} max={float(_dbg_tv.max()):.4f} "
                                f"std={_dbg_sd:.5f}" + ("  <<< BLACK?" if _dbg_sd < 0.05 else ""))
                oy, ox = y * 8, x * 8
                wy = _feather(out_tile, out_overlap, y != ys[0], y != ys[-1], tile_output.device)
                wx = _feather(out_tile, out_overlap, x != xs[0], x != xs[-1], tile_output.device)
                window = (wy[:, None] * wx[None, :]).view(1, 1, 1, out_tile, out_tile)
                result[:, :, :, oy:oy + out_tile, ox:ox + out_tile] += tile_output.float() * window
                weights[:, :, :, oy:oy + out_tile, ox:ox + out_tile] += window

    decoded = (result / weights.clamp_min(1e-6)).clamp(-2.0, 2.0)[:, :, :, :out_h, :out_w].to(latent.dtype)
    if _TRT_DEBUG:
        _trt_dbg_stats(f"chunk_out_{video_frames}f", decoded)
    return decoded


_ENGINE_FILE_RE = re.compile(r"^vae_decoder_tile_\d+_(\d+)f\.rtxplan$")


def _available_engine_frames() -> list[int]:
    """Return the sorted video-frame sizes of every usable decoder engine on disk."""
    found: set[int] = set()
    for d in ARTIFACTS_DIRS:
        try:
            if not d.is_dir():
                continue
            for p in d.iterdir():
                m = _ENGINE_FILE_RE.match(p.name)
                if m and p.is_file():
                    try:
                        if p.stat().st_size > 1_000_000:
                            found.add(int(m.group(1)))
                    except OSError:
                        continue
        except OSError:
            continue
    return sorted(found)


def pick_engine_frames(video_frames: int, preferred: str = "auto") -> int | None:
    """Pick the decoder engine frame size for a video of `video_frames` frames.

    Selection order:
    1. preferred (from the loader dropdown / settings node) if that engine exists;
    2. an engine matching video_frames exactly (1-shot decode);
    3. the largest engine that fits inside video_frames (chunked decode);
    4. if every engine is larger than the clip, the smallest engine (decode pads/crops);
    5. None only when no engine exists at all (the sole legitimate fallback case).
    """
    engines = _available_engine_frames()
    if not engines:
        return None
    if preferred != "auto":
        try:
            cand = int(preferred)
            if cand in engines:
                return cand
        except ValueError:
            pass
    if video_frames in engines:
        return video_frames
    fits = [e for e in engines if e <= video_frames]
    if fits:
        return fits[-1]
    return engines[0]


@torch.inference_mode()
def _decode_chunked(latent: torch.Tensor, latent_frames: int, engine_video_frames: int, vae: torch.nn.Module | None = None, dit_model: str | None = None) -> torch.Tensor:
    """Decode a long latent by splitting it into engine-sized chunks with 1 latent-frame overlap."""
    engine_latent = (engine_video_frames - 1) // 4 + 1
    lat_stride = engine_latent - 1
    _, _, _, lat_h, lat_w = latent.shape
    out_frames = (latent_frames - 1) * 4 + 1
    out_h, out_w = lat_h * 8, lat_w * 8
    result = torch.zeros((1, 3, out_frames, out_h, out_w), device="cuda", dtype=latent.dtype)
    starts = list(range(0, latent_frames - engine_latent + 1, lat_stride))
    if starts[-1] != latent_frames - engine_latent:
        starts.append(latent_frames - engine_latent)
    for start in starts:
        chunk = latent[:, :, start:start + engine_latent]
        sample = _decode_single_chunk(chunk, engine_latent, vae=vae, dit_model=dit_model)
        out_start = start * 4
        result[:, :, out_start:out_start + engine_video_frames] = sample
    return result


@torch.inference_mode()
def decode(latent: torch.Tensor, vae: torch.nn.Module | None = None, dit_model: str | None = None, engine_frames: str = "auto") -> torch.Tensor:
    """
    Decode [B,16,T_lat,H,W] to video [B,3,(T_lat-1)*4+1,H*8,W*8] in 1 shot using TensorRT engine.
    """
    if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != 16:
        raise ValueError(f"TensorRT decoder expects [1,16,T,H,W], got {tuple(latent.shape)}")
    _, _, latent_frames, _, _ = latent.shape
    video_frames = (latent_frames - 1) * 4 + 1

    engine_video_frames = pick_engine_frames(video_frames, engine_frames)
    if engine_video_frames is None:
        raise FileNotFoundError(
            "No TensorRT VAE decoder engine found. Build or download one first "
            "(e.g. vae_decoder_tile_256_{21,25,29,33,41,...}f.rtxplan)."
        )
    if engine_video_frames > video_frames:
        # The clip is shorter than every available engine: pad the latent to the
        # engine size, decode in 1 shot, then crop back to the actual length.
        engine_latent = (engine_video_frames - 1) // 4 + 1
        pad = engine_latent - latent_frames
        padded = torch.nn.functional.pad(latent, (0, 0, 0, 0, 0, pad))
        sample = _decode_single_chunk(padded, engine_latent, vae=vae, dit_model=dit_model)
        return sample[:, :, :video_frames]
    if engine_video_frames == video_frames:
        print(f"[SeedVR2 TensorRT] Decoding {engine_video_frames}f in 1 shot with dedicated {engine_video_frames}f TensorRT engine...")
        return _decode_single_chunk(latent, latent_frames, vae=vae, dit_model=dit_model)
    engine_latent = (engine_video_frames - 1) // 4 + 1
    n_chunks = (latent_frames + engine_latent - 2) // (engine_latent - 1)
    print(f"[SeedVR2 TensorRT] Decoding {n_chunks} chunks of {engine_video_frames}f with TensorRT engine...")
    return _decode_chunked(latent, latent_frames, engine_video_frames, vae=vae, dit_model=dit_model)


def resolve_engine_frames(preferred: str = "auto") -> int | None:
    """Return the largest available decoder engine video-frame size (for chunking)."""
    engines = _available_engine_frames()
    if not engines:
        return None
    if preferred != "auto":
        try:
            cand = int(preferred)
            if cand in engines:
                return cand
        except ValueError:
            pass
    return engines[-1]


def release() -> None:
    """Clear cached decoder execution contexts and streams to free GPU VRAM."""
    global _DECODER_ENGINES
    _DECODER_ENGINES.clear()
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _gc.collect()
        torch.cuda.empty_cache()
