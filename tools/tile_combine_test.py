"""Tile-combine parity test: does the feathering accumulate of tile FP16 encodes
match a single full-frame FP16 encode?

The TensorRT engine was already verified to match FP16 (parity_test_encoder.py,
cosine ~0.99997). This test isolates the TILING/COMBINE logic by feeding each
tile through the PyTorch encoder (stand-in for the engine) and comparing the
feathered combine against one full encode. If this mismatches, the tiling
wrapper in trt_encoder.py is the corruption source.

Usage:
    python tools/tile_combine_test.py --repo <node_root> \
        --width 512 --height 256 --frames 21 \
        --model ema_vae_fp16.safetensors --model-dir <models/SEEDVR2>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


class _EncoderModule(torch.nn.Module):
    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(video, memory_state=MemoryState.DISABLED)
        if self.quant_conv is not None:
            hidden = self.quant_conv(hidden, memory_state=MemoryState.DISABLED)
        return hidden


def configure_fixed_vae(vae: torch.nn.Module) -> None:
    if hasattr(vae, "disable_slicing"):
        vae.disable_slicing()
    if hasattr(vae, "set_memory_limit"):
        vae.set_memory_limit(None, None)
    for module in vae.modules():
        if isinstance(module, InflatedCausalConv3d):
            module.set_memory_limit(float("inf"))
            if hasattr(module, "set_memory_device"):
                module.set_memory_device(None)
        if hasattr(module, "slicing"):
            module.slicing = False


def positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def feather(length: int, overlap: int, left: bool, right: bool, device: torch.device) -> torch.Tensor:
    weight = torch.ones(length, device=device, dtype=torch.float32)
    if left and overlap:
        weight[:overlap] = torch.linspace(0.0, 1.0, overlap + 1, device=device)[1:]
    if right and overlap:
        weight[-overlap:] = torch.minimum(
            weight[-overlap:], torch.linspace(1.0, 0.0, overlap + 1, device=device)[1:]
        )
    return weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--skip-full", action="store_true",
                        help="skip the single full FP16 encode (OOMs on large inputs) and use "
                             "the FP16 tiled combine as reference")
    parser.add_argument("--engine", default=None,
                        help=".rtxplan encoder engine. If set, each tile runs the actual TRT "
                             "engine instead of the PyTorch stand-in (tests multi-tile engine "
                             "execution against a single full FP16 encode).")
    parser.add_argument("--model", default="ema_vae_fp16.safetensors")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    import yaml  # noqa: E402
    import inspect  # noqa: E402

    from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKL  # noqa: E402
    from src.models.video_vae_v3.modules.types import MemoryState  # noqa: E402
    from src.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d  # noqa: E402
    from src.models.video_vae_v3.modules.global_config import set_norm_limit  # noqa: E402
    from safetensors.torch import load_file  # noqa: E402

    global MemoryState, InflatedCausalConv3d
    MemoryState = MemoryState
    InflatedCausalConv3d = InflatedCausalConv3d

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", flush=True)
        return 2

    repo_path = Path(args.repo)
    vae_config_path = repo_path / "src" / "models" / "video_vae_v3" / "s8_c16_t4_inflation_sd3.yaml"
    with open(vae_config_path, "r", encoding="utf-8") as f:
        vae_kwargs = yaml.safe_load(f)
    sig = inspect.signature(VideoAutoencoderKL.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "args", "kwargs"}
    filtered_kwargs = {k: v for k, v in vae_kwargs.items() if k in valid_params}

    with torch.device("meta"):
        vae = VideoAutoencoderKL(**filtered_kwargs)

    vae_dir = Path(args.model_dir) if args.model_dir else (repo_path / "models" / "SEEDVR2")
    candidates = [vae_dir / args.model, vae_dir / "ema_vae_fp16.safetensors"]
    vae_file = next((p for p in candidates if p.exists()), None)
    if vae_file is None:
        print(f"ERROR: VAE file not found under {vae_dir}", flush=True)
        return 1
    print(f"Loading VAE from {vae_file} ...", flush=True)
    state_dict = load_file(str(vae_file), device="cuda")
    vae.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    vae = vae.to(device="cuda", dtype=torch.float16).eval()
    configure_fixed_vae(vae)
    set_norm_limit(float("inf"))
    from src.utils.debug import Debug
    _dbg = Debug(enabled=False)
    for _module in vae.modules():
        if isinstance(_module, InflatedCausalConv3d):
            _module.set_memory_limit(float("inf"))
        _module.debug = _dbg
    torch.cuda.empty_cache()

    frames = ((args.frames - 1) // 4) * 4 + 1
    width, height = (args.width // 8) * 8, (args.height // 8) * 8
    tile, overlap = args.tile, args.overlap
    torch.manual_seed(args.seed)
    video = torch.randn(1, 3, frames, height, width, device="cuda", dtype=torch.float16)
    mod = _EncoderModule(vae).eval().to(device="cuda", dtype=torch.float16)

    # Shared tiling vars (needed by the FP16 tiled reference and the engine combine).
    ys = positions(height, tile, overlap)
    xs = positions(width, tile, overlap)
    padded_h = max(height, ys[-1] + tile)
    padded_w = max(width, xs[-1] + tile)
    source = F.pad(video, (0, padded_w - width, 0, padded_h - height))
    latent_frames = (frames - 1) // 4 + 1
    raw_h, raw_w = padded_h // 8, padded_w // 8
    overlap_latent = overlap // 8
    tile_lat = tile // 8

    # ---- Reference: single full encode (mean 16ch). For large inputs that OOM,
    # fall back to a full FP16 tiled encode (the combine logic was verified to be
    # accurate in the small-size test, so this reference is still valid).
    full_mean = None
    if not args.skip_full:
        try:
            with torch.inference_mode():
                full_raw = mod(video)  # (1, 32, lat_frames, H/8, W/8)
            full_mean = full_raw[:, :16].float()
            print(f"Full encode: {tuple(full_raw.shape)}  mean16 std {full_mean.std():.4f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print("Full encode OOM -> using FP16 tiled combine as reference", flush=True)
            torch.cuda.empty_cache()
    else:
        print("Skipping full encode -> using FP16 tiled combine as reference", flush=True)
    if full_mean is None:
        # FP16 tiled reference (feather combine of per-tile FP16 encodes)
        ref = torch.zeros((1, 32, latent_frames, raw_h, raw_w), device="cuda", dtype=torch.float32)
        refw = torch.zeros_like(ref)
        with torch.inference_mode():
            for y in ys:
                for x in xs:
                    t_in = source[:, :, :, y:y + tile, x:x + tile].contiguous()
                    t_out = mod(t_in)
                    ly, lx = y // 8, x // 8
                    wy = feather(tile_lat, overlap_latent, y != ys[0], y != ys[-1], t_out.device)
                    wx = feather(tile_lat, overlap_latent, x != xs[0], x != xs[-1], t_out.device)
                    win = (wy[:, None] * wx[None, :]).view(1, 1, 1, tile_lat, tile_lat)
                    ref[:, :, :, ly:ly + tile_lat, lx:lx + tile_lat] += t_out.float() * win
                    refw[:, :, :, ly:ly + tile_lat, lx:lx + tile_lat] += win
        full_mean = (ref / refw.clamp_min(1e-6))[:, :16, :, :height // 8, :width // 8]
        print(f"FP16 tiled reference: {tuple(full_mean.shape)}", flush=True)

    # ---- Tiled encode + feather combine (mirror of trt_encoder._encode_single_chunk) ----
    result = torch.zeros((1, 32, latent_frames, raw_h, raw_w), device="cuda", dtype=torch.float32)
    weights = torch.zeros_like(result)

    # Optional: run the real TensorRT engine per tile.
    engine_ctx = None
    if args.engine:
        import tensorrt_rtx as trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        eng = runtime.deserialize_cuda_engine(Path(args.engine).read_bytes())
        names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
        ein = next(n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        eout = next(n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        ectx = eng.create_execution_context()
        eout_shape = tuple(eng.get_tensor_shape(eout))
        print(f"Engine: {ein}{tuple(eng.get_tensor_shape(ein))} -> {eout}{eout_shape}", flush=True)
        engine_ctx = (ectx, ein, eout, eout_shape)

    with torch.inference_mode():
        for y in ys:
            for x in xs:
                tile_input = source[:, :, :, y:y + tile, x:x + tile].contiguous()
                if engine_ctx is not None:
                    ectx, ein, eout, eout_shape = engine_ctx
                    tile_output = torch.empty(eout_shape, device="cuda", dtype=torch.float16)
                    ectx.set_tensor_address(ein, tile_input.data_ptr())
                    ectx.set_tensor_address(eout, tile_output.data_ptr())
                    if not ectx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                        raise RuntimeError(f"engine failed at tile y={y}, x={x}")
                    torch.cuda.current_stream().synchronize()
                else:
                    tile_output = mod(tile_input)  # engine stand-in (FP16)
                ly, lx = y // 8, x // 8
                wy = feather(tile_lat, overlap_latent, y != ys[0], y != ys[-1], tile_output.device)
                wx = feather(tile_lat, overlap_latent, x != xs[0], x != xs[-1], tile_output.device)
                window = (wy[:, None] * wx[None, :]).view(1, 1, 1, tile_lat, tile_lat)
                result[:, :, :, ly:ly + tile_lat, lx:lx + tile_lat] += tile_output.float() * window
                weights[:, :, :, ly:ly + tile_lat, lx:lx + tile_lat] += window

    tiled_mean = (result / weights.clamp_min(1e-6))[:, :16, :, :height // 8, :width // 8]
    print(f"Tiled combine: {tuple(tiled_mean.shape)}  tiles {len(ys)}x{len(xs)}={len(ys)*len(xs)}  "
          f"ys={ys} xs={xs}", flush=True)

    # ---- Compare ----
    a = full_mean
    b = tiled_mean.float()
    if tuple(a.shape) != tuple(b.shape):
        print(f"!! SHAPE MISMATCH full {tuple(a.shape)} vs tiled {tuple(b.shape)}", flush=True)
        return 0

    diff = (a - b).abs()
    rel = diff / a.abs().clamp_min(1e-6)
    cos = F.cosine_similarity(a.flatten(1), b.flatten(1), dim=-1)
    print(f"[mean16] max abs diff: {diff.max():.6f}  mean abs: {diff.mean():.6f}", flush=True)
    print(f"[mean16] relative err: mean {rel.mean():.6f}  p50 {rel.quantile(0.5):.6f}  p99 {rel.quantile(0.99):.6f}", flush=True)
    print(f"[mean16] cosine sim: {cos.item():.6f}", flush=True)

    # Where are the biggest errors? (spatial distribution)
    per_px = diff.mean(dim=(0, 1, 2))  # per spatial px
    h_lat, w_lat = per_px.shape
    err_map = per_px.cpu()
    # row/col means to see if errors concentrate at tile boundaries
    row_err = err_map.mean(dim=1)
    col_err = err_map.mean(dim=0)
    print(f"col error means (latent width {w_lat}):", flush=True)
    print("  " + " ".join(f"{i}:{v:.4f}" for i, v in enumerate(col_err)), flush=True)

    verdict = "MATCH" if cos.item() > 0.99 and diff.mean() < 0.05 else "MISMATCH"
    print(f"VERDICT: {verdict} (cos {cos.item():.6f}, mean abs {diff.mean():.6f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
