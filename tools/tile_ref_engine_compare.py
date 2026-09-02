"""Verify: real-video 101f input + 101f engine + 18-tile combine (944x528 portrait).

Runs FP16 reference (per-tile, temporal slicing) and TRT engine combine in
SEPARATE invocations so 101f fits in 16GB VRAM, then compares.

Usage:
    python tools/tile_ref_engine_compare.py --repo <node_root> \
        --input D:/latent_debug/enc_input_raw.pt \
        --engine .../vae_encoder_101f_tile256.rtxplan \
        --workdir D:/latent_debug \
        --model ema_vae_fp16.safetensors --model-dir .../models/SEEDVR2

    phase=ref    : compute FP16 tiled reference -> save ref_fp16.pt (then exit)
    phase=engine : run TRT engine combine     -> save engine_out.pt, compare vs ref_fp16.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def configure_fixed_vae(vae: torch.nn.Module) -> None:
    # Keep temporal slicing ENABLED (101f needs per-slice encoding for VRAM).
    if hasattr(vae, "set_memory_limit"):
        vae.set_memory_limit(None, None)


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
    parser.add_argument("--phase", choices=["ref", "engine"], required=True)
    parser.add_argument("--input", required=True, help="enc_input_raw.pt (1,3,T,H,W)")
    parser.add_argument("--engine", required=True, help=".rtxplan encoder engine")
    parser.add_argument("--workdir", required=True, help="dir for ref_fp16.pt / engine_out.pt")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--model", default="ema_vae_fp16.safetensors")
    parser.add_argument("--model-dir", default=None)
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
    vae_file = next((p for p in [vae_dir / args.model, vae_dir / "ema_vae_fp16.safetensors"] if p.exists()), None)
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

    video = torch.load(args.input, map_location="cuda").to(dtype=torch.float16)
    if video.dim() == 4:
        video = video.unsqueeze(0)
    frames = video.shape[2]
    height, width = video.shape[3], video.shape[4]
    print(f"Input: {tuple(video.shape)}", flush=True)

    tile, overlap = args.tile, args.overlap
    ys = positions(height, tile, overlap)
    xs = positions(width, tile, overlap)
    padded_h = max(height, ys[-1] + tile)
    padded_w = max(width, xs[-1] + tile)
    source = F.pad(video, (0, padded_w - width, 0, padded_h - height))
    tile_lat = tile // 8
    overlap_latent = overlap // 8
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    def feathered_tiles(encode_fn, tag):
        raw_h, raw_w = padded_h // 8, padded_w // 8
        latent_frames = (frames - 1) // 4 + 1
        ch = 32
        result = torch.zeros((1, ch, latent_frames, raw_h, raw_w), device="cuda", dtype=torch.float32)
        weights = torch.zeros_like(result)
        with torch.inference_mode():
            for y in ys:
                for x in xs:
                    tile_input = source[:, :, :, y:y + tile, x:x + tile].contiguous()
                    tile_output = encode_fn(tile_input)  # (1, 32, lat, tl, tl)
                    ly, lx = y // 8, x // 8
                    wy = feather(tile_lat, overlap_latent, y != ys[0], y != ys[-1], tile_output.device)
                    wx = feather(tile_lat, overlap_latent, x != xs[0], x != xs[-1], tile_output.device)
                    window = (wy[:, None] * wx[None, :]).view(1, 1, 1, tile_lat, tile_lat)
                    result[:, :, :, ly:ly + tile_lat, lx:lx + tile_lat] += tile_output.float() * window
                    weights[:, :, :, ly:ly + tile_lat, lx:lx + tile_lat] += window
                    print(f"  [{tag}] tile y={y} x={x} done", flush=True)
        return (result / weights.clamp_min(1e-6)), result  # combined, raw(unweighted sum ok)

    if args.phase == "ref":
        def encode_fp16(t_in):
            out = vae.encode(t_in).latent_dist.mode()  # mean 16ch
            # duplicate to 32ch to reuse the same combine math: [mean, mean]
            return torch.cat([out, out], dim=1)
        combined, _ = feathered_tiles(encode_fp16, "FP16")
        combined = combined[:, :16, :, :height // 8, :width // 8].float().cpu()
        torch.save(combined, workdir / "ref_fp16.pt")
        print(f"Saved {workdir / 'ref_fp16.pt'} {tuple(combined.shape)}", flush=True)
        return 0

    # phase == engine
    import tensorrt_rtx as trt
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    eng = runtime.deserialize_cuda_engine(Path(args.engine).read_bytes())
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    ein = next(n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    eout = next(n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    eout_shape = tuple(eng.get_tensor_shape(eout))
    ectx = eng.create_execution_context()
    print(f"Engine: {ein}{tuple(eng.get_tensor_shape(ein))} -> {eout}{eout_shape}", flush=True)

    def encode_trt(t_in):
        t_out = torch.empty(eout_shape, device="cuda", dtype=torch.float16)
        ectx.set_tensor_address(ein, t_in.data_ptr())
        ectx.set_tensor_address(eout, t_out.data_ptr())
        if not ectx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("engine execution failed")
        torch.cuda.current_stream().synchronize()
        return t_out

    combined, _ = feathered_tiles(encode_trt, "TRT")
    combined = combined[:, :16, :, :height // 8, :width // 8].float().cpu()
    torch.save(combined, workdir / "engine_out.pt")
    print(f"Saved {workdir / 'engine_out.pt'} {tuple(combined.shape)}", flush=True)

    # compare if ref exists
    ref_path = workdir / "ref_fp16.pt"
    if ref_path.exists():
        ref = torch.load(ref_path)
        if tuple(ref.shape) == tuple(combined.shape):
            diff = (ref - combined).abs()
            H, W = diff.shape[3], diff.shape[4]
            sp = diff.mean(dim=(0, 1, 2))
            print(f"[compare] max abs: {diff.max():.4f}  mean abs: {diff.mean():.4f}", flush=True)
            print(f"[compare] TL(H{H//8}xW{W//8}) mean: {sp[:H//8, :W//8].mean().item():.4f}", flush=True)
            print(f"[compare] TR mean: {sp[:H//8, -W//8:].mean().item():.4f}", flush=True)
            print(f"[compare] BL mean: {sp[-H//8:, :W//8].mean().item():.4f}", flush=True)
            print(f"[compare] BR mean: {sp[-H//8:, -W//8:].mean().item():.4f}", flush=True)
            # per-row/per-col
            row = sp.mean(dim=1)
            col = sp.mean(dim=0)
            print(f"row diffs: {[round(float(v),4) for v in row]}", flush=True)
            print(f"col diffs: {[round(float(v),4) for v in col]}", flush=True)
        else:
            print(f"!! shape mismatch ref {tuple(ref.shape)} vs engine {tuple(combined.shape)}", flush=True)
    else:
        print("ref_fp16.pt not found; run --phase ref first", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
