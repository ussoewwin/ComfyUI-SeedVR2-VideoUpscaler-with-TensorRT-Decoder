"""Chunk-combine parity test: does frame-direction chunking (as in
infer._trt_encode_batch) match a single full encode?

Spatial is fixed to 1 tile (256x256) so only the temporal chunking is tested.
Each chunk is encoded with the PyTorch encoder (engine stand-in; the engine was
already verified to match FP16), then combined with the SAME logic as
_trt_encode_batch (earlier chunk wins).

Usage:
    python tools/chunk_combine_test.py --repo <node_root> \
        --total 89 --engine-frames 45 --model ema_vae_fp16.safetensors \
        --model-dir <models/SEEDVR2>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--total", type=int, default=89, help="total frames (4n+1)")
    parser.add_argument("--engine-frames", type=int, default=45, help="chunk/engine frames (4n+1)")
    parser.add_argument("--overlap-frames", type=int, default=0,
                        help="temporal overlap between chunks in video frames. 0 = current "
                             "auto stride ((engine-4)//4*4). Larger overlap (e.g. 40-48) gives "
                             "later chunks full temporal context, at the cost of more chunks.")
    parser.add_argument("--spatial", type=int, default=256, help="spatial size (square; keep small for VRAM)")
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

    total = ((args.total - 1) // 4) * 4 + 1
    engine_frames = ((args.engine_frames - 1) // 4) * 4 + 1
    if engine_frames >= total:
        print("ERROR: engine_frames must be < total for chunking", flush=True)
        return 1
    spatial = ((args.spatial + 7) // 8) * 8
    torch.manual_seed(args.seed)
    video = torch.randn(1, 3, total, spatial, spatial, device="cuda", dtype=torch.float16)
    mod = _EncoderModule(vae).eval().to(device="cuda", dtype=torch.float16)
    lat_total = (total - 1) // 4 + 1

    # ---- Reference: full encode, then free ----
    with torch.inference_mode():
        full_raw = mod(video)
    full_mean = full_raw[:, :16].float().clone()
    del full_raw
    torch.cuda.empty_cache()
    print(f"Full encode: total {total}f mean16 {tuple(full_mean.shape)} std {full_mean.std():.4f}", flush=True)

    # ---- Chunked encode (mirror of infer._trt_encode_batch) ----
    if args.overlap_frames > 0:
        stride = engine_frames - args.overlap_frames
        if stride < 4:
            stride = 4
    else:
        stride = ((engine_frames - 4) // 4) * 4
        if stride < 4:
            stride = 4
    starts = list(range(0, total - engine_frames + 1, stride))
    if starts[-1] != total - engine_frames:
        starts.append(total - engine_frames)
    print(f"Chunk: engine {engine_frames}f stride {stride} starts {starts}", flush=True)

    lat_h = lat_w = spatial // 8
    latent = torch.zeros((1, 16, lat_total, lat_h, lat_w), device="cuda", dtype=torch.float16)
    parts = []
    with torch.inference_mode():
        for start in starts:
            chunk = video[:, :, start:start + engine_frames].contiguous()
            raw = mod(chunk)
            parts.append((raw[:, :16].float().clone(), start // 4))
            del raw, chunk
            torch.cuda.empty_cache()
    # Earlier chunk wins (current implementation: write in reverse)
    for lat, lat_start in reversed(parts):
        latent[:, :, lat_start:lat_start + lat.shape[2]] = lat.half()
    print(f"Chunked combine: {tuple(latent.shape)}  parts {[(s, p.shape[2]) for p, s in parts]}", flush=True)

    # ---- Compare per latent frame ----
    a = full_mean  # (1, 16, lat_total, h, w)
    b = latent.float()
    diff = (a - b).abs()
    rel = diff / a.abs().clamp_min(1e-6)
    cos_all = torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1), dim=-1)
    print(f"[all]   max abs diff: {diff.max():.6f}  mean abs: {diff.mean():.6f}  cos {cos_all.item():.6f}", flush=True)
    print("Per latent frame (mean abs diff / rel err):", flush=True)
    for t in range(lat_total):
        d = diff[0, :, t].mean().item()
        r = rel[0, :, t].mean().item()
        marker = " <-- chunk boundary" if t in [s // 4 for s in starts[1:]] else ""
        print(f"  lat {t:2d} (vid {t*4:3d}-{t*4+3:3d}): abs {d:.5f}  rel {r:.5f}{marker}", flush=True)

    verdict = "MATCH" if cos_all.item() > 0.99 and diff.mean() < 0.05 else "MISMATCH"
    print(f"VERDICT: {verdict} (cos {cos_all.item():.6f}, mean abs {diff.mean():.6f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
