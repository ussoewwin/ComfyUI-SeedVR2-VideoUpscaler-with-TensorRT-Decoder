"""Encoder parity test: TensorRT engine output vs PyTorch (FP16) on the same input.

Purpose: determine whether the TensorRT VAE encoder ENGINE itself matches the
PyTorch encoder (1 tile, no chunking). If they match, the corruption seen in
generation comes from the tiling/chunking wrapper, not the engine. If they
differ, the engine (ONNX export or TRT build) is the problem.

Usage (local, ComfyUI python_embeded):
    python tools/parity_test_encoder.py --repo <node_root> \
        --engine tensorrt_backend/artifacts/vae_encoder_21f_tile512.rtxplan \
        --tile 512 --frames 21 \
        --model ema_vae_fp16.safetensors --model-dir <models/SEEDVR2>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


class _EncoderModule(torch.nn.Module):
    """encoder + quant_conv, identical to cloud_export_gpu.py."""

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
    """Same as cloud_export_gpu.py / Studio: disable slicing and memory limits."""
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
    parser.add_argument("--repo", required=True, help="custom node root")
    parser.add_argument("--engine", required=True, help="path to .rtxplan encoder engine")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--frames", type=int, default=21)
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
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} / VRAM {props.total_memory/2**30:.1f} GiB", flush=True)

    repo_path = Path(args.repo)
    vae_config_path = repo_path / "src" / "models" / "video_vae_v3" / "s8_c16_t4_inflation_sd3.yaml"
    with open(vae_config_path, "r", encoding="utf-8") as f:
        vae_kwargs = yaml.safe_load(f)
    sig = inspect.signature(VideoAutoencoderKL.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "args", "kwargs"}
    filtered_kwargs = {k: v for k, v in vae_kwargs.items() if k in valid_params}

    print("Instantiating VideoAutoencoderKL on meta device...", flush=True)
    with torch.device("meta"):
        vae = VideoAutoencoderKL(**filtered_kwargs)

    vae_dir = Path(args.model_dir) if args.model_dir else (repo_path / "models" / "SEEDVR2")
    candidates = [vae_dir / args.model, vae_dir / "ema_vae_fp16.safetensors"]
    vae_file = next((p for p in candidates if p.exists()), None)
    if vae_file is None:
        print(f"ERROR: VAE file not found under {vae_dir} (looked for {args.model})", flush=True)
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

    # Normalize frame count to the engine profile (4n+1).
    frames = ((args.frames - 1) // 4) * 4 + 1
    tile = args.tile
    torch.manual_seed(args.seed)
    video = torch.randn(1, 3, frames, tile, tile, device="cuda", dtype=torch.float16)

    # ---- PyTorch (FP16) reference ----
    mod = _EncoderModule(vae).eval().to(device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        fp16_out = mod(video)
    print(f"FP16 out: {tuple(fp16_out.shape)}  range [{fp16_out.float().min():.4f}, {fp16_out.float().max():.4f}]  "
          f"std {fp16_out.float().std():.4f}", flush=True)

    # ---- TensorRT engine ----
    import tensorrt_rtx as trt  # noqa: E402
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(Path(args.engine).read_bytes())
    if engine is None:
        print(f"ERROR: could not deserialize {args.engine}", flush=True)
        return 1
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inp = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    in_shape = tuple(engine.get_tensor_shape(inp))
    out_shape = tuple(engine.get_tensor_shape(out))
    print(f"Engine: {inp}{in_shape} -> {out}{out_shape}", flush=True)
    if in_shape != (1, 3, frames, tile, tile):
        print(f"WARNING: engine input {in_shape} != requested {(1, 3, frames, tile, tile)}", flush=True)

    context = engine.create_execution_context()
    trt_out = torch.empty(out_shape, device="cuda", dtype=torch.float16)
    context.set_tensor_address(inp, video.data_ptr())
    context.set_tensor_address(out, trt_out.data_ptr())
    stream = torch.cuda.current_stream()
    t0 = time.perf_counter()
    if not context.execute_async_v3(stream.cuda_stream):
        print("ERROR: engine execution failed", flush=True)
        return 1
    stream.synchronize()
    print(f"TRT exec: {(time.perf_counter()-t0)*1000:.1f} ms", flush=True)
    print(f"TRT  out: {tuple(trt_out.shape)}  range [{trt_out.float().min():.4f}, {trt_out.float().max():.4f}]  "
          f"std {trt_out.float().std():.4f}", flush=True)

    if tuple(trt_out.shape) != tuple(fp16_out.shape):
        print(f"!! SHAPE MISMATCH: FP16 {tuple(fp16_out.shape)} vs TRT {tuple(trt_out.shape)}", flush=True)
        return 0

    # ---- Compare ----
    fp16_f = fp16_out.float()
    trt_f = trt_out.float()
    full_diff = (fp16_f - trt_f).abs()
    print(f"[all 32ch] max abs diff: {full_diff.max():.6f}  mean abs: {full_diff.mean():.6f}", flush=True)

    # The latent actually used downstream is the first 16 channels (posterior mean).
    mean_fp16 = fp16_f[:, :16]
    mean_trt = trt_f[:, :16]
    md = (mean_fp16 - mean_trt).abs()
    rel = md / mean_fp16.abs().clamp_min(1e-6)
    print(f"[mean16]   max abs diff: {md.max():.6f}  mean abs: {md.mean():.6f}", flush=True)
    print(f"[mean16]   relative err: mean {rel.mean():.6f}  p50 {rel.quantile(0.5):.6f}  p99 {rel.quantile(0.99):.6f}", flush=True)
    print(f"[mean16]   FP16 std {mean_fp16.std():.6f}  TRT std {mean_trt.std():.6f}", flush=True)

    # Structural comparison: cosine similarity per channel.
    cos = torch.nn.functional.cosine_similarity(
        mean_fp16.flatten(2).transpose(1, 2), mean_trt.flatten(2).transpose(1, 2), dim=-1)
    print(f"[mean16]   cosine sim: mean {cos.mean():.6f}  min {cos.min():.6f}", flush=True)

    # Spatial diff map: local anomalies (e.g. top-left) are invisible in global stats.
    if mean_fp16.dim() == 5:
        # channels-first (B,C,T,H,W)
        sp = md.mean(dim=(0, 1, 2))
    else:
        sp = md.mean(dim=(0, 1))
    H, W = sp.shape
    quad = {
        "top-left": sp[: max(1, H // 8), : max(1, W // 8)].mean().item(),
        "top-right": sp[: max(1, H // 8), -max(1, W // 8):].mean().item(),
        "bottom-left": sp[-max(1, H // 8):, : max(1, W // 8)].mean().item(),
        "bottom-right": sp[-max(1, H // 8):, -max(1, W // 8):].mean().item(),
        "whole": sp.mean().item(),
    }
    print(f"[mean16] spatial (corner {max(1,H//8)}x{max(1,W//8)}):", flush=True)
    for k, v in quad.items():
        ratio = v / quad["whole"] if quad["whole"] > 0 else 0
        print(f"    {k:12s}: {v:.4f}  ({ratio:.1f}x whole)", flush=True)

    verdict = "MATCH" if rel.mean() < 0.05 else "MISMATCH"
    print(f"VERDICT: {verdict} (mean rel err {rel.mean():.6f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
