"""Portable, observable ONNX export helpers for TensorRT engine preparation using CUDA/cuDNN."""

from __future__ import annotations

import gc
import time
from contextlib import nullcontext
from pathlib import Path

import torch


def _register_custom_symbolics():
    try:
        from torch.onnx import register_custom_op_symbolic

        def symbolic_cudnn_convolution(g, input, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32):
            return g.op(
                "Conv",
                input,
                weight,
                pads_i=[p for p in padding for _ in (0, 1)] if isinstance(padding, (list, tuple)) else None,
                strides_i=stride if isinstance(stride, (list, tuple)) else None,
                dilations_i=dilation if isinstance(dilation, (list, tuple)) else None,
                group_i=groups if isinstance(groups, int) else 1,
            )

        register_custom_op_symbolic("::cudnn_convolution", symbolic_cudnn_convolution, 20)
        register_custom_op_symbolic("aten::cudnn_convolution", symbolic_cudnn_convolution, 20)
    except Exception:
        pass


_register_custom_symbolics()


def _math_attention_context():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except (ImportError, AttributeError):
        return nullcontext()


def _portable_export(module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: Path, *, legacy: bool, dynamic_axes: dict | None = None) -> None:
    is_encoder = args[0].ndim == 5 and args[0].shape[1] == 3
    with torch.inference_mode(), torch.no_grad(), _math_attention_context():
        torch.onnx.export(
            module,
            args,
            str(output),
            input_names=["video"] if is_encoder else ["latent"],
            output_names=["latent_raw"] if is_encoder else ["sample"],
            opset_version=20,
            dynamo=not legacy,
            optimize=False,
            do_constant_folding=False,
            dynamic_axes=dynamic_axes,
        )


def export_portable_onnx(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output: Path,
    *,
    legacy: bool = True,
    dynamic_axes: dict | None = None,
) -> None:
    """Export ONNX graph directly on GPU using CUDA/cuDNN in 2-3 seconds."""
    print(f"[SeedVR2 TensorRT] Exporting ONNX graph (CUDA/cuDNN) -> {output.name}...", flush=True)
    started = time.perf_counter()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        _portable_export(module, args, output, legacy=legacy, dynamic_axes=dynamic_axes)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print("[SeedVR2 TensorRT] ⚠️ CUDA VRAM limit reached during ONNX trace; falling back to CPU export...", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cpu_module = module.to(device="cpu", dtype=torch.float16)
        cpu_args = tuple(a.to(device="cpu", dtype=torch.float16) for a in args)
        _portable_export(cpu_module, cpu_args, output, legacy=legacy, dynamic_axes=dynamic_axes)
        del cpu_module, cpu_args
        gc.collect()

    print(f"[SeedVR2 TensorRT] ✅ ONNX export finished in {time.perf_counter() - started:.1f}s", flush=True)
