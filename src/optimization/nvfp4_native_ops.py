"""
SeedVR2 NVFP4 native inference via ComfyUI comfy.ops construction-time injection.

NVFP4 safetensors carry ``comfy_quant`` (format ``nvfp4``) plus
``weight_scale`` (block, float8_e4m3fn) and ``weight_scale_2`` (tensor scale).
Native VRAM-saving load requires Linear modules that already implement
``_load_from_state_dict`` → ``comfy.ops._load_quantized_module`` at
``load_state_dict`` time. That is provided by
``comfy.ops.mixed_precision_ops`` (same path as INT8).

Post-load Linear replace does not interpret ``comfy_quant`` / NVFP4 scales
and expands weights — wrong for VRAM savings.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch


def checkpoint_is_nvfp4(checkpoint_path: Optional[str]) -> bool:
    """True if safetensors has at least one ``*.comfy_quant`` with format nvfp4."""
    if not checkpoint_path:
        return False
    path = str(checkpoint_path)
    if not (path.endswith(".safetensors") or path.endswith(".sft")):
        return False
    if not os.path.isfile(path):
        return False
    try:
        from safetensors import safe_open
    except ImportError:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.endswith(".comfy_quant"):
                    continue
                raw = f.get_tensor(key)
                if raw.dtype != torch.uint8:
                    continue
                conf = json.loads(raw.numpy().tobytes())
                if conf.get("format") == "nvfp4":
                    return True
    except Exception:
        return False
    return False


def get_nvfp4_mixed_precision_ops(compute_dtype: torch.dtype = torch.float16) -> Any:
    """
    Return ``comfy.ops.mixed_precision_ops`` for NVFP4 DiT loads.

    Empty ``quant_config``: layers with ``comfy_quant`` become QuantizedTensor;
    unmarked layers load as plain compute_dtype Parameters.

    When the GPU cannot run native NVFP4 matmul (``supports_nvfp4_compute``),
    ``nvfp4`` is listed in ``disabled`` so ComfyUI keeps packed QuantizedTensor
    storage (VRAM savings) but uses dequantized matmul — same as
    ``pick_operations`` for model configs. On Blackwell-class devices the
    format stays enabled for native tensor-core matmul.

    Native NVFP4 activation quantize (``comfy_kitchen.quantize_nvfp4``) accepts
    FP16/BF16 only. SeedVR2 LayerNorm / RMSNorm under ``torch.autocast`` often
    emit float32 into Linear; cast activations to ``compute_dtype`` before the
    stock MixedPrecision Linear path runs ``QuantizedTensor.from_float``.
    """
    import comfy.model_management as model_management
    import comfy.ops as comfy_ops

    disabled = []
    if not model_management.supports_nvfp4_compute():
        disabled = ["nvfp4"]

    ops = comfy_ops.mixed_precision_ops(
        quant_config={},
        compute_dtype=compute_dtype,
        full_precision_mm=False,
        disabled=disabled,
    )

    _BaseLinear = ops.Linear
    if compute_dtype in (torch.float16, torch.bfloat16):
        _act_dtype = compute_dtype
    else:
        _act_dtype = torch.float16

    class Linear(_BaseLinear):
        def forward(self, input, *args, **kwargs):
            if (
                isinstance(input, torch.Tensor)
                and getattr(self, "quant_format", None) == "nvfp4"
                and getattr(self, "layout_type", None) is not None
                and not getattr(self, "_full_precision_mm", False)
                and input.dtype not in (torch.float16, torch.bfloat16)
            ):
                input = input.to(dtype=_act_dtype)
            return super().forward(input, *args, **kwargs)

    ops.Linear = Linear
    return ops
