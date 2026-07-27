"""
HSWQ INT8 native inference via ComfyUI comfy.ops construction-time injection.

HSWQ safetensors carry ``comfy_quant`` + ``weight_scale``. Native VRAM-saving
load requires Linear modules that already implement
``_load_from_state_dict`` → ``comfy.ops._load_quantized_module`` at
``load_state_dict`` time. That is provided by
``comfy.ops.mixed_precision_ops``.

Post-load Linear replace (GGUF-style) does not interpret ``comfy_quant`` and
is the wrong path for this format.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch


def checkpoint_is_hswq_int8(checkpoint_path: Optional[str]) -> bool:
    """True if safetensors has at least one ``*.comfy_quant`` with format int8_tensorwise."""
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
                if conf.get("format") == "int8_tensorwise":
                    return True
    except Exception:
        return False
    return False


def get_hswq_mixed_precision_ops(compute_dtype: torch.dtype = torch.float16) -> Any:
    """
    Return ``comfy.ops.mixed_precision_ops`` with empty quant_config.

    Empty config is intentional: layers that carry ``comfy_quant`` become
    QuantizedTensor; layers without markers load as plain compute_dtype Parameters.
    """
    import comfy.ops as comfy_ops

    return comfy_ops.mixed_precision_ops(
        quant_config={},
        compute_dtype=compute_dtype,
        full_precision_mm=False,
        disabled=[],
    )


def resolve_linear_ops(operations: Optional[Any] = None) -> Any:
    """Return an object with ``.Linear`` (operations or ``torch.nn``)."""
    if operations is None:
        return torch.nn
    return operations


def prepare_hswq_state_dict_for_comfy_ops(state: dict) -> dict:
    """
    Move ``*.comfy_quant`` tensors to CPU in-place.

    ``comfy.ops._load_quantized_module`` does ``layer_conf.numpy().tobytes()``.
    That requires host memory; SeedVR2 often loads safetensors straight to CUDA,
    which raises: can't convert cuda device type tensor to numpy.
    Weight / scale tensors may stay on the target device.
    """
    for key, value in list(state.items()):
        if not key.endswith("comfy_quant"):
            continue
        if torch.is_tensor(value) and value.device.type != "cpu":
            state[key] = value.cpu()
    return state


def patch_ops_factory_device(model: torch.nn.Module, device: torch.device) -> int:
    """
    Point ``factory_kwargs["device"]`` at the materialization device.

    NaDiT is built under ``torch.device("meta")`` so Linear ``device`` is often
    ``None``/meta. ``_load_quantized_module`` places QuantizedTensor via that
    field; without this patch weights can remain on meta after assign load.
    """
    patched = 0
    for module in model.modules():
        fk = getattr(module, "factory_kwargs", None)
        if not isinstance(fk, dict) or "device" not in fk:
            continue
        fk["device"] = device
        patched += 1
    return patched
