"""HSWQ TC forward patch for SeedVR2 NVFP4 benchmark (Option C).

Patches SeedVR2's NVFP4 loading to use HSWQ's Tensor Core forward path
(pooled CUDA quantize + cuBLAS scaled_mm_nvfp4) instead of stock ComfyUI
mixed_precision Linear.

No ConvRot, no Hadamard rotation — just the TC forward path.

Note: `mixed_precision_ops()` returns a class (MixedPrecisionOps), and
`_load_quantized_module` is a module-level function in comfy.ops, NOT a
class attribute. Instead of wrapping the load path, the NVFP4 flag is set
lazily on first forward call.

SeedVR2 LayerNorm/RMSNorm emit float32 into Linear; NVFP4 CUDA kernels
only accept FP16/BF16. The original SeedVR2 wrapper cast activations to
compute_dtype before the stock path. We replicate that cast BEFORE the
HSWQ TC path (the TC path does not cast by itself).
"""
from __future__ import annotations

import os
import sys
import types
import torch

_TC_APPLIED = False
_ORIGINAL_GET_OPS = None


def _find_hswq_nodes(hswq_path=None, comfy_root=None):
    """Find HSWQ nodes directory."""
    if hswq_path and os.path.isdir(hswq_path):
        return hswq_path

    if comfy_root:
        candidate = os.path.join(
            comfy_root, "custom_nodes",
            "ComfyUI-HSWQ-Loader-and-Tools", "nodes",
        )
        if os.path.isdir(candidate):
            return candidate

    for p in sys.path:
        candidate = os.path.join(
            p, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools", "nodes",
        )
        if os.path.isdir(candidate):
            return candidate

    raise RuntimeError(
        "Cannot find ComfyUI-HSWQ-Loader-and-Tools/nodes. "
        "Pass --hswq_path or install HSWQ custom node."
    )


def _ensure_nvfp4_package(hswq_nodes):
    """Make nvfp4 package importable from HSWQ nodes directory."""
    nvfp4_dir = os.path.join(hswq_nodes, "nvfp4")
    if not os.path.isdir(nvfp4_dir):
        raise RuntimeError(f"HSWQ nvfp4 directory not found: {nvfp4_dir}")

    if hswq_nodes not in sys.path:
        sys.path.insert(0, hswq_nodes)

    if "nvfp4" in sys.modules:
        return

    # Try normal import first
    try:
        import nvfp4  # noqa: F401
        return
    except ImportError:
        pass

    # Create namespace package manually (no __init__.py needed)
    pkg = types.ModuleType("nvfp4")
    pkg.__path__ = [nvfp4_dir]
    sys.modules["nvfp4"] = pkg


def apply_tc_forward_patch(hswq_path=None, comfy_root=None):
    """Monkey-patch SeedVR2 to use HSWQ TC forward for NVFP4 Linear.

    Must be called BEFORE model loading. Patches:
    - src.optimization.nvfp4_native_ops.get_nvfp4_mixed_precision_ops
    - src.core.model_loader.get_nvfp4_mixed_precision_ops

    The patch:
    1. Gets stock ops from original function
    2. Replaces Linear.forward with HSWQ make_nvfp4_linear_forward
    3. Sets _hswq_nvfp4=True lazily on first forward for NVFP4 modules
    4. Casts float32 activations to compute_dtype BEFORE the TC path
       (NVFP4 CUDA kernels accept FP16/BF16 only; SeedVR2 norms emit
       float32, which the stock wrapper used to cast away)
    """
    global _TC_APPLIED, _ORIGINAL_GET_OPS
    if _TC_APPLIED:
        print("[TC PATCH] Already applied, skipping")
        return

    hswq_nodes = _find_hswq_nodes(hswq_path, comfy_root)
    print(f"[TC PATCH] HSWQ nodes: {hswq_nodes}")

    _ensure_nvfp4_package(hswq_nodes)

    # Import HSWQ TC forward maker
    from nvfp4.nvfp4_forward import make_nvfp4_linear_forward
    print("[TC PATCH] HSWQ nvfp4_forward imported OK")

    # Save original function
    import src.optimization.nvfp4_native_ops as nvfp4_ops_module
    _ORIGINAL_GET_OPS = nvfp4_ops_module.get_nvfp4_mixed_precision_ops

    def get_nvfp4_mixed_precision_ops_tc(compute_dtype=torch.float16):
        """Returns ops with HSWQ TC forward patch applied."""
        # Activation dtype: same rule as the original SeedVR2 wrapper
        if compute_dtype in (torch.float16, torch.bfloat16):
            _act_dtype = compute_dtype
        else:
            _act_dtype = torch.float16

        # Get base ops from original function
        ops = _ORIGINAL_GET_OPS(compute_dtype)

        # Save stock Linear.forward (SeedVR2 dtype-cast forward)
        stock_forward = ops.Linear.forward

        # HSWQ TC forward. It checks _hswq_nvfp4 flag; modules without the
        # flag fall through to stock_forward. We set the flag lazily here.
        tc_forward = make_nvfp4_linear_forward(stock_forward)

        def forward_nvfp4_tc(self, input, *args, **kwargs):
            # Lazily arm TC path on first forward for NVFP4-quantized modules.
            if getattr(self, "quant_format", None) == "nvfp4":
                self._hswq_nvfp4 = True
                # SeedVR2 norms emit float32 into Linear; NVFP4 CUDA kernels
                # only accept FP16/BF16. Cast before the TC path (the TC
                # forward does not cast by itself).
                if (
                    isinstance(input, torch.Tensor)
                    and getattr(self, "layout_type", None) is not None
                    and not getattr(self, "_full_precision_mm", False)
                    and input.dtype not in (torch.float16, torch.bfloat16)
                ):
                    input = input.to(dtype=_act_dtype)
            return tc_forward(self, input, *args, **kwargs)

        ops.Linear.forward = forward_nvfp4_tc

        print(
            "[TC PATCH] Ops patched: Linear.forward=HSWQ TC "
            "(lazy _hswq_nvfp4 flag + float32->compute_dtype cast)"
        )

        return ops

    # Patch in both modules (model_loader imports from nvfp4_native_ops)
    nvfp4_ops_module.get_nvfp4_mixed_precision_ops = get_nvfp4_mixed_precision_ops_tc

    import src.core.model_loader as model_loader_module
    model_loader_module.get_nvfp4_mixed_precision_ops = get_nvfp4_mixed_precision_ops_tc

    _TC_APPLIED = True
    print(
        "[TC PATCH] Applied to src.optimization.nvfp4_native_ops "
        "+ src.core.model_loader"
    )


def revert_tc_forward_patch():
    """Revert the TC forward patch."""
    global _TC_APPLIED, _ORIGINAL_GET_OPS
    if not _TC_APPLIED:
        return

    import src.optimization.nvfp4_native_ops as nvfp4_ops_module
    nvfp4_ops_module.get_nvfp4_mixed_precision_ops = _ORIGINAL_GET_OPS

    import src.core.model_loader as model_loader_module
    model_loader_module.get_nvfp4_mixed_precision_ops = _ORIGINAL_GET_OPS

    _TC_APPLIED = False
    print("[TC PATCH] Reverted")
