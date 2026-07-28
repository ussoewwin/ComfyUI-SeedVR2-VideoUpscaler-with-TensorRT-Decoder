# SeedVR2 Video Upscaler — NVFP4 Native Ops and torch.compile Fixes

Target custom node: `ComfyUI/custom_nodes/seedvr2_videoupscaler`  
Canonical commit: `a14db91b31c08bee62055e17521d4f1537bef03c`  
(`feat: NVFP4 DiT native ops and VAE torch.compile inductor fixes`)

This guide documents **(A) NVFP4 DiT native load** and **(B) VAE/DiT torch.compile inductor fixes** landed in that commit.  
Abandoned later experiments (max-autotune remaps, phase park/isolate) are **out of scope**.

Related prior guide: `md/SEEDVR2_INT8_NATIVE_OPS_GUIDE.md` (INT8 construction-time `comfy.ops`).

---

## 1. NVFP4 support overview

### Problem

SeedVR2 DiT packs quantized as **NVFP4** ship:

- `*.comfy_quant` JSON with `format == "nvfp4"`
- `weight_scale` (block, often `float8_e4m3fn`)
- `weight_scale_2` (tensor scale)

If Linear layers are plain `nn.Linear` and weights are expanded **after** load (post-load replace), ComfyUI never runs `comfy.ops._load_quantized_module`. Packed NVFP4 storage is lost → **VRAM savings gone**, same failure mode as pre-native INT8.

### Approach (same family as INT8)

1. Detect NVFP4 via `checkpoint_is_nvfp4()` (scan `*.comfy_quant`).
2. At DiT **construction** (meta), inject `comfy.ops.mixed_precision_ops` through `create_object(..., operations=...)`.
3. Before `load_state_dict`: reuse INT8 prep helpers (`prepare_hswq_state_dict_for_comfy_ops`, `patch_ops_factory_device`) so `comfy_quant` is CPU-readable and factory device is real.
4. If the GPU lacks native NVFP4 matmul (`supports_nvfp4_compute() == False`), put `"nvfp4"` in `disabled` so storage stays packed but matmul dequantizes (ComfyUI `pick_operations` pattern).
5. Wrap `ops.Linear.forward` to cast activations to FP16/BF16 when `comfy_kitchen.quantize_nvfp4` would reject float32.
6. In DiT upscale phase, **skip `torch.autocast`** for native NVFP4 so LayerNorm/RMSNorm do not feed float32 into NVFP4 Linear.

### Data flow

```
NVFP4 DiT .safetensors
  → checkpoint_is_nvfp4()
  → create_object(..., operations=get_nvfp4_mixed_precision_ops(...))
  → prepare_hswq_state_dict_for_comfy_ops() + patch_ops_factory_device()
  → load_state_dict → comfy.ops._load_quantized_module
  → inference: QuantizedTensor path (+ optional act cast; no full-forward autocast)
```

VAE is **not** NVFP4-quantized in this workstream; VAE torch.compile issues are separate (sections 5–9).

---

## 2. Added / modified files (NVFP4)

### New

| Path |
|------|
| `src/optimization/nvfp4_native_ops.py` |

### Modified

| Path | Role |
|------|------|
| `src/core/model_loader.py` | `_dit_comfy_quant_ops` / `_dit_needs_comfy_quant_prep`; inject ops for NVFP4 or INT8; flag `_dit_comfy_quant_native` |
| `src/core/generation_phases.py` | Skip DiT `torch.autocast` when native NVFP4 |
| `src/common/config.py` | Docstring: `operations` for INT8 / NVFP4 |
| `src/utils/model_registry.py` | Comment wording only (INT8 entry retained) |

(`__init__.py` / `fix_inductor.py` belong to torch.compile — sections 7–8.)

---

## 3. Full source of added / modified files (NVFP4)

Current tree at `a14db91` (working tree restored to that commit).

### `src/optimization/nvfp4_native_ops.py` (new, complete)

```python
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
```

### `src/core/model_loader.py` (complete; includes INT8 + NVFP4)

```python
"""
Model Weight Loading for SeedVR2

This module handles all weight loading operations for DiT and VAE models:
- Loading state dictionaries from multiple formats (SafeTensors, PyTorch, GGUF)
- Materializing models from meta device to target device
- Applying weights with dtype conversion
- GGUF quantized model support with dequantization
- Meta buffer initialization for non-persistent buffers

Key Features:
- Multi-format support: .safetensors, .pth, .gguf files
- Memory-efficient loading with meta device initialization
- Native FP8 weight handling with optimal performance
- GGUF quantization support (Q4_K_M, Q8_0, etc.)
- Automatic dtype conversion for compatibility
- Meta buffer initialization post-materialization

Main Functions:
- load_quantized_state_dict: Load state dict from checkpoint file
- materialize_model: Move model from meta device and load weights
- prepare_model_structure: Create model structure on meta device

GGUF Support:
- apply_gguf_parameters: Apply GGUF weights to model (handles meta and materialized)
- _load_gguf_state: Load GGUF quantized weights from file
- _load_gguf_weights: Apply GGUF weights to model with validation
- _validate_gguf_architecture: Validate GGUF model architecture
- _create_dequantize_method: Create dequantization callable
- _create_gguf_parameter: Create parameter preserving quantization info
- _set_parameter_on_meta_model: Set parameter on meta device model
- _set_parameter_on_materialized_model: Set parameter on materialized model
- _navigate_to_parameter: Navigate to module containing parameter
- _get_tensor_shape: Get logical shape of tensor (handling GGUF)
- _is_quantized_tensor: Check if tensor is GGUF quantized
- _report_parameter_mismatches: Report parameter mismatches

Meta Buffer Initialization:
- initialize_meta_buffers: Initialize meta buffers with timing wrapper
- initialize_meta_buffers_impl: Initialize non-persistent buffers on target device

Standard Loading:
- _load_model_weights: Orchestrate weight loading process
- _load_standard_weights: Apply SafeTensors/PyTorch weights
- _convert_state_dtype: Convert weight dtypes
- _log_weight_stats: Log weight statistics

This module is used by model_configuration for weight loading during materialization.
"""

import os
import torch
from omegaconf import OmegaConf
from typing import Dict, Any, Optional, Tuple, Union, Callable

# Import SafeTensors with fallback
try:
    from safetensors.torch import load_file as load_safetensors_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

from .infer import VideoDiffusionInfer
from ..common.config import create_object
from ..optimization.compatibility import (
    GGUF_AVAILABLE,
    GGMLQuantizationType,
    validate_gguf_availability
)
from ..optimization.int8_native_ops import (
    checkpoint_is_hswq_int8,
    get_hswq_mixed_precision_ops,
    patch_ops_factory_device,
    prepare_hswq_state_dict_for_comfy_ops,
)
from ..optimization.nvfp4_native_ops import (
    checkpoint_is_nvfp4,
    get_nvfp4_mixed_precision_ops,
)


def _dit_comfy_quant_ops(checkpoint_path: Optional[str], compute_dtype: torch.dtype):
    """
    Construction-time comfy.ops for DiT packs that use comfy_quant markers.

    INT8 (int8_tensorwise) and NVFP4 share the same injection requirement:
    mixed_precision Linear must exist before load_state_dict so
    _load_quantized_module keeps QuantizedTensor (VRAM savings).
    """
    if not checkpoint_path or str(checkpoint_path).endswith(".gguf"):
        return None
    if checkpoint_is_nvfp4(checkpoint_path):
        return get_nvfp4_mixed_precision_ops(compute_dtype)
    if checkpoint_is_hswq_int8(checkpoint_path):
        return get_hswq_mixed_precision_ops(compute_dtype)
    return None


def _dit_needs_comfy_quant_prep(checkpoint_path: Optional[str]) -> bool:
    if not checkpoint_path or str(checkpoint_path).endswith(".gguf"):
        return False
    return checkpoint_is_nvfp4(checkpoint_path) or checkpoint_is_hswq_int8(checkpoint_path)

# GGUF-specific imports (only when available)
if GGUF_AVAILABLE:
    import gguf
    import traceback
    from ..optimization.gguf_dequant import dequantize_tensor
    from ..optimization.gguf_ops import replace_linear_with_quantized

from ..utils.constants import get_script_directory, suppress_tensor_warnings

# Get script directory for config paths
script_directory = get_script_directory()


def load_quantized_state_dict(checkpoint_path: str, device: torch.device = torch.device("cpu"),
                              debug: Optional['Debug'] = None) -> Dict[str, torch.Tensor]:
    """
    Load model state dict from checkpoint with support for multiple formats.
    
    Handles .safetensors, .gguf, and .pth files. GGUF models support quantization
    for memory-efficient loading. Validates required libraries are installed.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Target device for tensor placement (torch.device object, defaults to CPU)
        debug: Optional Debug instance for logging
        
    Returns:
        dict: State dictionary loaded with appropriate format handler
        
    Notes:
        - SafeTensors files use optimized loading with direct device placement
        - PyTorch files use memory-mapped loading to reduce RAM usage
    """
    device_str = str(device)
    
    if checkpoint_path.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            error_msg = (
                f"Cannot load {os.path.basename(checkpoint_path)}\n"
                f"SafeTensors library is required but not installed.\n"
                f"Please install it with: pip install safetensors"
            )
            if debug:
                debug.log(error_msg, level="ERROR", category="dit", force=True)
                debug.log("This is a one-time installation that will enable loading of .safetensors files", 
                         level="INFO", category="info", force=True)
            raise ImportError(error_msg)
        
        # Try direct device loading first (optimal path)
        try:
            state = load_safetensors_file(checkpoint_path, device=device_str)
        except RuntimeError as e:
            # MPS allocator fallback: some PyTorch/macOS versions have issues with
            # direct MPS loading (allocation failures, watermark errors, etc.)
            error_msg = str(e).lower()
            is_mps_alloc_error = device.type == "mps" and any(
                keyword in error_msg for keyword in ["watermark", "allocat", "memory"]
            )
            
            if is_mps_alloc_error:
                # Transparent fallback - only log if debug enabled
                if debug:
                    debug.log("Using CPU intermediate loading for MPS compatibility", 
                            category="info", indent_level=1)
                state = load_safetensors_file(checkpoint_path, device="cpu")
                # Tensors will be moved to MPS during model.load_state_dict()
            else:
                # Re-raise if it's a different error (file corruption, etc.)
                raise

    elif checkpoint_path.endswith('.gguf'):
        validate_gguf_availability(f"load {os.path.basename(checkpoint_path)}", debug)
        state = _load_gguf_state(
                    checkpoint_path=checkpoint_path, 
                    device=device, 
                    debug=debug, 
                    handle_prefix="model.diffusion_model."
                )
    elif checkpoint_path.endswith('.pth'):
        state = torch.load(checkpoint_path, map_location=device_str, mmap=True, weights_only=True)
    else:
        raise ValueError(f"Unsupported checkpoint format. Expected .safetensors or .pth, got: {checkpoint_path}")
    
    return state


def _load_gguf_state(checkpoint_path: str, device: torch.device, debug: Optional['Debug'] = None,
                    handle_prefix: str = "model.diffusion_model.") -> Dict[str, torch.Tensor]:
    """
    Load GGUF state dict
    
    Args:
        checkpoint_path: Path to GGUF file
        device: Target device (torch.device object)
        debug: Debug instance
        handle_prefix: Prefix to strip from tensor names
        
    Returns:
        State dictionary with loaded tensors
    """
    reader = gguf.GGUFReader(checkpoint_path)

    # Filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)
        
    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    state_dict = {}
    total_tensors = len(reader.tensors)
    
    device_str = str(device)
    debug.log(f"Loading {total_tensors} tensors to {str(device_str)}...", category="dit")
    
    # Suppress expected warnings: GGUF tensors are read-only numpy arrays that trigger warnings when converted
    suppress_tensor_warnings()
    
    for i, (sd_key, tensor) in enumerate(tensors):
        tensor_name = tensor.name
        
        # Create tensor directly on target device to avoid CPU->GPU copy overhead
        # For meta-initialized models, this directly materializes to the target device
        torch_tensor = torch.from_numpy(tensor.data).to(device, non_blocking=False)
            
        # Get original shape from metadata or infer from tensor shape
        shape = _get_tensor_logical_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            
        # Handle tensors based on quantization type
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # For unquantized tensors, just reshape
            torch_tensor = torch_tensor.view(*shape)
        else:
            # For quantized tensors, keep them quantized but track original shape
            torch_tensor = GGUFTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape, debug=debug)
            
        state_dict[sd_key] = torch_tensor
        
        # Progress reporting
        if (i + 1) % 100 == 0:
            debug.log(f"Loaded {i+1}/{total_tensors} tensors...", category="dit", indent_level=1)

    debug.log(f"Successfully loaded {len(state_dict)} tensors to {device_str}", category="success")

    return state_dict


def _get_tensor_logical_shape(reader: 'gguf.GGUFReader', tensor_name: str) -> Optional[torch.Size]:
    """
    Extract the logical (unquantized) shape from GGUF metadata
    """
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


class GGUFTensor(torch.Tensor):
    """
    Tensor wrapper for GGUF quantized tensors that preserves quantization info
    """
    def __init__(self, *args, tensor_type, tensor_shape, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        
    def __new__(cls, *args, tensor_type, tensor_shape, debug, **kwargs):
        # Create tensor with requires_grad=False to avoid gradient issues
        tensor = super().__new__(cls, *args, **kwargs)
        tensor.requires_grad_(False)
        tensor.tensor_type = tensor_type
        tensor.tensor_shape = tensor_shape
        tensor.debug = debug
        return tensor
    
    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", self.tensor_shape if hasattr(self, "tensor_shape") else new.shape)
        new.debug = getattr(self, "debug", None)
        new.requires_grad_(False)  # Ensure no gradients
        return new
    
    @property
    def shape(self):
        # Always return the logical tensor shape, not the quantized data shape
        if hasattr(self, "tensor_shape"):
            return self.tensor_shape
        else:
            # Fallback to actual data shape if tensor_shape is not available
            return self.size()
        
    def size(self, *args):
        # Override size() to also return logical shape
        if hasattr(self, "tensor_shape") and len(args) == 0:
            return self.tensor_shape
        elif hasattr(self, "tensor_shape") and len(args) == 1:
            return self.tensor_shape[args[0]]
        else:
            return super().size(*args)
        
    def dequantize(self, device=None, dtype=torch.float16, dequant_dtype=None):
        """Dequantize this tensor to its original shape"""
        if device is None:
            device = self.device
            
        # Suppress expected warning when converting from GGUFTensor subclass to regular tensor
        suppress_tensor_warnings()

        # Check if already unquantized
        if self.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # Return regular tensor, not GGUFTensor
            result = self.to(device, dtype)
            if isinstance(result, GGUFTensor):
                # Convert to regular tensor to avoid __torch_function__ calls
                result = torch.tensor(result, dtype=dtype, device=device, requires_grad=False)
            return result
        
        # Try fast dequantization with crash protection
        try:
            result = dequantize_tensor(self, dtype, dequant_dtype)
            final_result = result.to(device)
            
            # Ensure we return a regular tensor, not GGUFTensor
            if isinstance(final_result, GGUFTensor):
                final_result = torch.tensor(final_result.data, dtype=dtype, device=device, requires_grad=False)
                
            return final_result
        except Exception as e:
            self.debug.log(f"Fast dequantization failed: {e}", level="WARNING", category="dit", force=True)
            self.debug.log(f"Falling back to numpy dequantization", level="WARNING", category="dit", force=True)
            
        # Fallback to numpy (slower but reliable)
        try:
            numpy_data = self.cpu().numpy()
            dequantized = gguf.quants.dequantize(numpy_data, self.tensor_type)
            result = torch.from_numpy(dequantized).to(device, dtype)
            result.requires_grad_(False)
            final_result = result.reshape(self.tensor_shape)
            # from_numpy already returns a regular tensor, no conversion needed
            return final_result
        except Exception as e:
            self.debug.log(f"Numpy fallback also failed: {e}", level="WARNING", category="dit", force=True)
            self.debug.log(f"Tensor type: {self.tensor_type}", level="WARNING", category="dit", force=True, indent_level=1)
            self.debug.log(f"Shape: {self.shape}", level="WARNING", category="dit", force=True, indent_level=1)
            self.debug.log(f"Target shape: {self.tensor_shape}", level="WARNING", category="dit", force=True, indent_level=1)
            traceback.print_exc()
            
            # Return regular tensor as last resort
            result = self.to(device, dtype)
            if isinstance(result, GGUFTensor):
                result = torch.tensor(result.data, dtype=dtype, device=device, requires_grad=False)
            return result
        
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Override torch function calls to automatically dequantize"""
        if kwargs is None:
            kwargs = {}
        
        # Find the GGUFTensor instance(s) in args
        gguf_tensors = [arg for arg in args if isinstance(arg, cls)]
        if not gguf_tensors:
            return super().__torch_function__(func, types, args, kwargs)
        
        # Use the first GGUFTensor instance for attribute access
        self = gguf_tensors[0]
        
        # Check if the tensor is fully constructed and still quantized
        tensor_type = getattr(self, 'tensor_type', None)
        if tensor_type is None:
            # Tensor is either being constructed or already dequantized
            return super().__torch_function__(func, types, args, kwargs)
        
        # Check if tensor is already unquantized (F32/F16)
        if tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            return super().__torch_function__(func, types, args, kwargs)
        
        # Check if debug exists before using it
        debug = getattr(self, 'debug', None)
        
        # Handle linear operations specially
        if func == torch.nn.functional.linear:
            if len(args) >= 2 and isinstance(args[1], cls):  # weight is the second argument
                try:
                    weight_tensor = args[1]
                    dequantized_weight = weight_tensor.dequantize(device=args[0].device, dtype=args[0].dtype)
                    new_args = (args[0], dequantized_weight) + args[2:]
                    return func(*new_args, **kwargs)
                except Exception as e:
                    if debug:
                        debug.log(f"Error in linear dequantization: {e}", level="WARNING", category="dit", force=True)
                        debug.log(f"Function: {func}", level="WARNING", category="dit", force=True, indent_level=1)
                        debug.log(f"Args: {[arg.shape if hasattr(arg, 'shape') else type(arg) for arg in args]}", level="WARNING", category="dit", force=True, indent_level=1)
                    raise
        
        # Handle matrix multiplication operations that need dequantization
        if func in {torch.matmul, torch.mm, torch.bmm, torch.addmm, torch.addmv,
                    torch.addr, torch.baddbmm, torch.chain_matmul}:
            try:
                new_args = []
                for arg in args:
                    if isinstance(arg, cls):
                        new_args.append(arg.dequantize())
                    else:
                        new_args.append(arg)
                return func(*tuple(new_args), **kwargs)
            except Exception as e:
                if debug:
                    debug.log(f"Error in {func.__name__} dequantization: {e}", level="WARNING", category="dit", force=True)
                raise

        # Handle conv2d/conv3d operations (critical for GGUF VAE models)
        # Conv3d layers (InflatedCausalConv3d) are not replaced by layer replacement
        if func in {torch.nn.functional.conv2d, torch.nn.functional.conv3d}:
            if len(args) >= 2 and isinstance(args[1], cls):  # weight is second arg
                try:
                    weight_tensor = args[1]
                    dequantized_weight = weight_tensor.dequantize(device=args[0].device, dtype=args[0].dtype)
                    new_args = (args[0], dequantized_weight) + args[2:]
                    return func(*new_args, **kwargs)
                except Exception as e:
                    if debug:
                        debug.log(f"Error in conv dequantization: {e}", level="WARNING", category="dit", force=True)
                    raise
        
        # For ALL other operations, delegate to parent WITHOUT dequantization
        # This includes .cpu(), .to(), .device, .dtype, .shape, etc.
        return super().__torch_function__(func, types, args, kwargs)


def prepare_model_structure(
    runner: VideoDiffusionInfer,
    model_type: str,
    checkpoint_path: str,
    config: OmegaConf,
    debug: 'Debug',
    block_swap_config: Optional[Dict[str, Any]] = None
) -> VideoDiffusionInfer:
    """
    Prepare model structure on meta device without loading weights.
    This uses zero memory as meta device doesn't allocate real memory.
    
    Args:
        runner: VideoDiffusionInfer instance
        model_type: "dit" or "vae"
        checkpoint_path: Path to checkpoint (stored for later loading)
        config: Model configuration
        debug: Debug instance for logging (required)
        block_swap_config: BlockSwap config (stored for DiT, optional)
        
    Returns:
        runner: Updated runner with model structure on meta device
    """
    if debug is None:
        raise ValueError(f"Debug instance required for prepare_model_structure")
    
    is_dit = (model_type == "dit")
    model_type_upper = "DiT" if is_dit else "VAE"
    model_config = config.dit.model if is_dit else config.vae.model
    
    # Always create on meta device for zero memory usage
    debug.log(f"Creating {model_type_upper} model structure on meta device", 
             category=model_type, force=True)
    debug.start_timer(f"{model_type}_structure")

    # comfy_quant packs (INT8 / NVFP4) need construction-time mixed_precision_ops
    # so load_state_dict hits _load_quantized_module (not post-load Linear replace).
    create_kwargs = {}
    if is_dit:
        ops = _dit_comfy_quant_ops(checkpoint_path, torch.float16)
        if ops is not None:
            create_kwargs["operations"] = ops
            fmt = "NVFP4" if checkpoint_is_nvfp4(checkpoint_path) else "INT8"
            debug.log(
                f"{fmt} detected: injecting comfy.ops.mixed_precision_ops at DiT construction",
                category=model_type,
                force=True,
            )
    
    with torch.device("meta"):
        model = create_object(model_config, **create_kwargs)
    
    debug.end_timer(f"{model_type}_structure", f"{model_type_upper} structure created")
    
    # Store model and config for later materialization
    if is_dit:
        runner.dit = model
        runner._dit_checkpoint = checkpoint_path
        runner._dit_block_swap_config = block_swap_config
        runner._dit_comfy_quant_native = bool(create_kwargs)
    else:
        runner.vae = model  
        runner._vae_checkpoint = checkpoint_path
    
    return runner


def materialize_model(runner: VideoDiffusionInfer, model_type: str, device: torch.device, 
                     config: OmegaConf, debug: 'Debug') -> None:
    """
    Materialize model weights from checkpoint to memory.
    Call this right before the model is needed.
    
    Args:
        runner: Runner with model structure on meta device
        model_type: "dit" or "vae"
        device: Target device for inference (torch.device object)
        config: Full configuration
        debug: Debug instance
    """
    if debug is None:
        raise ValueError(f"Debug instance required for materialize_model")
        
    is_dit = (model_type == "dit")
    model_type_upper = "DiT" if is_dit else "VAE"
    
    # Get model and checkpoint path
    if is_dit:
        model = runner.dit
        checkpoint_path = runner._dit_checkpoint
        block_swap_config = runner._dit_block_swap_config
        override_dtype = getattr(runner, '_dit_dtype_override', None)
    else:
        model = runner.vae
        checkpoint_path = runner._vae_checkpoint
        block_swap_config = None
        override_dtype = getattr(runner, '_vae_dtype_override', None)
    
    # Check if already materialized
    if model is None:
        debug.log(f"No {model_type_upper} model structure found", level="WARNING", category=model_type, force=True)
        return
    param_device = next(model.parameters()).device
    if param_device.type != 'meta':
        debug.log(f"{model_type_upper} already materialized on {model.device}", category=model_type)
        return
    
    # Determine target device for materialization
    offload_device_str = None
    if hasattr(runner, f'_{model_type}_offload_device'):
        offload_device_str = getattr(runner, f'_{model_type}_offload_device')

    # If offload_device is set and not "none", materialize to offload device
    if offload_device_str and offload_device_str != "none":
        target_device = torch.device(offload_device_str)
        offload_reason = " (offload device)"
    else:
        # Otherwise materialize to inference device
        target_device = device
        offload_reason = ""
    
    # Start materialization
    debug.start_timer(f"{model_type}_materialize")
    
    # Load weights (this materializes from meta to target device)
    model = _load_model_weights(model, checkpoint_path, target_device, True,
                               model_type_upper, offload_reason, debug, override_dtype) 
   
    # Apply model-specific configurations (includes BlockSwap and torch.compile)
    # Import here to avoid circular dependency 
    from .model_configuration import apply_model_specific_config
    model = apply_model_specific_config(model, runner, config, is_dit, debug)
    
    debug.end_timer(f"{model_type}_materialize", f"{model_type_upper} materialized")
    
    # Clean up checkpoint paths (no longer needed after weights are loaded)
    # Note: Config attributes (_dit_block_swap_config, _dit_compile_args) are preserved
    # for configuration change detection on subsequent runs
    if is_dit:
        runner._dit_checkpoint = None
        runner._dit_dtype_override = None
    else:
        runner._vae_checkpoint = None
        runner._vae_dtype_override = None


def _load_model_weights(model: torch.nn.Module, checkpoint_path: str, target_device: torch.device, 
                        used_meta: bool, model_type: str, cpu_reason: str, 
                        debug: Optional['Debug'] = None, override_dtype: Optional[torch.dtype] = None) -> torch.nn.Module:
    """
    Load model weights from checkpoint file with optimized GGUF support.
    
    For meta-initialized models, materializes to target device.
    For standard models, loads weights and applies state dict.
    
    Args:
        model: Model instance (may be on meta device)
        checkpoint_path: Path to checkpoint file
        target_device: Target device for weights (torch.device object)
        used_meta: Whether model was created on meta device
        model_type: Model type string for logging
        cpu_reason: Reason string if using CPU
        debug: Debug instance
        override_dtype: Optional dtype override for weights
        
    Returns:
        Model with loaded weights
    """
    model_type_lower = model_type.lower()
    
    # Log loading action
    action = "Materializing" if used_meta else "Loading"
    target_device_str = str(target_device).upper()
    debug.log(f"{action} {model_type} weights to {target_device_str}{cpu_reason}: {checkpoint_path}", 
             category=model_type_lower, force=True)
    
    # Load state dict from file
    debug.start_timer(f"{model_type_lower}_weights_load")
    state = load_quantized_state_dict(checkpoint_path, target_device, debug)
    debug.end_timer(f"{model_type_lower}_weights_load", f"{model_type} weights loaded from file")

    # INT8 / NVFP4: comfy.ops parses comfy_quant via .numpy() (CPU only), and
    # meta-built mixed_precision Linear needs factory_kwargs["device"] set to
    # the materialization target so QuantizedTensor is not left on meta.
    if model_type_lower == "dit" and _dit_needs_comfy_quant_prep(checkpoint_path):
        prepare_hswq_state_dict_for_comfy_ops(state)
        n_patch = patch_ops_factory_device(model, target_device)
        fmt = "NVFP4" if checkpoint_is_nvfp4(checkpoint_path) else "INT8"
        debug.log(
            f"{fmt} load prep: comfy_quant→CPU, factory_kwargs device={target_device} "
            f"({n_patch} modules)",
            category=model_type_lower,
            force=True,
        )
    
    # Apply dtype conversion if requested
    if override_dtype is not None:
        state = _convert_state_dtype(state, override_dtype, model_type, debug)
    
    # Log weight statistics
    _log_weight_stats(state, used_meta, model_type, debug)
    
    # Handle GGUF or standard loading
    if checkpoint_path.endswith('.gguf'):
        model = _load_gguf_weights(model, state, used_meta, model_type_lower, debug)
    else:
        model = _load_standard_weights(model, state, used_meta, model_type, model_type_lower, debug)
    
    # Clean up state dict
    del state
    
    # Initialize meta buffers if needed
    if used_meta:
        initialize_meta_buffers(model, target_device, debug)
    
    return model


def _convert_state_dtype(state: Dict[str, torch.Tensor], target_dtype: torch.dtype, 
                        model_type: str, debug: Optional['Debug'] = None) -> Dict[str, torch.Tensor]:
    """Convert floating point tensors in state dict to target dtype."""
    debug.log(f"Converting {model_type} weights to {target_dtype} during loading", category="precision")
    debug.start_timer(f"{model_type.lower()}_dtype_convert")
    
    for key in state:
        if torch.is_tensor(state[key]) and state[key].is_floating_point():
            state[key] = state[key].to(target_dtype)
    
    debug.end_timer(f"{model_type.lower()}_dtype_convert", f"{model_type} weights converted to {target_dtype}")
    return state


def _log_weight_stats(state: Dict[str, torch.Tensor], used_meta: bool, model_type: str, debug: Optional['Debug'] = None) -> None:
    """Log statistics about loaded weights."""
    num_params = len(state)
    total_size_mb = sum(p.nelement() * p.element_size() for p in state.values()) / (1024 * 1024)
    action = "Materializing" if used_meta else "Applying"
    debug.log(f"{action} {model_type}: {num_params} parameters, {total_size_mb:.2f}MB total", 
             category=model_type.lower())


def apply_gguf_parameters(model: torch.nn.Module, state: Dict[str, torch.Tensor], 
                           model_state: Dict[str, torch.Tensor], debug: Optional['Debug'] = None) -> Dict[str, Any]:
    """
    Apply GGUF parameters to model, handling both meta and materialized models.
    
    Returns:
        Statistics dictionary with loaded count, quantized count, and parameter names
    """
    loaded_names = set()
    quantized_count = 0
    
    for name, param in state.items():
        if name not in model_state:
            continue
            
        model_param = model_state[name]
        param_shape = _get_tensor_shape(param)
        
        if param_shape != model_param.shape:
            debug.log(f"Unexpected shape mismatch for {name}: {param_shape} vs {model_param.shape}", 
                     level="ERROR", category="dit", force=True)
            raise ValueError(f"Shape mismatch for parameter {name}")
        
        # Apply parameter based on device type
        with torch.no_grad():
            if model_param.device.type == 'meta':
                _set_parameter_on_meta_model(model, name, param, debug)
            else:
                _set_parameter_on_materialized_model(model, name, param, debug)
        
        loaded_names.add(name)
        if _is_quantized_tensor(param):
            quantized_count += 1
    
    return {
        'loaded': len(loaded_names), 
        'quantized': quantized_count, 
        'loaded_names': loaded_names
    }


def _set_parameter_on_meta_model(model: torch.nn.Module, param_name: str, 
                                 param_value: torch.Tensor, debug: Optional['Debug'] = None) -> None:
    """Set parameter on meta device model."""
    module, attr_name = _navigate_to_parameter(model, param_name)
    new_param = _create_gguf_parameter(param_value, debug)
    setattr(module, attr_name, new_param)


def _set_parameter_on_materialized_model(model: torch.nn.Module, param_name: str, 
                                         param_value: torch.Tensor, debug: Optional['Debug'] = None) -> None:
    """Set parameter on already materialized model."""
    module, attr_name = _navigate_to_parameter(model, param_name)
    
    if _is_quantized_tensor(param_value):
        # For quantized tensors, replace with wrapped parameter
        new_param = _create_gguf_parameter(param_value, debug)
        setattr(module, attr_name, new_param)
    else:
        # For regular tensors, just copy
        existing_param = getattr(module, attr_name)
        existing_param.copy_(param_value)


def _navigate_to_parameter(model: torch.nn.Module, param_path: str) -> Tuple[torch.nn.Module, str]:
    """
    Navigate to the module containing a parameter.
    
    Args:
        model: Root model
        param_path: Dot-separated path to parameter
        
    Returns:
        Tuple of (parent module, parameter name)
    """
    path_parts = param_path.split('.')
    module = model
    
    # Navigate to parent module
    for part in path_parts[:-1]:
        module = getattr(module, part)
    
    return module, path_parts[-1]


def _create_gguf_parameter(tensor: torch.Tensor, debug: Optional['Debug'] = None) -> torch.nn.Parameter:
    """
    Create a parameter from a GGUF tensor, preserving quantization info.
    
    Args:
        tensor: GGUF tensor (may be quantized)
        debug: Debug instance for logging
        
    Returns:
        Parameter with GGUF attributes and dequantize method if quantized
    """
    param = torch.nn.Parameter(tensor, requires_grad=False)
    
    # Preserve GGUF attributes if present
    if hasattr(tensor, 'tensor_type'):
        param.tensor_type = tensor.tensor_type
        param.tensor_shape = tensor.tensor_shape
        
        # Add dequantize method for runtime dequantization
        param.gguf_dequantize = _create_dequantize_method(tensor, debug)
    
    return param


def _get_tensor_shape(tensor: torch.Tensor) -> torch.Size:
    """Get the logical shape of a tensor (handling GGUF quantized tensors)."""
    if hasattr(tensor, 'tensor_shape'):
        return tensor.tensor_shape
    return tensor.shape


def _is_quantized_tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is GGUF quantized."""
    return hasattr(tensor, 'tensor_type') and hasattr(tensor, 'tensor_shape')


def _report_parameter_mismatches(state: Dict[str, torch.Tensor], 
                                 model_state: Dict[str, torch.Tensor], 
                                 loaded_names: set, debug: Optional['Debug'] = None) -> None:
    """Report any parameter mismatches between GGUF and model."""
    # Check for unmatched GGUF parameters
    unmatched = [name for name in state if name not in model_state]
    if unmatched:
        debug.log(f"Warning: {len(unmatched)} parameters from GGUF not found in model", 
                 level="WARNING", category="dit", force=True)
        debug.log(f"First few unmatched: {unmatched[:5]}", level="WARNING", category="dit", force=True, indent_level=1)
    
    # Check for missing model parameters  
    missing = [name for name in model_state if name not in loaded_names]
    if missing:
        debug.log(f"Warning: {len(missing)} model parameters not loaded from GGUF", 
                 level="WARNING", category="dit", force=True)
        debug.log(f"First few missing: {missing[:5]}", level="WARNING", category="dit", force=True, indent_level=1)


def initialize_meta_buffers(model: torch.nn.Module, target_device: torch.device, debug: Optional['Debug'] = None) -> None:
    """Initialize meta buffers with timing."""
    debug.start_timer("buffer_init")
    initialized = initialize_meta_buffers_impl(model, target_device, debug)
    if initialized > 0:
        debug.log(f"Initialized {initialized} non-persistent buffers", category="success")
    debug.end_timer("buffer_init", "Buffer initialization")


def initialize_meta_buffers_impl(model: torch.nn.Module, target_device: torch.device, debug: Optional['Debug'] = None) -> int:
    """
    Initialize any buffers still on meta device after materialization.
    
    Non-persistent buffers aren't included in state_dict and remain on meta
    device after load_state_dict. This function moves them to the target device.
    
    Args:
        model: Model potentially containing meta device buffers
        target_device: Target device for initialization (torch.device object)
        debug: Debug instance for logging
        
    Returns:
        Number of buffers initialized
    """
    initialized_count = 0
    
    # Simply initialize all meta device buffers to zeros on target device
    for name, buffer in model.named_buffers():
        if buffer is not None and buffer.device.type == 'meta':
            # Get the module that owns this buffer
            module_path = name.rsplit('.', 1)[0] if '.' in name else ''
            buffer_name = name.rsplit('.', 1)[1] if '.' in name else name
            
            # Get the actual module
            if module_path:
                module = model
                for part in module_path.split('.'):
                    module = getattr(module, part)
            else:
                module = model
            
            # Create a zero tensor of the same shape on target device
            # This is safe for all non-persistent buffers (caches, dummy tensors, etc.)
            initialized_buffer = torch.zeros_like(buffer, device=target_device)
            module.register_buffer(buffer_name, initialized_buffer, persistent=False)
            initialized_count += 1
    
    return initialized_count


def _load_standard_weights(model: torch.nn.Module, state: Dict[str, torch.Tensor], 
                          used_meta: bool, model_type: str, model_type_lower: str,
                          debug: Optional['Debug'] = None) -> torch.nn.Module:
    """Load standard (non-GGUF) weights into model."""
    debug.start_timer(f"{model_type_lower}_state_apply")
    model.load_state_dict(state, strict=False, assign=True)
    
    action = "materialized" if used_meta else "applied"
    debug.end_timer(f"{model_type_lower}_state_apply", f"{model_type} weights {action}")
    
    if used_meta:
        debug.log(f"{model_type} materialized directly from meta with loaded weights", category=model_type_lower)
    else:
        debug.log(f"{model_type} weights applied", category=model_type_lower)
    
    return model


def _load_gguf_weights(model: torch.nn.Module, state: Dict[str, torch.Tensor], 
                      used_meta: bool, model_type_lower: str, debug: Optional['Debug'] = None) -> torch.nn.Module:
    """
    Load GGUF quantized weights into model with architecture validation.
    
    Args:
        model: Target model
        state: GGUF state dict with quantized tensors
        used_meta: Whether model was initialized on meta device
        model_type_lower: Lowercase model type for logging
        debug: Debug instance
        
    Returns:
        Model with GGUF weights loaded
    """
    debug.log("Loading GGUF weights", category="dit")
    
    # Get model state dict for validation
    model_state = model.state_dict()
    
    # Validate architecture compatibility
    _validate_gguf_architecture(state, model_state, debug)
    
    # Load GGUF parameters
    stats = apply_gguf_parameters(model, state, model_state, debug)
    
    # Log results
    debug.log(f"GGUF loading complete: {stats['loaded']} parameters loaded", category="success")
    debug.log(f"Quantized parameters: {stats['quantized']}", category="info")
    
    # Report any mismatches
    _report_parameter_mismatches(state, model_state, stats['loaded_names'], debug)
    
    # Replace Linear/Conv2d layers with quantized versions for optimal precision handling
    if stats['quantized'] > 0:
        debug.log("Replacing layers with GGUF-optimized versions for precision handling", category="dit")
        
        replacements, quant_types = replace_linear_with_quantized(model, debug=debug)
        
        if replacements > 0:
            debug.log(f"Replaced {replacements} layers with GGUF-optimized versions", category="success")
            
            # Show actual quantization types found and precision strategy
            if quant_types:
                qtypes_str = ', '.join([f"{qtype}:{count}" for qtype, count in quant_types.items()])
                debug.log(
                    f"GGUF precision path: {qtypes_str} → FP16 (preserve) → BF16/FP32 (compute)", 
                    category="precision"
                )
            else:
                debug.log(
                    "GGUF precision: Dequantizing to FP16 first, then converting to compute dtype", 
                    category="precision"
                )
        else:
            debug.log("Warning: No layers were replaced despite having quantized parameters", 
                     level="WARNING", category="dit", force=True)
    
    return model


def _validate_gguf_architecture(state: Dict[str, torch.Tensor], 
                                model_state: Dict[str, torch.Tensor], debug: Optional['Debug'] = None) -> None:
    """
    Validate GGUF model architecture matches target model.
    
    Raises:
        ValueError: If architecture mismatch is detected
    """
    key_params = [
        "blocks.0.attn.proj_qkv.vid.weight",
        "blocks.0.attn.proj_qkv.txt.weight", 
        "blocks.0.mlp.vid.proj_in.weight"
    ]

    for key in key_params:
        if key in state and key in model_state:
            model_shape = model_state[key].shape
            gguf_shape = _get_tensor_shape(state[key])
            
            if model_shape != gguf_shape:
                # Check if it's just a quantization difference
                if hasattr(state[key], 'tensor_shape') and state[key].tensor_shape == model_shape:
                    continue
                    
                raise ValueError(
                    f"GGUF model architecture mismatch: This GGUF model is incompatible with the current architecture.\n\n"
                    f"Detected mismatch:\n"
                    f"  Parameter: {key}\n"
                    f"  Expected shape: {model_shape}\n"
                    f"  GGUF shape: {gguf_shape}\n\n"
                    f"Possible solutions:\n"
                    f"1. Use a GGUF model that matches the current architecture\n"
                    f"2. Try using a regular FP16 model instead\n"
                    f"3. Verify you're using the correct model variant (3B vs 7B)"
                )
    
    debug.log(f"Architecture check complete, no shape mismatch", category="success")


def _create_dequantize_method(tensor: torch.Tensor, debug: Optional['Debug'] = None) -> callable:
    """
    Create a dequantization method for a GGUF tensor.
    
    Args:
        tensor: GGUF quantized tensor with tensor_type and tensor_shape attributes
        debug: Debug instance
        
    Returns:
        Callable dequantization method
    """
    def dequantize(device: Optional[torch.device] = None, 
                   dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Dequantize GGUF tensor on demand."""
        if hasattr(tensor, 'dequantize'):
            return tensor.dequantize(device, dtype)
        
        try:
            # Fallback to manual dequantization using gguf library
            numpy_data = tensor.cpu().numpy()
            dequantized = gguf.quants.dequantize(numpy_data, tensor.tensor_type)
            result = torch.from_numpy(dequantized).to(device or tensor.device, dtype)
            result.requires_grad_(False)
            return result.reshape(tensor.tensor_shape)
        except Exception as e:
            if debug:
                debug.log(f"Warning: Could not dequantize tensor: {e}", level="WARNING", category="dit", force=True)
            return tensor.to(device or tensor.device, dtype)
    
    return dequantize
```

### `src/core/generation_phases.py` (complete; includes NVFP4 autocast skip)

```python
"""
Generation Logic Module for SeedVR2

This module implements a four-phase batch processing pipeline for video upscaling:
- Phase 1: Batch VAE encoding of all input frames
- Phase 2: Batch DiT upscaling of all encoded latents
- Phase 3: Batch VAE decoding of all upscaled latents
- Phase 4: Post-processing and final video assembly

This architecture minimizes model swapping overhead by completing each phase
for all batches before moving to the next phase, significantly improving
performance especially when using model offloading.

Key Features:
- Four-phase pipeline (encode-all → upscale-all → decode-all → postprocess-all) for efficiency
- Native FP8 pipeline support for 2x speedup and 50% VRAM reduction
- Temporal overlap support for smooth transitions between batches
- Adaptive dtype detection and configuration
- Memory-efficient pre-allocated batch processing
- Stream-based assembly eliminates memory spikes for long videos
- Advanced video format handling (4n+1 constraint)
- Clean separation of concerns with phase-specific resource management
- Each phase handles its own cleanup in finally blocks
"""

import os
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable

from .generation_utils import (
    setup_video_transform,
    pad_video_temporal,
    check_interrupt,
    ensure_precision_initialized,
    _draw_tile_boundaries,
    load_text_embeddings,
    blend_overlapping_frames,
    calculate_optimal_batch_params,
    script_directory
)
from .model_configuration import apply_model_specific_config
from .model_loader import materialize_model
from .alpha_upscaling import process_alpha_for_batch
from .infer import VideoDiffusionInfer
from ..common.seed import set_seed
from ..optimization.nvfp4_native_ops import checkpoint_is_nvfp4
from ..optimization.memory_manager import (
    cleanup_dit,
    cleanup_vae,
    cleanup_text_embeddings,
    manage_tensor,
    manage_model_device,
    release_tensor_memory,
    release_tensor_collection
)
from ..optimization.performance import (
    optimized_video_rearrange, 
    optimized_single_video_rearrange, 
    optimized_sample_to_image_format
)
from ..utils.color_fix import (
    lab_color_transfer,
    wavelet_adaptive_color_correction,
    hsv_saturation_histogram_match, 
    wavelet_reconstruction,
    adaptive_instance_normalization
)


def _prepare_video_batch(
    images: torch.Tensor,
    start_idx: int,
    end_idx: int,
    uniform_padding: int = 0,
    debug: Optional['Debug'] = None,
    log_info: bool = False
) -> torch.Tensor:
    """
    Extract and prepare video batch with uniform padding and permutation.
    
    Args:
        images: Source video frames [T, H, W, C]
        start_idx: Start frame index
        end_idx: End frame index (exclusive)
        uniform_padding: Number of frames to pad (0 = no padding)
        debug: Debug instance for optional logging
        log_info: If True, log padding operations (used during encoding only)
        
    Returns:
        Prepared video in TCHW format
    """
    # Extract frames (view/slice, not copy)
    video = images[start_idx:end_idx]
    
    # Apply uniform padding if needed
    if uniform_padding > 0:
        if log_info and debug:
            current_frames = end_idx - start_idx
            debug.log(f"Sequence of {current_frames} frames", category="video", force=True, indent_level=1)
            debug.log(f"Padding batch: {uniform_padding} frame{'s' if uniform_padding != 1 else ''} added ({current_frames} → {current_frames + uniform_padding}) for uniform batches", 
                     category="video", force=True, indent_level=1)
        video = pad_video_temporal(video, count=uniform_padding, temporal_dim=0, prepend=False, debug=None)
    
    # Permute to TCHW format
    video = video.permute(0, 3, 1, 2)
    
    return video


def _apply_4n1_padding(video: torch.Tensor) -> torch.Tensor:
    """
    Apply 4n+1 temporal padding constraint required by VAE.
    
    Args:
        video: Video tensor in TCHW format
        
    Returns:
        Padded video in TCHW format
    """
    t = video.size(0)
    if t % 4 != 1:
        video = optimized_single_video_rearrange(video)  # TCHW -> CTHW
        video = pad_video_temporal(video, temporal_dim=1, prepend=False, debug=None)
        video = optimized_single_video_rearrange(video)  # CTHW -> TCHW
    return video


def _reconstruct_and_transform_batch(
    ctx: Dict[str, Any],
    batch_idx: int,
    debug: Optional['Debug'] = None
) -> torch.Tensor:
    """
    Reconstruct and transform a video batch for color correction (Phase 4).
    
    Args:
        ctx: Context with input_images, batch_metadata, video_transform
        batch_idx: Index of batch to reconstruct
        debug: Debug instance for logging
        
    Returns:
        Transformed video in CTHW format, ready for color correction
    """
    start_idx, end_idx, uniform_padding = ctx['batch_metadata'][batch_idx]
    
    # Prepare video batch
    video = _prepare_video_batch(
        images=ctx['input_images'],
        start_idx=start_idx,
        end_idx=end_idx,
        uniform_padding=uniform_padding,
        debug=None,
        log_info=False
    )
    
    # Apply 4n+1 padding using shared helper
    video = _apply_4n1_padding(video)
    
    # Extract RGB and transform
    if ctx.get('is_rgba', False):
        rgb_video = video[:, :3, :, :]
    else:
        rgb_video = video
    
    transformed_video = ctx['video_transform'](rgb_video)
    
    del video
    
    return transformed_video


def encode_all_batches(
    runner: 'VideoDiffusionInfer',
    ctx: Dict[str, Any],
    images: torch.Tensor,
    debug: 'Debug',
    batch_size: int = 5,
    uniform_batch_size: bool = False,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    temporal_overlap: int = 0,
    resolution: int = 1080,
    max_resolution: int = 0,
    input_noise_scale: float = 0.0,
    color_correction: str = "wavelet"
) -> Dict[str, Any]:
    """
    Phase 1: VAE Encoding for all batches
    
    Encodes video frames to latents in batches, handling temporal overlap and 
    memory optimization. Creates context automatically if not provided.
    
    Args:
        runner: VideoDiffusionInfer instance with loaded models (required)
        ctx: Generation context from setup_generation_context (required)
        images: Input frames tensor [T, H, W, C] range [0,1] (required) 
        debug: Debug instance for logging (required)
        batch_size: Frames per batch (4n+1 format: 1, 5, 9, 13...)
        uniform_batch_size: Pad final batch to match batch_size for uniform batches
        seed: Random seed for deterministic VAE sampling (default: 42)
        progress_callback: Optional callback(current, total, frames, phase_name)
        temporal_overlap: Overlapping frames between batches for continuity
        resolution: Target resolution for shortest edge
        max_resolution: Maximum resolution for any edge (0 = no limit)
        input_noise_scale: Scale for input noise (0.0-1.0). Adds noise to input images
                          before VAE encoding to reduce artifacts at high resolutions.
        color_correction: Color correction method - "wavelet", "adain", or "none" (default: "wavelet")
                         Determines if transformed videos need to be stored for later use.
        
    Returns:
        dict: Context containing:
            - batch_metadata: Lightweight indices for on-demand transform reconstruction
            - all_latents: List of encoded latents ready for upscaling
            - Other state for subsequent phases
            
    Raises:
        ValueError: If required inputs are missing or invalid
        RuntimeError: If encoding fails
    """
    if debug is None:
        raise ValueError("Debug instance must be provided to encode_all_batches")
    
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 1: VAE encoding ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase1_encoding")

    # Context must be provided
    if ctx is None:
        raise ValueError("Generation context must be provided to encode_all_batches")
    
    # Validate and store inputs
    if images is None:
        raise ValueError("Images to encode must be provided")
    else:
        # MPS: keep on device to avoid sync overhead in Phase 4 color correction
        if ctx['vae_device'].type == 'mps' and images.device.type != 'mps':
            ctx['input_images'] = images.to(ctx['vae_device'])
        else:
            ctx['input_images'] = images
    
    # Get total frame count from context (set in video_upscaler before encoding)
    total_frames = ctx.get('total_frames', len(images))
    
    # Set it if not already set (for standalone/CLI usage)
    if 'total_frames' not in ctx:
        ctx['total_frames'] = total_frames
    
    if total_frames == 0:
        raise ValueError("No frames to process")
    
    # Setup video transformation pipeline and compute dimensions if not already done
    if 'true_target_dims' not in ctx:
        sample_frame = images[0].permute(2, 0, 1).unsqueeze(0)
        setup_video_transform(ctx, resolution, max_resolution, debug, sample_frame)
        del sample_frame
    else:
        setup_video_transform(ctx, resolution, max_resolution, debug)
    
    # Detect if input is RGBA (4 channels)
    ctx['is_rgba'] = images[0].shape[-1] == 4
    
    # Display batch optimization tip if applicable
    if total_frames > 0:
        batch_params = calculate_optimal_batch_params(total_frames, batch_size, temporal_overlap)
        if batch_params['best_batch'] != batch_size and batch_params['best_batch'] <= total_frames:
            debug.log("", category="none", force=True)
            debug.log(f"Tip: For {total_frames} frames, batch_size={batch_params['best_batch']} matches video length optimally", category="tip", force=True)
            debug.log(f"Matching batch_size to shot length improves temporal coherence", category="tip", force=True, indent_level=1)
            debug.log("", category="none", force=True)
    
    # Calculate batching parameters
    step = batch_size - temporal_overlap if temporal_overlap > 0 else batch_size
    if step <= 0:
        step = batch_size
        temporal_overlap = 0
        debug.log(f"temporal_overlap >= batch_size, resetting to 0", level="WARNING", category="setup", force=True)
    
    # Store actual temporal overlap used (may differ from parameter if reset)
    ctx['actual_temporal_overlap'] = temporal_overlap
    
    # Calculate number of batches
    num_encode_batches = 0
    for idx in range(0, total_frames, step):
        end_idx = min(idx + batch_size, total_frames)
        if idx > 0 and end_idx - idx <= temporal_overlap:
            break
        num_encode_batches += 1
    
    # Pre-allocate lists for memory efficiency
    ctx['all_latents'] = [None] * num_encode_batches
    ctx['all_ori_lengths'] = [None] * num_encode_batches
    if color_correction != "none":
        ctx['batch_metadata'] = [None] * num_encode_batches
    
    encode_idx = 0
    
    try:
        # Materialize VAE if still on meta device
        if runner.vae and next(runner.vae.parameters()).device.type == 'meta':
            materialize_model(runner, "vae", ctx['vae_device'], runner.config, debug)
        else:
            # Model already materialized (cached) - apply any pending configs if needed
            if getattr(runner, '_vae_config_needs_application', False):
                debug.log("Applying updated VAE configuration", category="vae", force=True)
                apply_model_specific_config(runner.vae, runner, runner.config, False, debug)
        
        # Initialize precision after VAE is materialized with actual weights
        ensure_precision_initialized(ctx, runner, debug)

        # Cache VAE now that it's fully configured and ready for inference
        if ctx['cache_context']['vae_cache'] and not ctx['cache_context']['cached_vae']:
            runner.vae._model_name = ctx['cache_context']['vae_model']
            ctx['cache_context']['global_cache'].set_vae(
                {'node_id': ctx['cache_context']['vae_id'], 'cache_model': True}, 
                runner.vae, ctx['cache_context']['vae_model'], debug
            )
            ctx['cache_context']['vae_newly_cached'] = True
            
            # If both models now cached, cache runner template
            dit_is_cached = ctx['cache_context']['cached_dit'] or ctx['cache_context']['dit_newly_cached']
            if dit_is_cached:
                ctx['cache_context']['global_cache'].set_runner(
                    ctx['cache_context']['dit_id'], ctx['cache_context']['vae_id'], 
                    runner, debug
                )
        
        # Set deterministic seed for VAE encoding (separate from diffusion noise)
        # Uses seed + 1,000,000 to avoid collision with upscaling batch seeds
        # This ensures VAE sampling is deterministic while maintaining quality
        seed_vae = seed + 1000000
        set_seed(seed_vae)
        debug.log(f"Using seed: {seed_vae} (VAE uses seed+1000000 for deterministic sampling)", category="vae")
        
        # Move VAE to GPU for encoding (no-op if already there)
        manage_model_device(model=runner.vae, target_device=ctx['vae_device'], 
                          model_name="VAE", debug=debug, runner=runner)
        
        debug.log_memory_state("After VAE loading for encoding", detailed_tensors=False)

        # Initialize tile_boundaries for encoding debug
        if runner.tile_debug == "encode" and runner.encode_tiled:
            debug.encode_tile_boundaries = []
            debug.log("Tile debug enabled: encode tile boundaries will be visualized", category="vae", force=True)
            debug.log("Remember to disable --tile_debug in production to remove overlay visualization", category="tip", indent_level=1, force=True)
        
        # Process encoding
        for batch_idx in range(0, total_frames, step):
            check_interrupt(ctx)
            
            # Calculate indices with temporal overlap
            if batch_idx == 0:
                start_idx = 0
                end_idx = min(batch_size, total_frames)
            else:
                start_idx = batch_idx
                end_idx = min(start_idx + batch_size, total_frames)
                if end_idx - start_idx <= temporal_overlap:
                    break
            
            current_frames = end_idx - start_idx
            is_uniform_padding = uniform_batch_size and current_frames < batch_size
            
            debug.log(f"Encoding batch {encode_idx+1}/{num_encode_batches}", category="vae", force=True)
            debug.start_timer(f"encode_batch_{encode_idx+1}")
            
            # Save original length before any padding
            ori_length = current_frames
            
            # Prepare video batch with uniform padding
            video = _prepare_video_batch(
                images=images,
                start_idx=start_idx,
                end_idx=end_idx,
                uniform_padding=batch_size - current_frames if is_uniform_padding else 0,
                debug=debug,
                log_info=True
            )
            if is_uniform_padding:
                current_frames = batch_size
            
            video = manage_tensor(
                tensor=video,
                target_device=ctx['vae_device'],
                tensor_name=f"video_batch_{encode_idx+1}",
                dtype=ctx['compute_dtype'],
                debug=debug,
                reason="VAE encoding",
                indent_level=1
            )

            # Check temporal dimension for 4n+1 padding
            t = video.size(0)
            
            # Log sequence size if not already logged (for non-uniform batches)
            if not is_uniform_padding:
                debug.log(f"Sequence of {t} frames", category="video", force=True, indent_level=1)

            # Apply 4n+1 padding using shared helper
            if t % 4 != 1:
                target = ((t-1)//4+1)*4+1
                padding_frames = target - t
                debug.log(f"Padding batch: {padding_frames} frame{'s' if padding_frames != 1 else ''} added ({t} → {target}) to meet 4n+1 constraint", 
                         category="video", force=True, indent_level=1)
                # Apply 4n+1 padding to match exact frame count from encoding
                video = _apply_4n1_padding(video)

            # Apply transformations (matches reconstruction logic)
            if ctx.get('is_rgba', False):
                debug.log(f"Extracted Alpha channel for edge-guided upscaling", category="alpha", indent_level=1)
                rgb_video = video[:, :3, :, :]
            else:
                rgb_video = video

            transformed_video = ctx['video_transform'](rgb_video)

            # Apply input noise if requested (to reduce artifacts at high resolutions)
            if input_noise_scale > 0:
                debug.log(f"Applying input noise (scale: {input_noise_scale:.2f})", category="video", indent_level=1)
                
                # Generate noise matching the video shape
                noise = torch.randn_like(transformed_video)
                
                # Subtle noise amplitude
                noise = noise * 0.05
                
                # Linear blend factor: 0 at scale=0, 0.5 at scale=1
                blend_factor = input_noise_scale * 0.5
                
                # Apply blend
                transformed_video = transformed_video * (1 - blend_factor) + (transformed_video + noise) * blend_factor
                
                del noise

            # Store original length for proper trimming later
            ctx['all_ori_lengths'][encode_idx] = ori_length

            # Store batch frame indices for on-demand reconstruction
            if color_correction != "none":
                ctx['batch_metadata'][encode_idx] = (start_idx, end_idx, batch_size - ori_length if is_uniform_padding else 0)
            
            # Extract and store Alpha and RGB from padded original video (before encoding)
            if ctx.get('is_rgba', False):
                if 'all_alpha_channels' not in ctx:
                    ctx['all_alpha_channels'] = [None] * num_encode_batches
                if 'all_input_rgb' not in ctx:
                    ctx['all_input_rgb'] = [None] * num_encode_batches
                
                # Extract from padded RGBA video (format: T, 4, H, W)
                alpha_channel = video[:, 3:4, :, :]
                rgb_video_original = video[:, :3, :, :]
                
                # Store on tensor_offload_device to save VRAM (or keep on device if none)
                if ctx['tensor_offload_device'] is not None:
                    ctx['all_alpha_channels'][encode_idx] = manage_tensor(
                        tensor=alpha_channel,
                        target_device=ctx['tensor_offload_device'],
                        tensor_name=f"alpha_channel_{encode_idx+1}",
                        debug=debug,
                        reason="storing Alpha channel for upscaling",
                        indent_level=1
                    )
                    ctx['all_input_rgb'][encode_idx] = manage_tensor(
                        tensor=rgb_video_original,
                        target_device=ctx['tensor_offload_device'],
                        tensor_name=f"rgb_original_{encode_idx+1}",
                        debug=debug,
                        reason="storing RGB edge guidance for Alpha upscaling",
                        indent_level=1
                    )
                else:
                    ctx['all_alpha_channels'][encode_idx] = alpha_channel
                    ctx['all_input_rgb'][encode_idx] = rgb_video_original
                
                del alpha_channel, rgb_video_original

            del video

            # Move to VAE device with correct dtype for encoding
            transformed_video = manage_tensor(
                tensor=transformed_video,
                target_device=ctx['vae_device'],
                tensor_name=f"transformed_video_{encode_idx+1}",
                dtype=ctx['compute_dtype'],
                debug=debug,
                reason="VAE encoding",
                indent_level=1
            )
            
            # Encode to latents
            cond_latents = runner.vae_encode([transformed_video])

            # Don't store transformed_video - will reconstruct on-demand in Phase 4
            del transformed_video, rgb_video
            
            # Convert from VAE dtype to compute dtype and offload to avoid VRAM accumulation
            if ctx['tensor_offload_device'] is not None and (cond_latents[0].is_cuda or cond_latents[0].is_mps):
                ctx['all_latents'][encode_idx] = manage_tensor(
                    tensor=cond_latents[0],
                    target_device=ctx['tensor_offload_device'],
                    tensor_name=f"latent_{encode_idx+1}",
                    dtype=ctx['compute_dtype'],
                    debug=debug,
                    reason="storing encoded latents for upscaling",
                    indent_level=1
                )
            else:
                # Stay on current device but convert to compute dtype
                ctx['all_latents'][encode_idx] = manage_tensor(
                    tensor=cond_latents[0],
                    target_device=cond_latents[0].device,
                    tensor_name=f"latent_{encode_idx+1}",
                    dtype=ctx['compute_dtype'],
                    debug=debug,
                    reason="VAE dtype → compute dtype",
                    indent_level=1
                )
            
            del cond_latents
            
            debug.end_timer(f"encode_batch_{encode_idx+1}", f"Encoded batch {encode_idx+1}")
            
            if progress_callback:
                progress_callback(encode_idx+1, num_encode_batches, 
                                current_frames, "Phase 1: Encoding")
            
            encode_idx += 1
            
    except Exception as e:
        debug.log(f"Error in Phase 1 (Encoding): {e}", level="ERROR", category="error", force=True)
        raise
    finally:
        # Offload VAE to configured offload device if specified
        if ctx['vae_offload_device'] is not None:
            manage_model_device(model=runner.vae, target_device=ctx['vae_offload_device'], 
                                model_name="VAE", debug=debug, reason="VAE offload", runner=runner)
    
    debug.end_timer("phase1_encoding", "Phase 1: VAE encoding complete", show_breakdown=True)
    debug.log_memory_state("After phase 1 (VAE encoding)", show_tensors=False)
    
    return ctx


def upscale_all_batches(
    runner: 'VideoDiffusionInfer',
    ctx: Dict[str, Any],
    debug: 'Debug',
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    seed: int = 42,
    latent_noise_scale: float = 0.0,
    cache_model: bool = False
) -> Dict[str, Any]:
    """
    Phase 2: DiT Upscaling for all encoded batches.
    
    Processes all encoded latents through the diffusion model for upscaling.
    Requires context from encode_all_batches with encoded latents.
    
    Args:
        runner: VideoDiffusionInfer instance with loaded models (required)
        ctx: Context from encode_all_batches containing latents (required)
        debug: Debug instance for logging (required)
        progress_callback: Optional callback(current, total, frames, phase_name)
        seed: Random seed for reproducible generation
        latent_noise_scale: Noise scale for latent space augmentation (0.0-1.0).
                           Adds noise during diffusion conditioning. Can soften details
                           but may help with certain artifacts. 0.0 = no noise (crisp),
                           1.0 = maximum noise (softer)
        cache_model: If True, keep DiT model for reuse instead of deleting it
        
    Returns:
        dict: Updated context containing:
            - all_upscaled_latents: List of upscaled latents ready for decoding
            - Preserved state from encoding phase
            
    Raises:
        ValueError: If context is missing or has no encoded latents
        RuntimeError: If upscaling fails
    """
    if debug is None:
        raise ValueError("Debug instance must be provided to upscale_all_batches")
    
    if ctx is None:
        raise ValueError("Context is required for upscale_all_batches. Run encode_all_batches first.")
        
    # Validate we have encoded latents
    if 'all_latents' not in ctx or not ctx['all_latents']:
        raise ValueError("No encoded latents found. Run encode_all_batches first.")
    
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 2: DiT upscaling ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase2_upscaling")
    
    # Load text embeddings if not already loaded
    if ctx.get('text_embeds') is None:
        ctx['text_embeds'] = load_text_embeddings(script_directory, ctx['dit_device'], ctx['compute_dtype'], debug)
        debug.log("Loaded text embeddings for DiT", category="dit")
    
    # Configure diffusion parameters
    # Force cfg_scale = 1.0 for one-step distilled models (CFG is incompatible with distillation)
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion(device=ctx['dit_device'], dtype=ctx['compute_dtype'])

    # Count valid latents
    num_valid_latents = len([l for l in ctx['all_latents'] if l is not None])

    # Safety check for empty latents
    if num_valid_latents == 0:
        debug.log("No valid latents to upscale", level="WARNING", category="dit", force=True)
        ctx['all_upscaled_latents'] = []
        return ctx
    
    # Pre-allocate list for upscaled latents
    ctx['all_upscaled_latents'] = [None] * num_valid_latents
    
    upscale_idx = 0
    
    try:
        # Materialize DiT if still on meta device
        if runner.dit and next(runner.dit.parameters()).device.type == 'meta':
            materialize_model(runner, "dit", ctx['dit_device'], runner.config, debug)
        else:
            # Model already materialized (cached) - apply any pending configs if needed
            if getattr(runner, '_dit_config_needs_application', False):
                debug.log("Applying updated DiT configuration", category="dit", force=True)
                apply_model_specific_config(runner.dit, runner, runner.config, True, debug)
    
        # Initialize precision after DiT is materialized with actual weights
        ensure_precision_initialized(ctx, runner, debug)

        # Cache DiT now that it's fully configured and ready for inference
        if ctx['cache_context']['dit_cache'] and not ctx['cache_context']['cached_dit']:
            runner.dit._model_name = ctx['cache_context']['dit_model']
            ctx['cache_context']['global_cache'].set_dit(
                {'node_id': ctx['cache_context']['dit_id'], 'cache_model': True}, 
                runner.dit, ctx['cache_context']['dit_model'], debug
            )
            ctx['cache_context']['dit_newly_cached'] = True
            
            # If both models now cached, cache runner template
            vae_is_cached = ctx['cache_context']['cached_vae'] or ctx['cache_context']['vae_newly_cached']
            if vae_is_cached:
                ctx['cache_context']['global_cache'].set_runner(
                    ctx['cache_context']['dit_id'], ctx['cache_context']['vae_id'], 
                    runner, debug
                )
        
        # Move DiT to GPU for upscaling (no-op if already there)
        manage_model_device(model=runner.dit, target_device=ctx['dit_device'], 
                            model_name="DiT", debug=debug, runner=runner)

        debug.log_memory_state("After DiT loading for upscaling", detailed_tensors=False)

        for batch_idx, latent in enumerate(ctx['all_latents']):
            if latent is None:
                continue
            
            check_interrupt(ctx)
            
            debug.log(f"Upscaling batch {upscale_idx+1}/{num_valid_latents}", category="generation", force=True)
            # Reset seed for each batch to ensure identical RNG state
            # This ensures identical inputs produce identical outputs regardless of batch position
            set_seed(seed)
            debug.log(f"Using seed: {seed} for deterministic generation", category="dit")

            debug.start_timer(f"upscale_batch_{upscale_idx+1}")
            
            # Move to DiT device with correct dtype for upscaling (no-op if already there)
            latent = manage_tensor(
                tensor=latent,
                target_device=ctx['dit_device'],
                tensor_name=f"latent_{upscale_idx+1}",
                dtype=ctx['compute_dtype'],
                debug=debug,
                reason="DiT upscaling",
                indent_level=1
            )

            # Generate noise (randn_like automatically uses latent's device)
            base_noise = torch.randn_like(latent, dtype=ctx['compute_dtype'])
            
            noises = [base_noise]
            aug_noises = [base_noise * 0.1 + torch.randn_like(base_noise) * 0.05]
            
            # Log latent noise application if enabled
            if latent_noise_scale > 0:
                debug.log(f"Applying latent noise (scale: {latent_noise_scale:.3f})", category="generation")
            
            def _add_noise(x, aug_noise):
                if latent_noise_scale == 0.0:
                    return x
                t = torch.tensor([1000.0], device=ctx['dit_device'], dtype=ctx['compute_dtype']) * latent_noise_scale
                shape = torch.tensor(x.shape[1:], device=ctx['dit_device'])[None]
                t = runner.timestep_transform(t, shape)
                x = runner.schedule.forward(x, aug_noise, t)
                del t, shape
                return x
            
            # Generate condition
            condition = runner.get_condition(
                noises[0],
                task="sr",
                latent_blur=_add_noise(latent, aug_noises[0]),
            )
            conditions = [condition]
            
            # Detect DiT model dtype (handle CompatibleDiT wrapper)
            dit_model = runner.dit.dit_model if hasattr(runner.dit, 'dit_model') else runner.dit
            try:
                dit_dtype = next(dit_model.parameters()).dtype
            except StopIteration:
                dit_dtype = ctx['compute_dtype']  # Fallback for meta device or empty model
            
            # Use autocast if DiT dtype differs from compute dtype.
            # Skip autocast on MPS (CompatibleDiT already handles dtype conversion).
            # Skip autocast for native NVFP4: ComfyUI UNet/Flux keeps activations in
            # FP16/BF16 without wrapping the whole forward in autocast. Under autocast,
            # LayerNorm/RMSNorm emit float32, and comfy_kitchen CUDA quantize_nvfp4
            # rejects dtype code 0 (float32) — only FP16/BF16 (DISPATCH_HALF_DTYPE).
            # Stock comfy.ops MixedPrecision Linear does not cast before from_float.
            nvfp4_native = (
                bool(getattr(runner, "_dit_comfy_quant_native", False))
                and checkpoint_is_nvfp4(getattr(runner, "_dit_checkpoint", None))
            )
            debug.start_timer(f"dit_inference_{upscale_idx+1}")
            with torch.no_grad():
                use_autocast = (
                    not nvfp4_native
                    and dit_dtype != ctx['compute_dtype']
                    and ctx['dit_device'].type != 'mps'
                )
                if use_autocast:
                    with torch.autocast(ctx['dit_device'].type, ctx['compute_dtype'], enabled=True):
                        upscaled_latents = runner.inference(
                            noises=noises,
                            conditions=conditions,
                            **ctx['text_embeds'],
                        )
                else:
                    upscaled_latents = runner.inference(
                        noises=noises,
                        conditions=conditions,
                        **ctx['text_embeds'],
                    )
            debug.end_timer(f"dit_inference_{upscale_idx+1}", f"DiT inference {upscale_idx+1}")
            
            # Offload upscaled latents to avoid VRAM accumulation
            if ctx['tensor_offload_device'] is not None and (upscaled_latents[0].is_cuda or upscaled_latents[0].is_mps):
                ctx['all_upscaled_latents'][upscale_idx] = manage_tensor(
                    tensor=upscaled_latents[0],
                    target_device=ctx['tensor_offload_device'],
                    tensor_name=f"upscaled_latent_{upscale_idx+1}",
                    debug=debug,
                    reason="storing upscaled latents for decoding",
                    indent_level=1
                )
            else:
                ctx['all_upscaled_latents'][upscale_idx] = upscaled_latents[0]
            
            # Free original latent - release tensor memory first
            release_tensor_memory(ctx['all_latents'][batch_idx])
            ctx['all_latents'][batch_idx] = None
            
            del noises, aug_noises, latent, conditions, condition, base_noise, upscaled_latents
            
            debug.end_timer(f"upscale_batch_{upscale_idx+1}", f"Upscaled batch {upscale_idx+1}")
            
            if progress_callback:
                progress_callback(upscale_idx+1, num_valid_latents,
                                1, "Phase 2: Upscaling")
            
            upscale_idx += 1
            
    except Exception as e:
        debug.log(f"Error in Phase 2 (Upscaling): {e}", level="ERROR", category="error", force=True)
        raise
    finally:
        # Log BlockSwap summary if it was used
        if hasattr(runner, '_blockswap_active') and runner._blockswap_active:
            swap_summary = debug.get_swap_summary()
            if swap_summary and swap_summary.get('total_swaps', 0) > 0:
                total_time = swap_summary.get('block_total_ms', 0) + swap_summary.get('io_total_ms', 0)
                debug.log("BlockSwap Summary", category="blockswap")
                debug.log(f"BlockSwap overhead: {total_time:.2f}ms", category="blockswap", indent_level=1)
                debug.log(f"Total swaps: {swap_summary['total_swaps']}", category="blockswap", indent_level=1)
                
                # Show block swap details
                if 'block_swaps' in swap_summary and swap_summary['block_swaps'] > 0:
                    avg_ms = swap_summary.get('block_avg_ms', 0)
                    total_ms = swap_summary.get('block_total_ms', 0)
                    min_ms = swap_summary.get('block_min_ms', 0)
                    max_ms = swap_summary.get('block_max_ms', 0)
                    
                    debug.log(f"Block swaps: {swap_summary['block_swaps']} "
                            f"(avg: {avg_ms:.2f}ms, min: {min_ms:.2f}ms, max: {max_ms:.2f}ms, total: {total_ms:.2f}ms)", 
                            category="blockswap", indent_level=1)
                    
                    # Show most frequently swapped block
                    if 'most_swapped_block' in swap_summary:
                        debug.log(f"Most swapped: Block {swap_summary['most_swapped_block']} "
                                f"({swap_summary['most_swapped_count']} times)", category="blockswap", indent_level=1)
                
                # Show I/O swap details if present
                if 'io_swaps' in swap_summary and swap_summary['io_swaps'] > 0:
                    debug.log(f"I/O swaps: {swap_summary['io_swaps']} "
                            f"(avg: {swap_summary.get('io_avg_ms', 0):.2f}ms, total: {swap_summary.get('io_total_ms', 0):.2f}ms)", 
                            category="blockswap", indent_level=1)

        # Cleanup DiT as it's no longer needed after upscaling
        cleanup_dit(runner=runner, debug=debug, cache_model=cache_model)
        
        # Cleanup text embeddings as they're no longer needed after upscaling
        cleanup_text_embeddings(ctx, debug)
    
    debug.end_timer("phase2_upscaling", "Phase 2: DiT upscaling complete", show_breakdown=True)
    debug.log_memory_state("After phase 2 (DiT upscaling)", show_tensors=False)
    
    return ctx


def decode_all_batches(
    runner: 'VideoDiffusionInfer',
    ctx: Dict[str, Any],
    debug: 'Debug',
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    cache_model: bool = False
) -> Dict[str, Any]:
    """
    Phase 3: VAE Decoding.
    
    Decodes all upscaled latents back to pixel space and writes directly to
    pre-allocated final_video tensor. This avoids memory duplication by not
    storing intermediate batch_samples.
    
    Requires context from upscale_all_batches with upscaled latents.
    
    Args:
        runner: VideoDiffusionInfer instance with loaded models (required)
        ctx: Context from upscale_all_batches containing upscaled latents (required)
        debug: Debug instance for logging (required)
        progress_callback: Optional callback(current, total, frames, phase_name)
        cache_model: If True, keep VAE model for reuse instead of deleting it
        
    Returns:
        dict: Updated context containing:
            - final_video: Pre-allocated tensor with decoded samples (unnormalized, in [-1,1])
            - decode_batch_info: List of (start_idx, end_idx, ori_length) for Phase 4 processing
            - VAE cleanup completed
            
    Raises:
        ValueError: If context is missing or has no upscaled latents
        RuntimeError: If decoding fails
    """
    if debug is None:
        raise ValueError("Debug instance must be provided to decode_all_batches")
    
    if ctx is None:
        raise ValueError("Context is required for decode_all_batches. Run upscale_all_batches first.")
    
    # Validate we have upscaled latents
    if 'all_upscaled_latents' not in ctx or not ctx['all_upscaled_latents']:
        raise ValueError("No upscaled latents found. Run upscale_all_batches first.")
    
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 3: VAE decoding ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase3_decoding")

    # Count valid latents
    num_valid_latents = len([l for l in ctx['all_upscaled_latents'] if l is not None])
    num_batches = len([l for l in ctx['all_ori_lengths'] if l is not None])
    
    # Get output dimensions from context (set during Phase 1)
    if 'true_target_dims' not in ctx:
        raise ValueError("true_target_dims not found in context. Run encode_all_batches first.")
    true_h, true_w = ctx['true_target_dims']
    total_frames = ctx.get('total_frames', 0)
    C = 4 if ctx.get('is_rgba', False) else 3
    
    # Pre-allocate final_video at the START of decode phase (before any batch processing)
    # This ensures we only need memory for final_video + 1 batch, not final_video + all batch_samples
    # MPS: keep on device (unified memory, no benefit to CPU offload)
    if ctx['tensor_offload_device'] is not None:
        target_device = ctx['tensor_offload_device']
    elif ctx['vae_device'].type == 'mps':
        target_device = ctx['vae_device']
    else:
        target_device = 'cpu'
    channels_str = "RGBA" if C == 4 else "RGB"
    required_gb = (total_frames * true_h * true_w * C * 2) / (1024**3)
    debug.log(f"Pre-allocating output tensor: {total_frames} frames, {true_w}x{true_h}px, {channels_str} ({required_gb:.2f}GB)", 
              category="setup", force=True)
    
    # NumPy互換性を確保するため、compute_dtypeがbfloat16の場合はfloat16を使用
    storage_dtype = torch.float16 if ctx['compute_dtype'] == torch.bfloat16 else ctx['compute_dtype']
    ctx['final_video'] = torch.empty((total_frames, true_h, true_w, C), dtype=storage_dtype, device=target_device)
    
    # Track batch write positions for Phase 4 processing
    # Each entry: (write_start, write_end, batch_idx, ori_length)
    ctx['decode_batch_info'] = []
    
    # Get temporal overlap from context (set during Phase 1)
    temporal_overlap = ctx.get('actual_temporal_overlap', 0)
    
    # Track padding removed for final summary
    total_padding_removed = 0
    
    current_write_idx = 0
    decode_idx = 0
    
    try:
        # VAE should already be materialized from encoding phase
        if runner.vae and next(runner.vae.parameters()).device.type == 'meta':
            materialize_model(runner, "vae", ctx['vae_device'], runner.config, debug)

        # Precision should already be initialized from encoding phase
        ensure_precision_initialized(ctx, runner, debug)

        # Move VAE to GPU for decoding (no-op if already there)
        manage_model_device(model=runner.vae, target_device=ctx['vae_device'], 
                          model_name="VAE", debug=debug, runner=runner)
        
        debug.log_memory_state("After VAE loading for decoding", detailed_tensors=False)

        # Initialize tile_boundaries for decoding debug
        if runner.tile_debug == "decode" and runner.decode_tiled:
            debug.decode_tile_boundaries = []
            debug.log("Tile debug enabled: decode tile boundaries will be visualized", category="vae", force=True)
            debug.log("Remember to disable --tile_debug in production to remove overlay visualization", category="tip", indent_level=1, force=True)
        
        # Process decoding
        for batch_idx, upscaled_latent in enumerate(ctx['all_upscaled_latents']):
            if upscaled_latent is None:
                continue
            
            check_interrupt(ctx)
            
            debug.log(f"Decoding batch {decode_idx+1}/{num_valid_latents}", category="vae", force=True)
            debug.start_timer(f"decode_batch_{decode_idx+1}")
            
            # Move to VAE device with correct dtype for decoding (no-op if already there)
            upscaled_latent = manage_tensor(
                tensor=upscaled_latent,
                target_device=ctx['vae_device'],
                tensor_name=f"upscaled_latent_{decode_idx+1}",
                dtype=ctx['compute_dtype'],
                debug=debug,
                reason="VAE decoding",
                indent_level=1
            )
            
            # Decode latent
            debug.start_timer("vae_decode")
            samples = runner.vae_decode([upscaled_latent])
            debug.end_timer("vae_decode", "VAE decode")
            
            # Process samples - get the single decoded sample
            debug.start_timer("optimized_video_rearrange")
            samples = optimized_video_rearrange(samples)
            debug.end_timer("optimized_video_rearrange", "Video rearrange")
            
            # Get the decoded sample (always single-element list)
            sample = samples[0]
            del samples
            
            # Get original length for this batch (before any padding was added)
            ori_length = ctx['all_ori_lengths'][decode_idx] if decode_idx < len(ctx['all_ori_lengths']) else sample.shape[0]
            
            # Trim temporal padding: sample is in [T, C, H, W] format after rearrange
            if ori_length < sample.shape[0]:
                padding_removed = sample.shape[0] - ori_length
                debug.log(f"Trimming temporal padding: {padding_removed} frames removed ({sample.shape[0]} → {ori_length})", 
                         category="video", indent_level=1)
                sample = sample[:ori_length]
                total_padding_removed += padding_removed
            
            # Trim spatial padding to true target dimensions
            current_h, current_w = sample.shape[-2:]
            if current_h != true_h or current_w != true_w:
                debug.log(f"Trimming spatial padding: {current_w}x{current_h} → {true_w}x{true_h}", 
                         category="video", indent_level=1)
                sample = sample[:, :, :true_h, :true_w]
            
            # Convert to output format: [T, C, H, W] → [T, H, W, C]
            # Note: We keep values in [-1, 1] range - normalization happens in Phase 4
            sample = optimized_sample_to_image_format(sample)  # T, C, H, W → T, H, W, C
            
            # Calculate write position with temporal overlap handling
            batch_frames = sample.shape[0]
            if decode_idx == 0 or temporal_overlap == 0:
                # First batch or no overlap: write all frames
                write_start = current_write_idx
                write_end = current_write_idx + batch_frames
            else:
                # Subsequent batches with overlap: blend overlapping region
                if temporal_overlap < batch_frames and current_write_idx >= temporal_overlap:
                    # Blend overlapping region in-place on final_video
                    prev_tail = ctx['final_video'][current_write_idx - temporal_overlap:current_write_idx]
                    cur_head = sample[:temporal_overlap]
                    
                    # Move to same device for blending if needed
                    if prev_tail.device != cur_head.device:
                        cur_head = cur_head.to(prev_tail.device)
                    
                    blended = blend_overlapping_frames(prev_tail, cur_head, temporal_overlap)
                    ctx['final_video'][current_write_idx - temporal_overlap:current_write_idx] = blended
                    
                    debug.log(f"Blended {temporal_overlap} overlapping frames at positions {current_write_idx - temporal_overlap}-{current_write_idx}", 
                             category="video", indent_level=1)
                    
                    # Write only non-overlapping part
                    sample = sample[temporal_overlap:]
                    batch_frames = sample.shape[0]
                    del prev_tail, cur_head, blended
                
                write_start = current_write_idx
                write_end = current_write_idx + batch_frames
            
            # Move sample to target device and write directly to final_video
            # sampleをtarget_deviceに移動し、final_videoに直接書き込む
            # final_videoのdtype（storage_dtype）を使用して一貫性を確保
            target_dtype = ctx['final_video'].dtype
            sample = manage_tensor(
                tensor=sample,
                target_device=target_device,
                tensor_name=f"sample_{decode_idx+1}",
                dtype=target_dtype,  # final_videoのdtype（float16）に合わせる
                debug=debug,
                reason="writing to final_video",
                indent_level=1
            )
            
            # Write to final_video - for RGBA, write only RGB channels (VAE outputs 3 channels)
            if ctx.get('is_rgba', False):
                ctx['final_video'][write_start:write_end, :, :, :3] = sample
            else:
                ctx['final_video'][write_start:write_end] = sample
            
            # Store batch info for Phase 4 processing
            ctx['decode_batch_info'].append((write_start, write_end, decode_idx, ori_length))
            current_write_idx = write_end
            
            debug.log(f"Wrote {batch_frames} frames to positions {write_start}-{write_end}", 
                     category="video", indent_level=1)
            
            # Free memory immediately - no batch_samples storage
            release_tensor_memory(ctx['all_upscaled_latents'][batch_idx])
            ctx['all_upscaled_latents'][batch_idx] = None
            del upscaled_latent, sample
            
            debug.end_timer(f"decode_batch_{decode_idx+1}", f"Decoded batch {decode_idx+1}")
            
            if progress_callback:
                progress_callback(decode_idx+1, num_valid_latents,
                                1, "Phase 3: Decoding")
            
            decode_idx += 1
        
        # Store padding stats for Phase 4 final summary
        ctx['total_padding_removed'] = total_padding_removed
            
    except Exception as e:
        debug.log(f"Error in Phase 3 (Decoding): {e}", level="ERROR", category="error", force=True)
        raise
    finally:
        # Cleanup VAE as it's no longer needed
        cleanup_vae(runner=runner, debug=debug, cache_model=cache_model)
        
        # Clean up upscaled latents storage
        if 'all_upscaled_latents' in ctx:
            release_tensor_collection(ctx['all_upscaled_latents'])
            del ctx['all_upscaled_latents']
        
    debug.end_timer("phase3_decoding", "Phase 3: VAE decoding complete", show_breakdown=True)
    debug.log_memory_state("After phase 3 (VAE decoding)", show_tensors=False)
    
    return ctx


def postprocess_all_batches(
    ctx: Dict[str, Any],
    debug: 'Debug',
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    color_correction: str = "wavelet",
    prepend_frames: int = 0,
    temporal_overlap: int = 0,
    batch_size: int = 5
) -> Dict[str, Any]:
    """
    Phase 4: Post-processing and Final Assembly.
    
    Processes final_video slices in-place: applies alpha upscaling, color correction,
    and normalization. Reads from and writes back to the same final_video tensor
    to avoid memory duplication.
    
    Args:
        ctx: Context from decode_all_batches containing final_video (required)
        debug: Debug instance for logging (required)
        progress_callback: Optional callback(current, total, frames, phase_name)
        color_correction: Color correction method - "wavelet", "adain", or "none" (default: "wavelet")
        prepend_frames: Number of prepended frames to remove from final output (default: 0)
        temporal_overlap: Number of overlapping frames between batches for blending (default: 0)
        batch_size: Frames per batch used during encoding for overlap calculation (default: 5)
        
    Returns:
        dict: Updated context containing:
            - final_video: Assembled video tensor [T, H, W, C] range [0,1] with overlap blended and prepended frames removed
            - All intermediate storage cleared for memory efficiency
            
    Raises:
        ValueError: If context is missing or has no final_video
    """
    if debug is None:
        raise ValueError("Debug instance must be provided to postprocess_all_batches")
    
    if ctx is None:
        raise ValueError("Context is required for postprocess_all_batches. Run decode_all_batches first.")
    
    # Validate we have final_video (pre-allocated in decode_all_batches)
    if 'final_video' not in ctx or ctx['final_video'] is None:
        raise ValueError("final_video not found. Run decode_all_batches first.")
    
    # Validate we have batch info for processing
    if 'decode_batch_info' not in ctx or not ctx['decode_batch_info']:
        raise ValueError("decode_batch_info not found. Run decode_all_batches first.")
    
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 4: Post-processing ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase4_postprocessing")
    
    # Total_frames represents the original input frame count (set in Phase 1)
    total_frames = ctx.get('total_frames', 0)
    
    # Early exit if no frames to process
    if total_frames == 0:
        ctx['final_video'] = torch.empty((0, 0, 0, 0), dtype=ctx['compute_dtype'])
        debug.log("No frames to process", level="WARNING", category="generation", force=True)
        return ctx
    
    # Get batch info from decode phase
    batch_info_list = ctx['decode_batch_info']
    num_valid_samples = len(batch_info_list)
    
    # Calculate total post-processing work units
    # For RGBA: each batch needs 2 steps (alpha processing + color correction/assembly)
    # For RGB: each batch needs 1 step (color correction/assembly only)
    has_alpha_processing = (ctx.get('is_rgba', False) and 
                           'all_alpha_channels' in ctx and 
                           'all_input_rgb' in ctx and
                           isinstance(ctx.get('all_alpha_channels'), list))
    
    if has_alpha_processing:
        total_postprocessing_steps = num_valid_samples * 2  # Alpha + main processing
    else:
        total_postprocessing_steps = num_valid_samples  # Main processing only
    
    current_postprocessing_step = 0
    
    # Get padding stats from Phase 3
    total_padding_removed = ctx.get('total_padding_removed', 0)
    
    # Alpha processing - handle RGBA inputs with edge-guided upscaling
    # Process alpha on final_video slices in-place
    if has_alpha_processing:
        debug.log("Processing Alpha channel with edge-guided upscaling...", category="alpha")
        
        # Validate alpha channel data exists
        if not isinstance(ctx.get('all_alpha_channels'), list) or not isinstance(ctx.get('all_input_rgb'), list):
            debug.log("WARNING: Alpha channel data malformed, skipping alpha processing", 
                     level="WARNING", category="alpha", force=True)
        else:
            for write_start, write_end, batch_idx, ori_length in batch_info_list:
                # Bounds checking for alpha channel lists
                if batch_idx >= len(ctx['all_alpha_channels']) or ctx['all_alpha_channels'][batch_idx] is None:
                    continue
                    
                # Validate alpha channel tensor integrity
                if not isinstance(ctx['all_alpha_channels'][batch_idx], torch.Tensor):
                    debug.log(f"WARNING: Alpha channel {batch_idx} is not a tensor, skipping", 
                             level="WARNING", category="alpha", force=True)
                    continue
                
                debug.log(f"Processing Alpha batch {batch_idx+1}/{num_valid_samples}", category="alpha", force=True)
                debug.start_timer(f"alpha_batch_{batch_idx+1}")

                # Get RGB slice from final_video for alpha processing
                # final_video is [T, H, W, C], process_alpha_for_batch expects list of [T, C, H, W]
                rgb_slice = ctx['final_video'][write_start:write_end, :, :, :3]  # Only RGB
                rgb_tchw = rgb_slice.permute(0, 3, 1, 2)  # [T, H, W, 3] → [T, 3, H, W]
                
                # Process Alpha and merge with RGB
                processed_samples = process_alpha_for_batch(
                    rgb_samples=[rgb_tchw],
                    alpha_original=ctx['all_alpha_channels'][batch_idx],
                    rgb_original=ctx['all_input_rgb'][batch_idx],
                    device=ctx['vae_device'],
                    compute_dtype=ctx['compute_dtype'],
                    debug=debug
                )
                
                # processed_samples[0] is [T, 4, H, W] (RGBA)
                # Extract only the alpha channel and write to final_video's alpha slot
                processed_rgba = processed_samples[0]  # [T, 4, H, W]
                alpha_channel = processed_rgba[:, 3:4, :, :]  # [T, 1, H, W]
                alpha_thwc = alpha_channel.permute(0, 2, 3, 1)  # [T, 1, H, W] → [T, H, W, 1]
                
                alpha_thwc = manage_tensor(
                    tensor=alpha_thwc,
                    target_device=ctx['final_video'].device,
                    tensor_name=f"alpha_channel_{batch_idx+1}",
                    dtype=ctx['compute_dtype'],
                    debug=debug,
                    reason="writing alpha channel to final_video",
                    indent_level=1
                )
                
                # Write only the alpha channel to the 4th channel slot
                ctx['final_video'][write_start:write_end, :, :, 3:4] = alpha_thwc
                
                del rgb_slice, rgb_tchw, processed_samples, processed_rgba, alpha_channel, alpha_thwc
                
                # Free memory immediately
                release_tensor_memory(ctx['all_alpha_channels'][batch_idx])
                ctx['all_alpha_channels'][batch_idx] = None

                release_tensor_memory(ctx['all_input_rgb'][batch_idx])
                ctx['all_input_rgb'][batch_idx] = None
            
                debug.end_timer(f"alpha_batch_{batch_idx+1}", f"Alpha batch {batch_idx+1}")
                
                # Update progress for alpha processing step
                current_postprocessing_step += 1
                if progress_callback:
                    progress_callback(current_postprocessing_step, total_postprocessing_steps,
                                    1, "Phase 4: Post-processing")

        debug.log("Alpha processing complete for all batches", category="alpha")
    
    try:
        # Process each batch slice in final_video in-place
        for info_idx, (write_start, write_end, batch_idx, ori_length) in enumerate(batch_info_list):
            check_interrupt(ctx)
            
            debug.log(f"Post-processing batch {info_idx+1}/{num_valid_samples}", category="video", force=True)
            debug.start_timer(f"postprocess_batch_{info_idx+1}")
            
            # Get slice from final_video - currently in [T, H, W, C] format, values in [-1, 1]
            sample_thwc = ctx['final_video'][write_start:write_end]
            
            # For RGBA, we only process RGB channels for color correction
            # Alpha was already written during alpha processing above
            if ctx.get('is_rgba', False) and sample_thwc.shape[-1] == 4:
                sample_thwc_rgb = sample_thwc[..., :3]  # [T, H, W, 3]
                sample = sample_thwc_rgb.permute(0, 3, 1, 2)  # [T, H, W, 3] → [T, 3, H, W]
            else:
                sample = sample_thwc.permute(0, 3, 1, 2)  # [T, H, W, C] → [T, C, H, W]
            
            # Move to VAE device for processing
            sample = manage_tensor(
                tensor=sample,
                target_device=ctx['vae_device'],
                tensor_name=f"sample_{info_idx+1}",
                dtype=ctx['compute_dtype'],
                debug=debug,
                reason="post-processing",
                indent_level=1
            )
            
            # Reconstruct transformed video on-demand for color correction
            input_video = None
            if color_correction != "none" and ctx.get('batch_metadata') is not None:
                if batch_idx < len(ctx['batch_metadata']) and ctx['batch_metadata'][batch_idx] is not None:
                    # Reconstruct transformation
                    transformed_video = _reconstruct_and_transform_batch(ctx, batch_idx, debug)
                    input_video = optimized_single_video_rearrange(transformed_video)
                    del transformed_video
                    
                    # For batches after the first with temporal overlap, the overlap frames
                    # were blended in Phase 3 and are not part of this slice. Skip them.
                    actual_overlap = ctx.get('actual_temporal_overlap', 0)
                    if info_idx > 0 and actual_overlap > 0:
                        input_video = input_video[actual_overlap:]
                    
                    # Trim input_video to match sample length (handles padding differences)
                    if input_video.shape[0] > sample.shape[0]:
                        input_video = input_video[:sample.shape[0]]
                    
                    # Trim spatial dimensions to true target size
                    if 'true_target_dims' in ctx:
                        true_h, true_w = ctx['true_target_dims']
                        if input_video.shape[-2] != true_h or input_video.shape[-1] != true_w:
                            input_video = input_video[:, :, :true_h, :true_w]
            
            # Apply color correction if enabled (RGB only)
            if color_correction != "none" and input_video is not None:
                # Check if RGBA (samples are in T, C, H, W format at this point)
                has_alpha = ctx.get('is_rgba', False)
                alpha_channel = None
                
                if has_alpha:
                    # Check actual channel count
                    if sample.shape[1] == 4:
                        # Extract and temporarily store alpha for reattachment after color correction
                        alpha_channel = sample[:, 3:4, :, :]  # (T, 1, H, W)
                        sample = sample[:, :3, :, :]  # Keep only RGB (T, 3, H, W)
                
                # Ensure both tensors are on same device (GPU) for color correction
                if input_video.device != sample.device:
                    input_video = manage_tensor(
                        tensor=input_video,
                        target_device=sample.device,
                        tensor_name=f"input_video_{info_idx+1}",
                        debug=debug,
                        reason="color correction",
                        indent_level=1
                    )
                    
                # Apply selected color correction method
                debug.start_timer(f"color_correction_{color_correction}")
                
                if color_correction == "lab":
                    debug.log("Applying LAB perceptual color transfer", category="video", force=True, indent_level=1)
                    sample = lab_color_transfer(sample, input_video, debug, luminance_weight=0.8)
                elif color_correction == "wavelet_adaptive":
                    debug.log("Applying wavelet with adaptive saturation correction", category="video", force=True, indent_level=1)
                    sample = wavelet_adaptive_color_correction(sample, input_video, debug)
                elif color_correction == "wavelet":
                    debug.log("Applying wavelet color reconstruction", category="video", force=True, indent_level=1)
                    sample = wavelet_reconstruction(sample, input_video, debug)
                elif color_correction == "hsv":
                    debug.log("Applying HSV hue-conditional saturation matching", category="video", force=True, indent_level=1)
                    sample = hsv_saturation_histogram_match(sample, input_video, debug)
                elif color_correction == "adain":
                    debug.log("Applying AdaIN color correction", category="video", force=True, indent_level=1)
                    sample = adaptive_instance_normalization(sample, input_video)
                else:
                    debug.log(f"Unknown color correction method: {color_correction}", level="WARNING", category="video", force=True, indent_level=1)
                
                debug.end_timer(f"color_correction_{color_correction}", f"Color correction ({color_correction})")
                
                # Free the reconstructed transformed video
                del input_video

                # Recombine with Alpha if it was present in input
                if has_alpha and alpha_channel is not None:
                    # Concatenate in channels-first: (T, 3, H, W) + (T, 1, H, W) -> (T, 4, H, W)
                    sample = torch.cat([sample, alpha_channel], dim=1)
            
            else:
                debug.log("Color correction disabled (set to none)", category="video", indent_level=1)
            
            # Convert to final format: [T, C, H, W] → [T, H, W, C]
            sample = optimized_sample_to_image_format(sample)
            
            # Apply normalization only to RGB channels, preserve Alpha as-is
            if ctx.get('is_rgba', False) and sample.shape[-1] == 4:
                # Split RGBA: sample is (T, H, W, C) format after optimized_sample_to_image_format
                rgb_channels = sample[..., :3]  # (T, H, W, 3)
                alpha_channel = sample[..., 3:4]  # (T, H, W, 1)
                
                # Normalize only RGB from [-1, 1] to [0, 1]
                rgb_channels.clamp_(-1, 1).mul_(0.5).add_(0.5)
                
                # Merge back with unchanged Alpha
                sample = torch.cat([rgb_channels, alpha_channel], dim=-1)
            else:
                # RGB only: apply normalization as usual
                sample.clamp_(-1, 1).mul_(0.5).add_(0.5)
            
            # Draw tile boundaries for debugging (if tile info available)
            for phase, attr in [('encode', 'encode_tile_boundaries'), ('decode', 'decode_tile_boundaries')]:
                tiles = getattr(debug, attr, None)
                if tiles:
                    sample = _draw_tile_boundaries(sample, debug, tiles, phase)
                    break
            
            # Move to final_video device and write back in-place
            target_dtype = ctx['final_video'].dtype
            sample = manage_tensor(
                tensor=sample,
                target_device=ctx['final_video'].device,
                tensor_name=f"sample_{info_idx+1}_final",
                dtype=target_dtype,
                debug=debug,
                reason="writing processed result to final_video",
                indent_level=1
            )
            
            # Write back to final_video in-place
            # For RGBA, write only RGB channels (alpha already written during alpha processing)
            if ctx.get('is_rgba', False) and ctx['final_video'].shape[-1] == 4:
                ctx['final_video'][write_start:write_end, :, :, :3] = sample
            else:
                ctx['final_video'][write_start:write_end] = sample
            
            # Free sample memory
            del sample, sample_thwc
            
            debug.end_timer(f"postprocess_batch_{info_idx+1}", f"Post-processed batch {info_idx+1}")
            
            # Update progress for main processing step
            current_postprocessing_step += 1
            if progress_callback:
                progress_callback(current_postprocessing_step, total_postprocessing_steps,
                                1, "Phase 4: Post-processing")

        # Verify final assembly
        if ctx['final_video'] is not None:
            # Remove prepended frames if any were added at the start
            frames_before_removal = ctx['final_video'].shape[0]
            
            if prepend_frames > 0:
                if prepend_frames < ctx['final_video'].shape[0]:
                    debug.log(f"Removing {prepend_frames} prepended frames from output", category="video", force=True)
                    ctx['final_video'] = ctx['final_video'][prepend_frames:]
                else:
                    debug.log(f"Warning: prepend_frames ({prepend_frames}) >= total frames ({ctx['final_video'].shape[0]}), skipping removal", 
                            level="WARNING", category="video", force=True)

            final_shape = ctx['final_video'].shape
            Tf, Hf, Wf, Cf = final_shape[0], final_shape[1], final_shape[2], final_shape[3]
            channels_str = "RGBA" if Cf == 4 else "RGB" if Cf == 3 else f"{Cf}-channel"
            
            # Build message showing prepend and/or padding removal if applicable
            frame_info = f"{Tf} frames"
            adjustments = []

            if prepend_frames > 0 and prepend_frames < frames_before_removal:
                adjustments.append(f"{prepend_frames} prepend")

            if total_padding_removed > 0:
                adjustments.append(f"{total_padding_removed} padding")
            
            # Use actual temporal overlap from encoding (may have been reset)
            actual_overlap = ctx.get('actual_temporal_overlap', temporal_overlap)
            
            # Calculate and include temporal overlap blending info
            if actual_overlap > 0:
                frames_blended = (num_valid_samples - 1) * actual_overlap
                adjustments.append(f"{frames_blended} overlap")

            if adjustments:
                # Add back all removed/blended frames to get true computed count
                total_computed = frames_before_removal + total_padding_removed
                if actual_overlap > 0:
                    total_computed += (num_valid_samples - 1) * actual_overlap
                frame_info += f" ({total_computed} computed with {' + '.join(adjustments)} removed)"
            
            debug.log(f"Output assembled: {frame_info}, Resolution: {Wf}x{Hf}px, Channels: {channels_str}", 
                    category="generation", force=True)
        else:
            ctx['final_video'] = torch.empty((0, 0, 0, 0), dtype=ctx['compute_dtype'])
            debug.log("No frames were processed", level="WARNING", category="generation", force=True)
            
    except Exception as e:
        debug.log(f"Error in Phase 4 (Post-processing): {e}", level="ERROR", category="generation", force=True)
        raise
    finally:
        # 1. Clean up decode_batch_info and padding stats
        if 'decode_batch_info' in ctx:
            del ctx['decode_batch_info']
        if 'total_padding_removed' in ctx:
            del ctx['total_padding_removed']
        
        # 2. Clean up video transform caches
        if 'video_transform' in ctx and ctx['video_transform'] is not None:
            if hasattr(ctx['video_transform'], 'transforms'):
                for transform in ctx['video_transform'].transforms:
                    # Clear cache attributes
                    for cache_attr in ['cache', '_cache']:
                        if hasattr(transform, cache_attr):
                            setattr(transform, cache_attr, None)
                    # Clear remaining attributes
                    if hasattr(transform, '__dict__'):
                        transform.__dict__.clear()
            del ctx['video_transform']
        
        # 3. Clean up storage lists (all_latents, all_alpha_channels, etc.)
        tensor_storage_keys = ['all_latents', 'all_alpha_channels', 'all_input_rgb']
        for key in tensor_storage_keys:
            if key in ctx and ctx[key]:
                release_tensor_collection(ctx[key])
                del ctx[key]
        
        # 4. Clean up non-tensor storage
        if 'all_ori_lengths' in ctx:
            del ctx['all_ori_lengths']
        if 'true_target_dims' in ctx:
            del ctx['true_target_dims']
        if 'batch_metadata' in ctx:
            del ctx['batch_metadata']
        if 'input_images' in ctx:
            release_tensor_memory(ctx['input_images'])
            del ctx['input_images']

    debug.end_timer("phase4_postprocessing", "Phase 4: Post-processing complete", show_breakdown=True)
    debug.log_memory_state("After phase 4 (Post-processing)", show_tensors=False)
    
    return ctx
```

### `src/common/config.py` (complete)

```python
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

"""
Configuration utility functions
"""

import importlib
from typing import Any, Callable, List, Union
from omegaconf import DictConfig, ListConfig, OmegaConf
from ..utils.model_registry import MODEL_CLASSES

try:
    OmegaConf.register_new_resolver("eval", eval)
except Exception as e:
    if "already registered" not in str(e):
        raise



def load_config(path: str, argv: List[str] = None) -> Union[DictConfig, ListConfig]:
    """
    Load a configuration. Will resolve inheritance.
    """
    
    #print(path)
    config = OmegaConf.load(path)
    if argv is not None:
        config_argv = OmegaConf.from_dotlist(argv)
        config = OmegaConf.merge(config, config_argv)
    config = resolve_recursive(config, resolve_inheritance)
    return config


def resolve_recursive(
    config: Any,
    resolver: Callable[[Union[DictConfig, ListConfig]], Union[DictConfig, ListConfig]],
) -> Any:
    config = resolver(config)
    if isinstance(config, DictConfig):
        for k in config.keys():
            v = config.get(k)
            if isinstance(v, (DictConfig, ListConfig)):
                config[k] = resolve_recursive(v, resolver)
    if isinstance(config, ListConfig):
        for i in range(len(config)):
            v = config.get(i)
            if isinstance(v, (DictConfig, ListConfig)):
                config[i] = resolve_recursive(v, resolver)
    return config


def resolve_inheritance(config: Union[DictConfig, ListConfig]) -> Any:
    """
    Recursively resolve inheritance if the config contains:
    __inherit__: path/to/parent.yaml or a ListConfig of such paths.
    """
    if isinstance(config, DictConfig):
        inherit = config.pop("__inherit__", None)

        if inherit:
            inherit_list = inherit if isinstance(inherit, ListConfig) else [inherit]

            parent_config = None
            for parent_path in inherit_list:
                assert isinstance(parent_path, str)
                parent_config = (
                    load_config(parent_path)
                    if parent_config is None
                    else OmegaConf.merge(parent_config, load_config(parent_path))
                )

            if len(config.keys()) > 0:
                config = OmegaConf.merge(parent_config, config)
            else:
                config = parent_config
    return config


def import_item(path: str, name: str) -> Any:
    """
    Import a python item, checking model registry first.
    
    Args:
        path: Module path
        name: Class/function name to import
        
    Returns:
        Imported object
    """
    # Simple lookup with path as key
    if path in MODEL_CLASSES:
        return MODEL_CLASSES[path]
    
    # Fallback to dynamic import for everything else
    try:
        return getattr(importlib.import_module(path), name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import '{name}' from '{path}': {e}")


def create_object(config: DictConfig, **extra_kwargs) -> Any:
    """
    Create an object from config.
    The config is expected to contains the following:
    __object__:
      path: path.to.module
      name: MyClass
      args: as_config | as_params (default to as_config)

    ``extra_kwargs`` are merged at construction time only (e.g. ``operations``
    for ComfyUI ``comfy.ops`` INT8 / NVFP4 injection). Not stored in YAML.
    """
    
    item = import_item(
        path=config.__object__.path,
        name=config.__object__.name,
    )
    args = config.__object__.get("args", "as_config")
    if args == "as_config":
        return item(config, **extra_kwargs)
    if args == "as_params":
        config = OmegaConf.to_object(config)
        config.pop("__object__")
        config.update(extra_kwargs)
        return item(**config)
    raise NotImplementedError(f"Unknown args type: {args}")
```

### `src/utils/model_registry.py` (complete)

```python
"""
Model Registry for SeedVR2
Central registry for model definitions, repositories, and metadata
"""

import os
from typing import List, Optional
from dataclasses import dataclass
from .constants import get_all_model_files

# Model class imports using relative imports
from ..models.dit_3b.nadit import NaDiT as NaDiT3B
from ..models.dit_7b.nadit import NaDiT as NaDiT7B
from ..models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper

# Model classes - simple registry with clear keys
MODEL_CLASSES = {
    "dit_3b.nadit": NaDiT3B,
    "dit_7b.nadit": NaDiT7B,
    "video_vae_v3.modules.attn_video_vae": VideoAutoencoderKLWrapper,
}

@dataclass
class ModelInfo:
    """Model metadata"""
    repo: str = "numz/SeedVR2_comfyUI"
    category: str = "dit" # 'model' or 'vae'
    precision: str = "fp16" # 'fp16', 'fp8_e4m3fn', 'Q4_K_M', etc.
    size: str = "3B" # '3B', '7B', etc.
    variant: Optional[str] = None # 'sharp', etc.
    sha256: Optional[str] = None # Cached hash

# Model registry with metadata
MODEL_REGISTRY = {
    # 3B models
    "seedvr2_ema_3b-Q4_K_M.gguf": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="3B", precision="Q4_K_M", sha256="e665e3909de1a8c88a69c609bca9d43ff5a134647face2ce4497640cc3597f0e"),
    "seedvr2_ema_3b-Q8_0.gguf": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="3B", precision="Q8_0", sha256="be0d60083a2051a265eb4b77f28edf494e6db67ffc250216f32b72292e5cbd96"),
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors": ModelInfo(size="3B", precision="fp8_e4m3fn", sha256="3bf1e43ebedd570e7e7a0b1b60d6a02e105978f505c8128a241cde99a8240cff"),
    "seedvr2_ema_3b_fp16.safetensors": ModelInfo(size="3B", precision="fp16", sha256="2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304"),
    
    # 7B models
    "seedvr2_ema_7b-Q4_K_M.gguf": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="Q4_K_M", sha256="db9cb2ad90ebd40d2e8c29da2b3fc6fd03ba87cd58cbadceccca13ad27162789"),
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="fp8_e4m3fn_mixed_block35_fp16", sha256="3d68b5ec0b295ae28092e355c8cad870edd00b817b26587d0cb8f9dd2df19bb2"),
    "seedvr2_ema_7b_fp16.safetensors": ModelInfo(size="7B", precision="fp16", sha256="7b8241aa957606ab6cfb66edabc96d43234f9819c5392b44d2492d9f0b0bbe4a"),
    # HSWQ INT8 (int8_tensorwise + ConvRot) — native INT8 inference (VRAM-saving path)
    "seedvr2_7b_int8_convrot.safetensors": ModelInfo(size="7B", precision="int8_tensorwise_convrot"),
    
    # 7B sharp variants
    "seedvr2_ema_7b_sharp-Q4_K_M.gguf": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="Q4_K_M", variant="sharp", sha256="7aed800ac4eb8e0d18569a954c0ff35f5a1caa3ed5d920e66cc31405f75b6e69"),
    "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="fp8_e4m3fn_mixed_block35_fp16", variant="sharp", sha256="0d2c5b8be0fda94351149c5115da26aef4f4932a7a2a928c6f184dda9186e0be"),
    "seedvr2_ema_7b_sharp_fp16.safetensors": ModelInfo(size="7B", precision="fp16", variant="sharp", sha256="20a93e01ff24beaeebc5de4e4e5be924359606c356c9c51509fba245bd2d77dd"),
    "seedvr2_7b_sharp_int8_convrot.safetensors": ModelInfo(size="7B", precision="int8_tensorwise_convrot", variant="sharp"),
    
    # VAE models
    "ema_vae_fp16.safetensors": ModelInfo(category="vae", precision="fp16", sha256="20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"),
}

# Configuration constants
DEFAULT_DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
DEFAULT_VAE = "ema_vae_fp16.safetensors"

def get_default_models(category: str) -> List[str]:
    """Get list of default models"""
    return [name for name, info in MODEL_REGISTRY.items() if info.category == category]

def get_model_repo(model_name: str) -> str:
    """Get repository for a specific model"""
    return MODEL_REGISTRY.get(model_name, ModelInfo()).repo

def resolve_dit_config_folder(dit_model: str) -> str:
    """
    Resolve configs_7b vs configs_3b from registry size and/or filename.

    Filename substring \"7b\"/\"3b\" is the historical rule. Registry size is used
    when the model is registered (including HSWQ INT8 names). Prefer explicit
    7b/3b tokens in the basename so untagged temp names do not silently pick 3B.
    """
    info = MODEL_REGISTRY.get(dit_model)
    if info is not None and info.category == "dit":
        size = (info.size or "").upper()
        if size == "7B":
            return "configs_7b"
        if size == "3B":
            return "configs_3b"

    name = dit_model.lower()
    if "7b" in name:
        return "configs_7b"
    if "3b" in name:
        return "configs_3b"
    return "configs_3b"

def get_available_dit_models() -> List[str]:
    """Get all available DiT models including those discovered on disk"""
    model_list = get_default_models("dit")
    
    try:
        # Get all model files from all paths
        model_files = get_all_model_files()
        
        # Add files not in registry
        discovered_models = [
            filename for filename in model_files
            if filename not in MODEL_REGISTRY
        ]
        
        # Add discovered models to the list
        model_list.extend(sorted(discovered_models))
    except:
        pass
    
    return model_list

def get_available_vae_models() -> List[str]:
    """Get all available VAE models from the registry"""
    model_list = get_default_models("vae")
    return model_list
```

---

## 4. Meaning (NVFP4)

| Change | Meaning |
|--------|---------|
| `checkpoint_is_nvfp4` | Ground-truth detect from safetensors markers; no filename guessing |
| `get_nvfp4_mixed_precision_ops` | Construction-time Linear that understands `comfy_quant` / NVFP4 scales |
| `disabled=["nvfp4"]` when no native compute | Keep packed weights on weaker GPUs; avoid hard fail |
| Activation cast in `Linear.forward` | `quantize_nvfp4` only accepts FP16/BF16; float32 from norms would crash |
| Skip DiT autocast | Prevents norms under autocast from emitting float32 into NVFP4 Linear (same root cause as above) |
| Shared prep with INT8 | `comfy_quant` `.numpy()` needs CPU; meta Linear needs real `factory_kwargs["device"]` |
| `_dit_comfy_quant_native` | Runtime flag so upscale phase can special-case NVFP4 without re-scanning every batch |

Without construction-time injection, NVFP4 would fully expand → large VRAM, defeating the pack.

---

## 5. torch.compile errors that appeared afterward

Observed while running workflows that used **NVFP4 DiT** with **FP16 VAE** and VAE `torch.compile` enabled. **VAE itself is not NVFP4-quantized**; the failure is inductor / Windows encoding / decomp registration.

### Error A — Windows locale decode (first failure)

Log shape:

```text
Configuring torch.compile for VAE submodules...
[WARNING] torch.compile failed for VAE submodules: 'cp932' codec can't decode byte 0x94 in position 618: illegal multibyte sequence
[WARNING]   Falling back to uncompiled VAE
Encoding batch ...
```

Facts:

- Japanese Windows OEM / preferred encoding = **cp932**.
- Inductor opens UTF-8 `.py.jinja` templates via `open()` **without** `encoding="utf-8"` → locale cp932 decode fails (byte `0x94` at ~618).
- Separately, `torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ('oem',)` is **strict**; MSVC / OEM console bytes can raise the same `UnicodeDecodeError`.
- SeedVR2 caught the exception and fell back to **uncompiled VAE** (generation continued without compile speedup).

### Error B — inductor decomp / fallback assert (next failure after encoding patch)

```text
AssertionError: both a fallback and a decomp for same op: aten.bmm.default
```

(Same pattern also reported for partial decomps such as `aten.addmm.default`, `aten.mm`, `aten.mv`, `aten.linear`.)

Facts:

- Inductor registers **partial** decompositions that return `NotImplemented` for general shapes, then calls `make_fallback`.
- Default `override_decomp=False` asserts if the op is already in the decomp table → compile abort for VAE submodule compile.

---

## 6. Countermeasure overview

Do **not** disable torch.compile or remap `max-autotune` for these errors.

1. Patch inductor **jinja** loader to always `encoding="utf-8"`, and rebind already-created `functools.partial` hooks in `mm_common` / flex templates.
2. Set `SUBPROCESS_DECODE_ARGS` to `(encoding, "replace")` / `("oem", "replace")` on inductor cpp_builder and `torch.utils.cpp_extension`.
3. Harden `CppCompileError` byte→str with `errors="replace"`; optionally rewrite legacy `_run_compile_cmd` decode sites.
4. Wrap `make_fallback` so if `op` is already in the active decomp table, set `override_decomp=True`; also rebind `torch._inductor.graph.make_fallback` if already imported.
5. Call `_fix_inductor_windows_encoding()` from **`__init__.py` at custom-node import** (before any compile), and keep the existing call from `model_configuration.py`.

---

## 7. Added / modified files (torch.compile)

| Path | Role |
|------|------|
| `src/core/fix_inductor.py` | UTF-8 jinja patch, OEM/cp932 replace, bmm/`make_fallback` override |
| `__init__.py` | Apply inductor fix at extension import time |

---

## 8. Full source of added / modified files (torch.compile)

### `src/core/fix_inductor.py` (complete)

```python
import os
import locale
import inspect
import textwrap
from functools import partial
from pathlib import Path


def _safe_windows_decode_args() -> tuple:
    """Encoding + errors=replace for MSVC/OEM bytes on Japanese Windows (cp932)."""
    try:
        enc = locale.getpreferredencoding(False) or "utf-8"
    except Exception:
        enc = "utf-8"
    return (enc, "replace")


def _load_template_utf8(name: str, template_dir: Path) -> str:
    """UTF-8 open for inductor jinja templates (locale cp932 breaks on UTF-8 bytes)."""
    with open(template_dir / f"{name}.py.jinja", encoding="utf-8") as f:
        return f.read()


def _patch_inductor_load_template_utf8() -> None:
    """
    Root cause of:
      UnicodeDecodeError: 'cp932' codec can't decode byte 0x94 in position 618

    torch._inductor.utils.load_template uses open() without encoding= → locale
    cp932 on Japanese Windows. Jinja templates under torch/_inductor are UTF-8
    (e.g. cutedsl_mm_grouped.py.jinja). Failure happens at inductor import time
    when mm_grouped.py calls load_kernel_template(...).

    Must patch utils.load_template BEFORE torch._inductor.kernel.mm_common
    creates functools.partial(load_template, ...). If mm_common is already
    imported, rebind its partials to the UTF-8 loader.
    """
    try:
        import torch._inductor.utils as inductor_utils

        inductor_utils.load_template = _load_template_utf8  # type: ignore[assignment]
    except Exception as e:
        print(f"[SeedVR2] Warning: Could not patch inductor load_template: {e}")
        return

    # Rebind partials if kernel helpers already imported with the old function
    try:
        import sys

        mm_common = sys.modules.get("torch._inductor.kernel.mm_common")
        if mm_common is not None:
            if hasattr(mm_common, "_KERNEL_TEMPLATE_DIR"):
                mm_common.load_kernel_template = partial(
                    _load_template_utf8,
                    template_dir=mm_common._KERNEL_TEMPLATE_DIR,
                )
            if hasattr(mm_common, "_KERNEL_TEMPLATE_FB_DIR"):
                mm_common.load_fb_kernel_template = partial(
                    _load_template_utf8,
                    template_dir=mm_common._KERNEL_TEMPLATE_FB_DIR,
                )
    except Exception as e:
        print(f"[SeedVR2] Warning: Could not rebind mm_common load_kernel_template: {e}")

    try:
        import sys

        flex_common = sys.modules.get("torch._inductor.kernel.flex.common")
        if flex_common is not None and hasattr(flex_common, "_FLEX_TEMPLATE_DIR"):
            flex_common.load_flex_template = partial(
                _load_template_utf8,
                template_dir=flex_common._FLEX_TEMPLATE_DIR,
            )
    except Exception as e:
        print(f"[SeedVR2] Warning: Could not rebind flex load_flex_template: {e}")


def _patch_inductor_bmm_make_fallback_override() -> None:
    """
    Fix a recurring inductor assertion during VAE torch.compile:

      AssertionError: both a fallback and a decomp for same op: aten.<op>.default

    Known ops that trigger this with the installed torch:
      - aten.bmm.default   (registered decomp only handles outer-product [B,M,1]x[B,1,N])
      - aten.addmm.default  (registered decomp only handles specific shape cases)
      - aten.mm / aten.mv / aten.linear (same pattern: partial decomposition)

    torch._inductor.decomposition registers decompositions that handle only
    specific cases and return NotImplemented otherwise. In the general case
    inductor falls through to make_fallback(), but make_fallback() asserts
    `op not in check_decomps unless override_decomp=True`. With override_decomp
    left at its default (False) this assertion fires.

    Wrap make_fallback so that whenever the op appears in the active decomp
    table (either the explicitly passed get_decomp_fn or the global
    torch._inductor.lowering.decompositions) we transparently set
    override_decomp=True. The registered decomp is still tried first by
    make_fallback and returns NotImplemented for unsupported shapes, so this
    changes nothing semantically — it just silences the assertion that was
    guarding against accidental double-registration.

    IMPORTANT: torch._inductor.graph does `from torch._inductor.lowering
    import make_fallback` at module top, which binds the symbol into graph.py
    itself. Rebinding `lowering.make_fallback` is NOT enough if graph.py is
    already imported — graph.py keeps using the old reference. We must also
    rebind `graph.make_fallback` when present in sys.modules.
    """
    try:
        import torch._inductor.lowering as inductor_lowering
    except Exception as e:
        print(f"[SeedVR2] Warning: could not import inductor.lowering for make_fallback patch: {e}")
        return

    if getattr(inductor_lowering, "_seedvr2_bmm_override_patched", False):
        return

    _orig_make_fallback = inductor_lowering.make_fallback

    def _patched_make_fallback(op, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            # Compute the active decomposition table the same way make_fallback does:
            #   check_decomps = get_decomp_fn() if get_decomp_fn is not None else decompositions
            get_decomp_fn = kwargs.get("get_decomp_fn")
            if get_decomp_fn is not None:
                check_decomps = get_decomp_fn()
            else:
                check_decomps = inductor_lowering.decompositions

            # If the op is registered there (even as a partial decomp returning
            # NotImplemented), allow the fallback path by setting override_decomp.
            if op in check_decomps:
                kwargs["override_decomp"] = True
        except Exception:
            pass
        return _orig_make_fallback(op, *args, **kwargs)

    inductor_lowering.make_fallback = _patched_make_fallback  # type: ignore[assignment]
    inductor_lowering._seedvr2_bmm_override_patched = True  # type: ignore[attr-defined]

    # Rebind in graph.py too — its `from .lowering import make_fallback` already
    # captured the original reference if it was imported before us.
    try:
        import sys

        graph_mod = sys.modules.get("torch._inductor.graph")
        if graph_mod is not None and getattr(graph_mod, "make_fallback", None) is _orig_make_fallback:
            graph_mod.make_fallback = _patched_make_fallback  # type: ignore[assignment]
    except Exception as e:
        print(f"[SeedVR2] Warning: could not rebind graph.make_fallback: {e}")


def _fix_inductor_windows_encoding() -> None:
    """
    Harden torch inductor / cpp_extension for Japanese Windows (cp932).

    Two independent failure modes:

    A) open() of UTF-8 jinja templates with locale encoding (position ~618)
       → patch load_template to encoding="utf-8"

    B) MSVC / OEM subprocess stdout decoded strictly as oem/cp932
       → SUBPROCESS_DECODE_ARGS with errors="replace"

    SeedVR2 previously only patched (B). VAE torch.compile still failed on (A).

    Also fixes a separate VAE torch.compile failure:
      AssertionError: both a fallback and a decomp for same op: aten.bmm.default
    """
    if os.name != "nt":
        # bmm override is needed regardless of OS — apply on Linux too.
        _patch_inductor_bmm_make_fallback_override()
        return

    # Prefer English MSVC diagnostics when VS respects VSLANG
    os.environ.setdefault("VSLANG", "1033")

    # --- (0) jinja template UTF-8 open (actual VAE compile crash site) ---
    _patch_inductor_load_template_utf8()

    # --- (0b) bmm make_fallback override (VAE compile assertion) ---
    _patch_inductor_bmm_make_fallback_override()

    # --- (1) inductor cpp_builder ---
    try:
        import torch._inductor.cpp_builder as cpp_builder

        cpp_builder.SUBPROCESS_DECODE_ARGS = _safe_windows_decode_args()
    except Exception as e:
        print(f"[SeedVR2] Warning: Could not patch inductor SUBPROCESS_DECODE_ARGS: {e}")

    # --- (2) torch.utils.cpp_extension (common cp932 crash site) ---
    try:
        import torch.utils.cpp_extension as cpp_extension

        # Keep OEM code page (MSVC console) but never strict-fail on bad bytes
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("oem", "replace")
    except Exception as e:
        print(f"[SeedVR2] Warning: Could not patch cpp_extension SUBPROCESS_DECODE_ARGS: {e}")

    # --- (3) CppCompileError constructor: utf-8 without replace ---
    try:
        import torch._inductor.exc as inductor_exc

        _orig_init = inductor_exc.CppCompileError.__init__

        def _patched_cpp_compile_error_init(self, cmd, output):  # type: ignore[no-untyped-def]
            if isinstance(output, (bytes, bytearray)):
                output = bytes(output).decode("utf-8", errors="replace")
            _orig_init(self, cmd, output)

        inductor_exc.CppCompileError.__init__ = _patched_cpp_compile_error_init  # type: ignore[method-assign]
    except Exception as e:
        print(f"[SeedVR2] Warning: Could not patch CppCompileError: {e}")

    # --- (4) Legacy source rewrite for older inductor _run_compile_cmd ---
    try:
        import torch._inductor.cpp_builder as cpp_builder

        target_func = getattr(cpp_builder, "_run_compile_cmd", None)
        if target_func is None:
            return

        try:
            source = inspect.getsource(target_func)
        except OSError:
            return

        source = textwrap.dedent(source)
        replacements = (
            ('e.stdout.decode("utf-8")', 'e.stdout.decode("utf-8", errors="replace")'),
            ("e.stdout.decode('utf-8')", "e.stdout.decode('utf-8', errors='replace')"),
            ('e.output.decode("utf-8")', 'e.output.decode("utf-8", errors="replace")'),
        )
        new_source = source
        changed = False
        for old_code, new_code in replacements:
            if old_code in new_source and new_code not in new_source:
                new_source = new_source.replace(old_code, new_code)
                changed = True

        if not changed:
            return

        local_scope: dict = {}
        exec(new_source, cpp_builder.__dict__, local_scope)
        if "_run_compile_cmd" in local_scope:
            cpp_builder._run_compile_cmd = local_scope["_run_compile_cmd"]

    except Exception as e:
        print(f"[SeedVR2] Warning: Could not patch torch.inductor _run_compile_cmd: {e}")
```

### `__init__.py` (complete)

```python
"""
ComfyUI-SeedVR2_VideoUpscaler
Official SeedVR2 integration for ComfyUI
"""

import sys
import subprocess

# Check critical dependencies early to provide better error messages
# and auto-install if possible, especially useful for Vast.ai / RunPod
def ensure_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name.split(">")[0].split("=")[0].split("<")[0]
    
    try:
        __import__(import_name)
        return  # Already available
    except (ImportError, ModuleNotFoundError):
        pass
    if True:  # Package is missing - install it
        print("\n" + "="*80)
        print(f"SeedVR2: '{import_name}' module not found.")
        print(f"SeedVR2: Current Python executable: {sys.executable}")
        print(f"SeedVR2: Attempting automatic installation of {package_name}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
            print(f"SeedVR2: Successfully installed {package_name}")
        except Exception as e:
            print(f"SeedVR2: Auto-installation failed: {e}")
            print("This often happens on Vast.ai / RunPod when pip installs to a different Python environment.")
            print(f"Please run the following command manually in your terminal:")
            print(f"    {sys.executable} -m pip install \"{package_name}\"")
        print("="*80 + "\n")

# All critical dependencies from requirements.txt
# (torch/torchvision/numpy are assumed present via ComfyUI)
_REQUIRED_PACKAGES = [
    ("safetensors", None),
    ("tqdm", None),
    ("psutil", None),
    ("einops", None),
    ("omegaconf>=2.3.0", "omegaconf"),
    ("diffusers>=0.33.1", "diffusers"),
    ("transformers", None),
    ("accelerate", None),
    ("peft>=0.17.0", "peft"),
    ("rotary_embedding_torch>=0.5.3", "rotary_embedding_torch"),
    ("opencv-python", "cv2"),
    ("gguf", None),
    ("matplotlib", None),
]

for pkg, imp in _REQUIRED_PACKAGES:
    ensure_package(pkg, imp)

# Windows cp932: patch inductor jinja open(encoding=utf-8) before any torch.compile
try:
    from .src.core.fix_inductor import _fix_inductor_windows_encoding

    _fix_inductor_windows_encoding()
except Exception as _seedvr2_inductor_fix_err:  # noqa: BLE001
    print(f"[SeedVR2] Warning: inductor Windows encoding fix skipped: {_seedvr2_inductor_fix_err}")

from .src.optimization.compatibility import ensure_triton_compat  # noqa: F401
from .src.interfaces import comfy_entrypoint, SeedVR2Extension

__all__ = ["comfy_entrypoint", "SeedVR2Extension"]
```

---

## 9. Meaning (torch.compile fixes)

| Change | Meaning |
|--------|---------|
| `load_template` → UTF-8 open | Stops Error A at jinja import (true crash site for many VAE compiles) |
| Rebind `mm_common` / flex partials | Fixes case where inductor already bound old `load_template` into partials |
| `SUBPROCESS_DECODE_ARGS` + `replace` | Stops Error A on MSVC/OEM subprocess stdout |
| `CppCompileError` / `_run_compile_cmd` hardening | Same class of decode failures in error paths |
| `make_fallback` + `override_decomp=True` when op in decomp table | Stops Error B without removing decomps; partial decomp still tried first |
| Rebind `graph.make_fallback` | `from lowering import make_fallback` would otherwise keep the unpatched function |
| Early call from `__init__.py` | Patch must land **before** first `torch.compile` / inductor import side effects |

Resulting behavior when successful: VAE submodule `torch.compile` proceeds instead of `Falling back to uncompiled VAE`.

---

## Audit anchors

| Item | Value |
|------|-------|
| Commit | `a14db91b31c08bee62055e17521d4f1537bef03c` |
| New module | `src/optimization/nvfp4_native_ops.py` |
| Compile fix module | `src/core/fix_inductor.py` |
| INT8 sibling guide | `md/SEEDVR2_INT8_NATIVE_OPS_GUIDE.md` |

---

(End of guide.)
