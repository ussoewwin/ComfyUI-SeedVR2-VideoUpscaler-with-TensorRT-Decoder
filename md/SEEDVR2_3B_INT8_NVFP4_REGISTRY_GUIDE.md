# SeedVR2 — 3B INT8 ConvRot / NVFP4 Registry Guide

<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../zhmd/SEEDVR2_3B_INT8_NVFP4_REGISTRY_GUIDE.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

Target custom node: `ComfyUI/custom_nodes/seedvr2_videoupscaler`  
Release tag: `v1.4`  
Date: 2026-07-31

This guide documents **v1.4** as shipped:

1. Registering **3B** HSWQ INT8 ConvRot and NVFP4 packs in `MODEL_REGISTRY` the same way as 7B
2. Leaving DiT load / config-folder selection / NVFP4 autocast behavior on the **existing** (v1.3) architecture

Related prior guides:

- `md/SEEDVR2_NVFP4_AND_TORCH_COMPILE_GUIDE.md` — NVFP4 native ops and torch.compile
- `md/SEEDVR2_INT8_NATIVE_OPS_GUIDE.md` — INT8 construction-time `comfy.ops`

Model packs (local examples):

- `models/SEEDVR2/seedvr2_3b_nvfp4.safetensors`
- `models/SEEDVR2/seedvr2_3b_int8_convrot.safetensors`

Reference quality numbers for the good NVFP4 path (unchanged architecture):

| Path | Time | Peak VRAM | MSE | SSIM |
|------|------|-----------|-----|------|
| FP16 | ~195.67 s | ~17.02 GiB | — | — |
| NVFP4 | ~18.07 s | ~7.64 GiB | **6.534034** | **0.975915** |

---

## 1. Adding 3B ConvRot INT8 / NVFP4

### What already worked before v1.4

Native VRAM for HSWQ INT8 / NVFP4 does **not** depend on the filename alone. Detection is **content-based**:

- Scan checkpoint for `*.comfy_quant` metadata
- At DiT **construction** (meta), inject `comfy.ops.mixed_precision_ops` via `create_object(..., operations=...)`
- On load, keep `QuantizedTensor` storage (packed INT8 / NVFP4) instead of expanding to FP16/BF16

That path lives in:

- `src/optimization/nvfp4_native_ops.py`
- `src/optimization/int8_native_ops.py`
- `src/core/model_loader.py` (`_dit_comfy_quant_ops`, `prepare_model_structure`)
- DiT `dit_3b` / `dit_7b` already thread `operations=` the same way

Smoke before the registry change already showed both 3B packs loading as `QuantizedTensor` (~1957 MB NVFP4 / ~3342 MB INT8 CUDA DiT weights, ~210 quantized Linears) **when** the native ops path ran.

### What was missing

`MODEL_REGISTRY` already listed 7B entries:

- `seedvr2_7b_int8_convrot.safetensors`
- `seedvr2_7b_nvfp4.safetensors`
- sharp variants

It did **not** list the 3B equivalents. Without registry entries:

- UI / download / default listing may omit the packs
- Tooling and docs lack first-class parity with 7B

### What v1.4 adds for 3B

Register the two 3B packs with the same precision tags as 7B:

| Filename | `ModelInfo` |
|---|---|
| `seedvr2_3b_int8_convrot.safetensors` | `size="3B"`, `precision="int8_tensorwise_convrot"` |
| `seedvr2_3b_nvfp4.safetensors` | `size="3B"`, `precision="nvfp4"` |

Filenames already contain `3b`, so historical config-folder selection in `_create_new_runner` already picks `configs_3b` (full function text in §3.2).

**Important:** Registry registration alone does not implement quantization. Native VRAM still comes from content detection + construction-time ops. Registry makes 3B packs first-class for listing and documentation parity with 7B.

---

## 2. Added / modified filenames

v1.4 code change is **one** source file:

| File | Role |
|---|---|
| `src/utils/model_registry.py` | Add 3B INT8 / NVFP4 `MODEL_REGISTRY` entries |

Docs for this release:

| File | Role |
|---|---|
| `md/SEEDVR2_3B_INT8_NVFP4_REGISTRY_GUIDE.md` | This English guide |
| `zhmd/SEEDVR2_3B_INT8_NVFP4_REGISTRY_GUIDE.md` | Chinese guide |
| `zhmd/v1.4.md` | Chinese release notes |
| `release/_gh_v1.4_en_with_switcher.md` | English GitHub Release body |
| `md/changelog.md` / `zhmd/changelogzh.md` | v1.4 Summary only |

Unchanged DiT paths (same as v1.3):

| File | Role |
|---|---|
| `src/core/model_loader.py` | Content-based construction-time ops; materialize clears `_dit_checkpoint` |
| `src/core/generation_phases.py` | NVFP4 autocast skip via `_dit_comfy_quant_native` + live `checkpoint_is_nvfp4(_dit_checkpoint)` |
| `src/core/model_configuration.py` | Historical `"7b" in dit_model` config-folder rule |

---

## 3. Full text of added / modified code

### 3.1 `src/utils/model_registry.py` — full `MODEL_REGISTRY`

```python
MODEL_REGISTRY = {
    # 3B models
    "seedvr2_ema_3b-Q4_K_M.gguf": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="3B", precision="Q4_K_M", sha256="e665e3909de1a8c88a69c609bca9d43ff5a134647face2ce4497640cc3597f0e"),
    "seedvr2_ema_3b-Q8_0.gguf": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="3B", precision="Q8_0", sha256="be0d60083a2051a265eb4b77f28edf494e6db67ffc250216f32b72292e5cbd96"),
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors": ModelInfo(size="3B", precision="fp8_e4m3fn", sha256="3bf1e43ebedd570e7e7a0b1b60d6a02e105978f505c8128a241cde99a8240cff"),
    "seedvr2_ema_3b_fp16.safetensors": ModelInfo(size="3B", precision="fp16", sha256="2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304"),
    # HSWQ INT8 / NVFP4 (native VRAM path; same as 7B)
    "seedvr2_3b_int8_convrot.safetensors": ModelInfo(size="3B", precision="int8_tensorwise_convrot"),
    "seedvr2_3b_nvfp4.safetensors": ModelInfo(size="3B", precision="nvfp4"),
    
    # 7B models
    "seedvr2_ema_7b-Q4_K_M.gguf": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="Q4_K_M", sha256="db9cb2ad90ebd40d2e8c29da2b3fc6fd03ba87cd58cbadceccca13ad27162789"),
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="fp8_e4m3fn_mixed_block35_fp16", sha256="3d68b5ec0b295ae28092e355c8cad870edd00b817b26587d0cb8f9dd2df19bb2"),
    "seedvr2_ema_7b_fp16.safetensors": ModelInfo(size="7B", precision="fp16", sha256="7b8241aa957606ab6cfb66edabc96d43234f9819c5392b44d2492d9f0b0bbe4a"),
    # HSWQ INT8 (int8_tensorwise + ConvRot) — native INT8 inference target (VRAM-saving path)
    "seedvr2_7b_int8_convrot.safetensors": ModelInfo(size="7B", precision="int8_tensorwise_convrot"),
    "seedvr2_7b_nvfp4.safetensors": ModelInfo(size="7B", precision="nvfp4"),
    
    # 7B sharp variants
    "seedvr2_ema_7b_sharp-Q4_K_M.gguf": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="Q4_K_M", variant="sharp", sha256="7aed800ac4eb8e0d18569a954c0ff35f5a1caa3ed5d920e66cc31405f75b6e69"),
    "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="fp8_e4m3fn_mixed_block35_fp16", variant="sharp", sha256="0d2c5b8be0fda94351149c5115da26aef4f4932a7a2a928c6f184dda9186e0be"),
    "seedvr2_ema_7b_sharp_fp16.safetensors": ModelInfo(size="7B", precision="fp16", variant="sharp", sha256="20a93e01ff24beaeebc5de4e4e5be924359606c356c9c51509fba245bd2d77dd"),
    "seedvr2_7b_sharp_int8_convrot.safetensors": ModelInfo(size="7B", precision="int8_tensorwise_convrot", variant="sharp"),
    "seedvr2_7b_sharp_nvfp4.safetensors": ModelInfo(size="7B", precision="nvfp4", variant="sharp"),
    
    # VAE models
    "ema_vae_fp16.safetensors": ModelInfo(category="vae", precision="fp16", sha256="20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"),
}
```

The **only new lines** for v1.4 are:

```python
    # HSWQ INT8 / NVFP4 (native VRAM path; same as 7B)
    "seedvr2_3b_int8_convrot.safetensors": ModelInfo(size="3B", precision="int8_tensorwise_convrot"),
    "seedvr2_3b_nvfp4.safetensors": ModelInfo(size="3B", precision="nvfp4"),
```

### 3.2 Unchanged — `_create_new_runner` config selection (`model_configuration.py`)

```python
def _create_new_runner(
    dit_model: str,
    vae_model: str,
    base_cache_dir: str,
    debug: Optional['Debug'] = None
) -> VideoDiffusionInfer:
    """
    Create a new VideoDiffusionInfer runner instance from scratch.
    
    Loads appropriate configuration file based on model size (3B or 7B), creates
    runner instance, and initializes with default settings. Called when no cached
    runner template is available or when model selection changes.
    
    Args:
        dit_model: DiT model filename (determines config selection)
                  - Contains "7b" → loads configs_7b/main.yaml
                  - Otherwise → loads configs_3b/main.yaml
        vae_model: VAE model filename (stored for reference, not used in config selection)
        base_cache_dir: Base directory for model files (not used directly but passed for context)
        debug: Debug instance for logging and timing
        
    Returns:
        VideoDiffusionInfer: Newly created runner with:
            - Loaded OmegaConf configuration
            - Initialized diffusion sampler and schedule
            - Config set to mutable (readonly=False)
            - No models loaded (structure only)
    """
    debug.log(f"Creating new runner: DiT={dit_model}, VAE={vae_model}", 
             category="runner", force=True)
    
    debug.start_timer("config_load")
    config_path = os.path.join(script_directory, 
                              './configs_7b' if "7b" in dit_model else './configs_3b', 
                              'main.yaml')
    config = load_config(config_path)
    debug.end_timer("config_load", "Config loading")
    
    debug.start_timer("runner_video_infer")
    runner = VideoDiffusionInfer(config, debug)
    OmegaConf.set_readonly(runner.config, False)
    debug.end_timer("runner_video_infer", "Video diffusion inference runner initialization")
    
    return runner
```

Because `seedvr2_3b_*.safetensors` does not contain `"7b"`, this rule selects `configs_3b` without any further DiT rewrite.

---

## 4. Meaning of the 3B registry change

### Registry entries

- Make `seedvr2_3b_int8_convrot.safetensors` and `seedvr2_3b_nvfp4.safetensors` official DiT entries with `size="3B"`.
- Align precision strings with 7B (`int8_tensorwise_convrot`, `nvfp4`) so tooling and docs stay consistent.
- Give UI / download / listing the same first-class names as 7B HSWQ packs.

### What this change is **not**

- It is not a second quantization implementation.
- It does not replace content-based `*.comfy_quant` detection.
- It does not rewrite `_create_new_runner`, `prepare_model_structure`, or the NVFP4 autocast-skip condition.
- Config-folder selection for these filenames already works via the historical `"7b" in dit_model` rule.

---

## 5. Existing native load path (full text — unchanged in v1.4)

Operators need the full surrounding code to understand why registering 3B packs is enough: quantized VRAM was already content-detected.

### 5.1 `_dit_comfy_quant_ops` / `_dit_needs_comfy_quant_prep` — `model_loader.py`

```python
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
```

### 5.2 Full `prepare_model_structure` — `model_loader.py`

```python
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
```

### 5.3 Full `materialize_model` — `model_loader.py`

```python
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
```

### 5.4 NVFP4 autocast skip at upscale — `generation_phases.py`

Contiguous block inside `upscale_all_batches` (dtype detect through `dit_inference` timer end):

```python
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
```

### 5.5 Full `nvfp4_native_ops.py`

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

### 5.6 Meaning of the existing path

| Piece | Role |
|---|---|
| `checkpoint_is_nvfp4` / `checkpoint_is_hswq_int8` | Content scan of `*.comfy_quant` |
| `_dit_comfy_quant_ops` | Choose NVFP4 or INT8 construction-time ops |
| `prepare_model_structure` | Inject `operations=` before meta create; set `_dit_comfy_quant_native` |
| `materialize_model` | Load weights; clear `_dit_checkpoint` after load |
| Upscale autocast skip | `_dit_comfy_quant_native` and `checkpoint_is_nvfp4(_dit_checkpoint)` |

v1.4 does **not** change this path. Adding registry rows makes the same path apply to listed 3B pack names in UI/tooling.

---

## Audit anchors

| Item | Value |
|---|---|
| Release | `v1.4` |
| Code change | `src/utils/model_registry.py` (two 3B HSWQ entries) |
| 3B INT8 pack | `seedvr2_3b_int8_convrot.safetensors` |
| 3B NVFP4 pack | `seedvr2_3b_nvfp4.safetensors` |
| Config rule | `"7b" in dit_model` → `configs_7b`, else `configs_3b` |
| Native VRAM | Content `*.comfy_quant` + construction-time `comfy.ops.mixed_precision_ops` |
| Reference NVFP4 quality | MSE **6.534034**, SSIM **0.975915** |

---

## Mapping to the requested outline

| Requested section | This guide |
|---|---|
| ① 3B convrot INT8 / NVFP4 addition | §1 |
| ② Added / modified filenames | §2 |
| ③ Full text of added / modified code | §3 |
| ④ Meaning | §4 |
| Existing native path (full code, unchanged) | §5 |
