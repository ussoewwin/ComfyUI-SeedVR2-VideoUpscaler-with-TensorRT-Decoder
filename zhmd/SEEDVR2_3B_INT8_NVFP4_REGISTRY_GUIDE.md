# SeedVR2 — 3B INT8 ConvRot / NVFP4 Registry 指南

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../md/SEEDVR2_3B_INT8_NVFP4_REGISTRY_GUIDE.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

目标自定义节点：`ComfyUI/custom_nodes/seedvr2_videoupscaler`  
发布标签：`v1.4`  
日期：2026-07-31

本指南记录 **v1.4** 的实际交付内容：

1. 在 `MODEL_REGISTRY` 中按与 7B 相同的方式登记 **3B** HSWQ INT8 ConvRot 与 NVFP4 权重包
2. DiT 加载 / 配置目录选择 / NVFP4 autocast 行为保持 **既有**（v1.3）架构不变

相关既有指南：

- `zhmd/SEEDVR2_NVFP4_AND_TORCH_COMPILE_GUIDE.md` — NVFP4 原生 ops 与 torch.compile
- `zhmd/SEEDVR2_INT8_NATIVE_OPS_GUIDE.md` — INT8 构建时 `comfy.ops`

模型权重包（本地示例）：

- `models/SEEDVR2/seedvr2_3b_nvfp4.safetensors`
- `models/SEEDVR2/seedvr2_3b_int8_convrot.safetensors`

良好 NVFP4 路径的参考质量（架构未改）：

| 路径 | 时间 | 峰值显存 | MSE | SSIM |
|------|------|----------|-----|------|
| FP16 | ~195.67 s | ~17.02 GiB | — | — |
| NVFP4 | ~18.07 s | ~7.64 GiB | **6.534034** | **0.975915** |

---

## 1. 增加 3B ConvRot INT8 / NVFP4

### v1.4 之前已经可用的能力

HSWQ INT8 / NVFP4 的原生显存 **不** 依赖文件名本身。检测是 **基于内容** 的：

- 扫描 checkpoint 中的 `*.comfy_quant` 元数据
- 在 DiT **构建**（meta）时，通过 `create_object(..., operations=...)` 注入 `comfy.ops.mixed_precision_ops`
- 加载时保持 `QuantizedTensor` 存储（打包 INT8 / NVFP4），而不是展开为 FP16/BF16

该路径位于：

- `src/optimization/nvfp4_native_ops.py`
- `src/optimization/int8_native_ops.py`
- `src/core/model_loader.py`（`_dit_comfy_quant_ops`、`prepare_model_structure`）
- DiT `dit_3b` / `dit_7b` 已以相同方式传递 `operations=`

在 registry 变更之前，只要原生 ops 路径生效，两个 3B 包即可作为 `QuantizedTensor` 加载（约 1957 MB NVFP4 / 约 3342 MB INT8 CUDA DiT 权重，约 210 个量化 Linear）。

### 当时缺少什么

`MODEL_REGISTRY` 已列出 7B 条目：

- `seedvr2_7b_int8_convrot.safetensors`
- `seedvr2_7b_nvfp4.safetensors`
- sharp 变体

但 **没有** 列出对应的 3B。缺少 registry 条目时：

- UI / 下载 / 默认列表可能省略这些包
- 工具与文档相对 7B 缺少一等公民对等

### v1.4 为 3B 增加的内容

以与 7B 相同的 precision 标签登记两个 3B 包：

| 文件名 | `ModelInfo` |
|---|---|
| `seedvr2_3b_int8_convrot.safetensors` | `size="3B"`, `precision="int8_tensorwise_convrot"` |
| `seedvr2_3b_nvfp4.safetensors` | `size="3B"`, `precision="nvfp4"` |

文件名已含 `3b`，因此 `_create_new_runner` 中的历史配置目录选择本就会选到 `configs_3b`（完整函数见 §3.2）。

**重要：** 仅登记 registry **不会**实现量化。原生显存仍来自内容检测 + 构建时 ops。Registry 使 3B 包在列表与文档上与 7B 对等。

---

## 2. 新增 / 修改的文件名

v1.4 的代码变更只有 **一个** 源文件：

| 文件 | 作用 |
|---|---|
| `src/utils/model_registry.py` | 增加 3B INT8 / NVFP4 的 `MODEL_REGISTRY` 条目 |

本版本文档：

| 文件 | 作用 |
|---|---|
| `md/SEEDVR2_3B_INT8_NVFP4_REGISTRY_GUIDE.md` | 英文指南 |
| `zhmd/SEEDVR2_3B_INT8_NVFP4_REGISTRY_GUIDE.md` | 本中文指南 |
| `zhmd/v1.4.md` | 中文发布说明 |
| `release/_gh_v1.4_en_with_switcher.md` | 英文 GitHub Release 正文 |
| `md/changelog.md` / `zhmd/changelogzh.md` | 仅更新 v1.4 Summary |

未改动的 DiT 路径（与 v1.3 相同）：

| 文件 | 作用 |
|---|---|
| `src/core/model_loader.py` | 基于内容的构建时 ops；materialize 清空 `_dit_checkpoint` |
| `src/core/generation_phases.py` | NVFP4 autocast 跳过：`_dit_comfy_quant_native` + 实时 `checkpoint_is_nvfp4(_dit_checkpoint)` |
| `src/core/model_configuration.py` | 历史规则 `"7b" in dit_model` 选择配置目录 |

---

## 3. 新增 / 修改代码全文

### 3.1 `src/utils/model_registry.py` — `MODEL_REGISTRY` 全文

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

v1.4 **仅新增** 的两行是：

```python
    # HSWQ INT8 / NVFP4 (native VRAM path; same as 7B)
    "seedvr2_3b_int8_convrot.safetensors": ModelInfo(size="3B", precision="int8_tensorwise_convrot"),
    "seedvr2_3b_nvfp4.safetensors": ModelInfo(size="3B", precision="nvfp4"),
```

### 3.2 未改动 — `_create_new_runner` 配置选择（`model_configuration.py`）

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

因为 `seedvr2_3b_*.safetensors` 不含 `"7b"`，该规则在无需改写 DiT 路径的情况下选择 `configs_3b`。

---

## 4. 3B registry 变更的含义

### Registry 条目

- 将 `seedvr2_3b_int8_convrot.safetensors` 与 `seedvr2_3b_nvfp4.safetensors` 登记为官方 DiT，且 `size="3B"`。
- precision 字符串与 7B 对齐（`int8_tensorwise_convrot`、`nvfp4`），便于工具与文档一致。
- 使 UI / 下载 / 列表获得与 7B HSWQ 包相同的一等公民名称。

### 本次变更 **不是**

- 不是第二套量化实现。
- 不替代基于内容的 `*.comfy_quant` 检测。
- 不改写 `_create_new_runner`、`prepare_model_structure`，也不改写 NVFP4 autocast 跳过条件。
- 对这些文件名，配置目录选择已可通过历史规则 `"7b" in dit_model` 正确工作。

---

## 5. 既有原生加载路径（全文 — v1.4 未改）

要理解「为何登记 3B 包就足够」，需要完整周边代码：量化显存早已按内容检测。

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

### 5.2 `prepare_model_structure` 全文 — `model_loader.py`

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

### 5.3 `materialize_model` 全文 — `model_loader.py`

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

### 5.4 Upscale 时的 NVFP4 autocast 跳过 — `generation_phases.py`

位于 `upscale_all_batches` 内的连续块（dtype 检测到 `dit_inference` 计时结束）：

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

### 5.5 `nvfp4_native_ops.py` 全文

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

### 5.6 既有路径的含义

| 部件 | 作用 |
|---|---|
| `checkpoint_is_nvfp4` / `checkpoint_is_hswq_int8` | 扫描 `*.comfy_quant` 内容 |
| `_dit_comfy_quant_ops` | 选择 NVFP4 或 INT8 构建时 ops |
| `prepare_model_structure` | 在 meta 创建前注入 `operations=`；设置 `_dit_comfy_quant_native` |
| `materialize_model` | 加载权重；加载后清空 `_dit_checkpoint` |
| Upscale autocast 跳过 | `_dit_comfy_quant_native` 与 `checkpoint_is_nvfp4(_dit_checkpoint)` |

v1.4 **不**改动该路径。增加 registry 行只是让同一路径在 UI/工具中对已列出的 3B 包名称可见。

---

## 审计锚点

| 项目 | 值 |
|---|---|
| 发布 | `v1.4` |
| 代码变更 | `src/utils/model_registry.py`（两条 3B HSWQ 条目） |
| 3B INT8 包 | `seedvr2_3b_int8_convrot.safetensors` |
| 3B NVFP4 包 | `seedvr2_3b_nvfp4.safetensors` |
| 配置规则 | `"7b" in dit_model` → `configs_7b`，否则 `configs_3b` |
| 原生显存 | 内容 `*.comfy_quant` + 构建时 `comfy.ops.mixed_precision_ops` |
| 参考 NVFP4 质量 | MSE **6.534034**，SSIM **0.975915** |

---

## 与请求大纲的对应

| 请求章节 | 本指南 |
|---|---|
| ① 3B convrot INT8 / NVFP4 追加 | §1 |
| ② 新增 / 修改的文件名 | §2 |
| ③ 新增 / 修改代码全文 | §3 |
| ④ 含义 | §4 |
| 既有原生路径（全文、未改） | §5 |
