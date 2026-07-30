<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/blob/main/zhmd/v1.4.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

**Tag:** `v1.4`  
**Code commit:** `2f90466` (3B registry + durable NVFP4 autocast skip)  
**Docs:** `525de59` (guide), `3c78610` (changelog)  
**Date:** 2026-07-31

This release registers **3B** HSWQ INT8 ConvRot and NVFP4 DiT packs the same way as 7B, resolves 3B/7B config folders via registry size, and adds a **durable** `runner._dit_is_nvfp4` flag so NVFP4 autocast skip still works after materialize clears `_dit_checkpoint`.

**Guide (full write-up):**  
https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/blob/main/md/SEEDVR2_3B_INT8_NVFP4_AND_DURABLE_AUTOCAST_FIX_GUIDE.md

**Related prior guides:**  
- https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/blob/main/md/SEEDVR2_NVFP4_AND_TORCH_COMPILE_GUIDE.md  
- https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/blob/main/md/SEEDVR2_INT8_NATIVE_OPS_GUIDE.md

---

## 1. Summary

| Topic | Status |
|-------|--------|
| **3B INT8 ConvRot pack** | Registered as `seedvr2_3b_int8_convrot.safetensors` (`precision="int8_tensorwise_convrot"`) |
| **3B NVFP4 pack** | Registered as `seedvr2_3b_nvfp4.safetensors` (`precision="nvfp4"`) |
| **Native VRAM path** | Unchanged architecture: content-based `*.comfy_quant` + construction-time `comfy.ops.mixed_precision_ops` (same as 7B) |
| **Config folder selection** | `_create_new_runner` uses `resolve_dit_config_folder()` instead of raw `"7b" in dit_model` |
| **NVFP4 autocast bug** | Skip depended on `_dit_checkpoint` after materialize set it to `None` → skip never armed |
| **Durable fix** | New `runner._dit_is_nvfp4` survives path clear; upscale phase reads that flag only |

Example model files:

- `models/SEEDVR2/seedvr2_3b_int8_convrot.safetensors`
- `models/SEEDVR2/seedvr2_3b_nvfp4.safetensors`

---

## 2. What changed

### 2.1 Register 3B INT8 / NVFP4 in `MODEL_REGISTRY`

In `src/utils/model_registry.py`:

```python
    # HSWQ INT8 / NVFP4 (native VRAM path; same as 7B)
    "seedvr2_3b_int8_convrot.safetensors": ModelInfo(size="3B", precision="int8_tensorwise_convrot"),
    "seedvr2_3b_nvfp4.safetensors": ModelInfo(size="3B", precision="nvfp4"),
```

This makes the 3B packs first-class for listing and for `resolve_dit_config_folder()` (`size="3B"` → `configs_3b`). Registry registration alone does **not** implement quantization; packed VRAM still comes from content detection + construction-time ops.

### 2.2 Resolve config folder via registry

`src/core/model_configuration.py` `_create_new_runner`:

**Before:** `'./configs_7b' if "7b" in dit_model else './configs_3b'`  
**After:** `resolve_dit_config_folder(os.path.basename(dit_model))`

Resolution order: registry `size` → basename `7b`/`3b` tokens → default `configs_3b`.

### 2.3 Durable `_dit_is_nvfp4` for autocast skip

**Bug:** After weights load, `materialize_model` sets `runner._dit_checkpoint = None`. The upscale phase used:

```python
nvfp4_native = (
    bool(getattr(runner, "_dit_comfy_quant_native", False))
    and checkpoint_is_nvfp4(getattr(runner, "_dit_checkpoint", None))
)
```

So `checkpoint_is_nvfp4(None)` was always false at inference. Autocast could enable → LayerNorm/RMSNorm → float32 activations → `comfy_kitchen` `quantize_nvfp4` rejects float32 (FP16/BF16 only). This affected **7B and 3B** NVFP4.

**Fix:**

1. `prepare_model_structure` (`model_loader.py`):

```python
runner._dit_is_nvfp4 = bool(create_kwargs) and checkpoint_is_nvfp4(checkpoint_path)
```

2. Upscale (`generation_phases.py`):

```python
nvfp4_native = bool(getattr(runner, "_dit_is_nvfp4", False))
use_autocast = (
    not nvfp4_native
    and dit_dtype != ctx['compute_dtype']
    and ctx['dit_device'].type != 'mps'
)
```

3. DiT cache reuse (`model_configuration.py`) recomputes `_dit_is_nvfp4` / `_dit_comfy_quant_native` from the restored checkpoint path.

`_dit_comfy_quant_native` alone is not enough for the skip: INT8 native also sets it; autocast skip is NVFP4-specific.

---

## 3. Code surface in this tag

| Path | Role |
|------|------|
| `src/utils/model_registry.py` | 3B INT8 / NVFP4 registry entries; `resolve_dit_config_folder` |
| `src/core/model_loader.py` | Set durable `_dit_is_nvfp4` at structure prepare |
| `src/core/generation_phases.py` | Autocast skip reads `_dit_is_nvfp4` only |
| `src/core/model_configuration.py` | Registry-based config folder; restore flags on DiT cache reuse |
| `md/SEEDVR2_3B_INT8_NVFP4_AND_DURABLE_AUTOCAST_FIX_GUIDE.md` | Full technical guide |
| `md/changelog.md` / `zhmd/changelogzh.md` | v1.4 changelog entries |

Also in recent `main` history (same release window): `benchmark/seedvr2_int8_bench.py`, `benchmark/seedvr2_nvfp4_bench.py` tracked under `benchmark/`.

---

## 4. Expected effect

- 3B INT8 ConvRot / NVFP4 packs appear as registered DiT models with correct `configs_3b`
- Native packed VRAM path continues to work when `*.comfy_quant` is present (unchanged load architecture)
- NVFP4 upscale no longer loses autocast skip after materialize (7B and 3B)
- Cache reuse keeps native-quant flags consistent

---

## 5. Scope of this GitHub Release

Tags current `main` on **ComfyUI-SeedVR2_VideoUpscaler** (includes `2f90466` and follow-up docs/benchmark commits).

Includes:
- 3B INT8 / NVFP4 registry + config-folder resolution
- durable NVFP4 autocast-skip flag
- technical guide and changelog v1.4
- tracked INT8 / NVFP4 benchmark scripts under `benchmark/`

Does not add binary release assets (model safetensors are not attached to this GitHub Release).
