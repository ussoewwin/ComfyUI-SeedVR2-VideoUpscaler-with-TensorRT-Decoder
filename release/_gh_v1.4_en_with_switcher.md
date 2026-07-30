<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/blob/main/zhmd/v1.4.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

**Tag:** `v1.4`  
**Scope:** 3B INT8 / NVFP4 registry registration in `MODEL_REGISTRY`  
**Date:** 2026-07-31

This release registers **3B** HSWQ INT8 ConvRot and NVFP4 DiT packs in `MODEL_REGISTRY` the same way as 7B. Quantized VRAM remains content-detected (`*.comfy_quant` + construction-time `comfy.ops.mixed_precision_ops`). Filenames already contain `3b`, so config folder selection stays on the historical `"7b" in dit_model` rule (`configs_3b`).

**Guide (full write-up):**  
https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/blob/main/md/SEEDVR2_3B_INT8_NVFP4_REGISTRY_GUIDE.md

**Related prior guides:**  
- https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/blob/main/md/SEEDVR2_NVFP4_AND_TORCH_COMPILE_GUIDE.md  
- https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/blob/main/md/SEEDVR2_INT8_NATIVE_OPS_GUIDE.md

---

## 1. Summary

| Topic | Status |
|-------|--------|
| **3B INT8 ConvRot pack** | Registered as `seedvr2_3b_int8_convrot.safetensors` (`precision="int8_tensorwise_convrot"`) |
| **3B NVFP4 pack** | Registered as `seedvr2_3b_nvfp4.safetensors` (`precision="nvfp4"`) |
| **Native VRAM path** | Unchanged: content-based `*.comfy_quant` + construction-time `comfy.ops.mixed_precision_ops` (same as 7B) |
| **Config folder selection** | Unchanged from v1.3 (`"7b" in dit_model` → `configs_7b`, else `configs_3b`) |

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

Registry registration alone does **not** implement quantization; packed VRAM still comes from content detection + construction-time ops.

---

## 3. Code surface in this tag

| Path | Role |
|------|------|
| `src/utils/model_registry.py` | 3B INT8 / NVFP4 registry entries |
| `md/SEEDVR2_3B_INT8_NVFP4_REGISTRY_GUIDE.md` | Technical guide |
| `md/changelog.md` / `zhmd/changelogzh.md` | v1.4 changelog entries |

DiT loader / autocast / `_create_new_runner` paths match **v1.3**.

---

## 4. Expected effect

- 3B INT8 ConvRot / NVFP4 packs appear as registered DiT models
- Native packed VRAM continues when `*.comfy_quant` is present (unchanged load architecture)
- `seedvr2_3b_*.safetensors` still select `configs_3b` via filename

---

## 5. Scope of this GitHub Release

Tags current `main` on **ComfyUI-SeedVR2_VideoUpscaler** for the 3B registry registration.

Includes:

- 3B INT8 / NVFP4 registry rows
- technical guide and changelog v1.4

Does not add binary release assets (model safetensors are not attached to this GitHub Release).
