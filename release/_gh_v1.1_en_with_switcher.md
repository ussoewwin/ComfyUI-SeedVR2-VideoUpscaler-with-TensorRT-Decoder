<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder/blob/main/zhmd/v1.1.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

**Tag:** `v1.1`  
**Commit:** `3515c33`  
**Date:** 2026-07-28

This release adds **native INT8 loading** for SeedVR2 DiT weights (`int8_tensorwise` + `comfy_quant` / `weight_scale`) via construction-time `comfy.ops.mixed_precision_ops`, so INT8 packs stay quantized through `load_state_dict` instead of expanding to full FP16.

**Guide (full write-up):** https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder/blob/main/md/SEEDVR2_INT8_NATIVE_OPS_GUIDE.md

---

## 1. Summary

| Topic | Status |
|-------|--------|
| **Problem** | Post-load Linear replace (GGUF-style) never hits `comfy.ops` `_load_quantized_module` → full INT8→FP16 expand → VRAM savings lost |
| **Fix** | Inject `mixed_precision_ops` when building NaDiT → load with QuantizedTensor path |
| **Scope** | **DiT only** (3B / 7B). VAE remains FP16 |
| **Gate** | `comfy_quant` JSON `format == int8_tensorwise` (not filename heuristics) |

---

## 2. Why post-load replace is wrong for this INT8 format

GGUF-style loaders swap Linears **after** weights are on the module. SeedVR2 INT8 packs that use `comfy_quant` need ComfyUI’s **state_dict load hooks** (`_load_from_state_dict` → `_load_quantized_module`). Building with plain `torch.nn.Linear` and swapping later skips that path.

---

## 3. Load path (construction-time ops)

1. **Detect** INT8 via `checkpoint_is_hswq_int8` (`*.comfy_quant` → `int8_tensorwise`).
2. **Build** DiT under meta with `create_object(..., operations=get_hswq_mixed_precision_ops(fp16))` (YAML unchanged; ops injected only for this path).
3. **Prep before load:** move `comfy_quant` tensors to CPU; patch `factory_kwargs["device"]` to the real device so QuantizedTensor is not stuck on meta.
4. **`load_state_dict`** — markers become QuantizedTensor; other layers stay normal Parameters.
5. Independent of the GGUF branch.

---

## 4. Code surface in this tag

| Path | Role |
|------|------|
| `src/optimization/int8_native_ops.py` | Detect / `mixed_precision_ops` / comfy_quant→CPU / factory device patch |
| `src/common/config.py` | `create_object(..., **extra_kwargs)` for `operations=` |
| `src/core/model_loader.py` | Detect → inject ops → prep → load |
| `src/utils/model_registry.py` | Register `seedvr2_7b_int8_convrot` / sharp; `resolve_dit_config_folder` uses registry `size` |
| `src/models/dit_3b/**`, `dit_7b/**` | Propagate `operations` / `ops.Linear` through MLP, embedding, patch, attention, window blocks |
| `seedvr2_int8_bench.py` | FP16 vs native INT8 DiT comparison (same loader path) |

---

## 5. Usage notes

- Requires a ComfyUI build that provides `comfy.ops.mixed_precision_ops`.
- Place DiT packs under `models/SEEDVR2/` (or your node’s model dir); select the INT8 safetensors in the node UI or pass paths to the bench script.
- Example weights: `seedvr2_7b_int8_convrot.safetensors`, `seedvr2_7b_sharp_int8_convrot.safetensors`.
- VAE example remains FP16 (e.g. `ema_vae_fp16.safetensors`).

Bench (from this custom node root):

```bash
python seedvr2_int8_bench.py \
  --fp16 seedvr2_ema_7b_fp16.safetensors \
  --int8 seedvr2_7b_int8_convrot.safetensors \
  --vae ema_vae_fp16.safetensors
```

---

## 6. Scope of this GitHub Release

Tags commit **`3515c33`** on **ComfyUI-SeedVR2_VideoUpscaler** (SeedVR2 DiT native INT8 via construction-time `comfy.ops`).

**Not included:** uploading binary model weights as release assets (distribute packs separately).

---

## 7. Links

- **Repository:** https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler  
- **This release:** https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/releases/tag/v1.1  
- **Commit:** https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/commit/3515c330ad1ec2a6e20d3fb4e905cd0465a142b1  
- **Guide:** https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/blob/main/md/SEEDVR2_INT8_NATIVE_OPS_GUIDE.md  
