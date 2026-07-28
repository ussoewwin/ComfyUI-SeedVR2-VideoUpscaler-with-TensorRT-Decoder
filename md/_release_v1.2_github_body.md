**Tag:** `v1.2`  
**Commit:** `c56373b` (includes feature commit `a14db91`)  
**Date:** 2026-07-28

This release adds **native NVFP4 loading** for SeedVR2 DiT weights (`format == nvfp4` + `weight_scale` / `weight_scale_2`) via construction-time `comfy.ops.mixed_precision_ops`, and **Windows / inductor fixes** so FP16 VAE `torch.compile` no longer fails with cp932 decode or `aten.bmm` fallback+decomp asserts.

**Guide (full write-up):** https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/blob/main/md/SEEDVR2_NVFP4_AND_TORCH_COMPILE_GUIDE.md

---

## 1. Summary

| Topic | Status |
|-------|--------|
| **NVFP4 problem** | Post-load Linear replace never hits `comfy.ops` `_load_quantized_module` → packed NVFP4 expands → VRAM savings lost (same family as INT8) |
| **NVFP4 fix** | Inject `mixed_precision_ops` when building NaDiT → load with QuantizedTensor path; skip DiT autocast; cast Linear activations to FP16/BF16 when needed |
| **torch.compile problem** | VAE compile on Japanese Windows: `'cp932' codec can't decode…` → uncompiled fallback; then `AssertionError: both a fallback and a decomp for same op: aten.bmm.default` |
| **torch.compile fix** | UTF-8 jinja load + OEM/cp932 `errors="replace"` + `make_fallback(..., override_decomp=True)` when op already in decomp table; apply at custom-node import |
| **Scope** | **DiT** NVFP4 (and existing INT8 path). **VAE** remains FP16 (compile fixes only) |

---

## 2. NVFP4 load path (construction-time ops)

1. **Detect** via `checkpoint_is_nvfp4()` (`*.comfy_quant` → `format == nvfp4`).
2. **Build** DiT under meta with `create_object(..., operations=get_nvfp4_mixed_precision_ops(...))`.
3. **Prep before load:** reuse INT8 helpers (`prepare_hswq_state_dict_for_comfy_ops`, `patch_ops_factory_device`).
4. If GPU lacks native NVFP4 compute, put `"nvfp4"` in `disabled` so storage stays packed and matmul dequantizes.
5. Wrap `ops.Linear.forward` so activations are FP16/BF16 when `quantize_nvfp4` would reject float32.
6. In DiT upscale phase, **skip `torch.autocast`** for native NVFP4 so LayerNorm/RMSNorm do not feed float32 into NVFP4 Linear.

Independent of the GGUF branch; shares the INT8 construction-time ops family.

---

## 3. torch.compile / inductor fixes

| Failure | Countermeasure |
|---------|----------------|
| Jinja / template `open()` under locale **cp932** | Patch `load_template` to `encoding="utf-8"`; rebind existing `functools.partial` hooks |
| `SUBPROCESS_DECODE_ARGS = ('oem',)` strict decode | Use `(encoding, "replace")` / `("oem", "replace")` |
| `both a fallback and a decomp for same op: aten.bmm.default` | Wrap `make_fallback` → `override_decomp=True` when op is already in the decomp table; rebind `graph.make_fallback` |
| Patch timing | Call `_fix_inductor_windows_encoding()` from `__init__.py` at import (and keep model_configuration call) |

Does **not** disable compile or remap `max-autotune` for these errors.

---

## 4. Code surface in this tag

| Path | Role |
|------|------|
| `src/optimization/nvfp4_native_ops.py` | **New** — detect / `mixed_precision_ops` / activation cast |
| `src/core/model_loader.py` | Shared INT8/NVFP4 construction-time ops + load prep |
| `src/core/generation_phases.py` | Skip DiT autocast for native NVFP4 |
| `src/core/fix_inductor.py` | UTF-8 jinja, OEM replace, bmm/`make_fallback` override |
| `__init__.py` | Apply inductor fix at extension import |
| `src/common/config.py`, `src/utils/model_registry.py` | Comments / wording |
| `md/SEEDVR2_NVFP4_AND_TORCH_COMPILE_GUIDE.md` | Full guide (sections 1–9) |

Feature work landed in `a14db91`; guide + tag tip at `c56373b`.

---

## 5. Usage notes

- Requires a ComfyUI build that provides `comfy.ops.mixed_precision_ops` and NVFP4 / kitchen support as needed for your GPU.
- Place DiT packs under `models/SEEDVR2/` (or your node’s model dir); select the NVFP4 safetensors in the node UI.
- VAE remains FP16; enable VAE `torch.compile` as usual after this fix lands.
- INT8 native path from **v1.1** remains available.

---

## 6. Scope of this GitHub Release

Tags commit **`c56373b`** on **ComfyUI-SeedVR2_VideoUpscaler** (NVFP4 DiT native ops + VAE torch.compile inductor fixes + guide).

**Not included:** uploading binary model weights as release assets (distribute packs separately).
