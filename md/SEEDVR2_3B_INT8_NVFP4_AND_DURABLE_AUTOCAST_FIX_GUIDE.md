# SeedVR2 — 3B INT8 ConvRot / NVFP4 Registry + Durable NVFP4 Autocast Skip

<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../zhmd/SEEDVR2_3B_INT8_NVFP4_AND_DURABLE_AUTOCAST_FIX_GUIDE.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

Target custom node: `ComfyUI/custom_nodes/seedvr2_videoupscaler`  
Canonical commit: `2f90466cc78312f21677012eeabfd0ddcb7259d9`  
(`Support 3B NVFP4/INT8 registry and durable NVFP4 autocast skip`)

This guide documents the **2026-07-31** fix for:

1. Registering **3B** HSWQ INT8 ConvRot and NVFP4 packs the same way as 7B
2. A **durable** NVFP4 autocast-skip flag so inference still skips autocast after materialize clears the checkpoint path

Related prior guides:

- `md/SEEDVR2_NVFP4_AND_TORCH_COMPILE_GUIDE.md` — NVFP4 native ops and torch.compile
- `md/SEEDVR2_INT8_NATIVE_OPS_GUIDE.md` — INT8 construction-time `comfy.ops` (if present)

Model packs (local examples):

- `models/SEEDVR2/seedvr2_3b_nvfp4.safetensors`
- `models/SEEDVR2/seedvr2_3b_int8_convrot.safetensors`

---

## 1. Adding 3B ConvRot INT8 / NVFP4

### What already worked before this commit

Native VRAM for HSWQ INT8 / NVFP4 does **not** depend on the filename alone. Detection is **content-based**:

- Scan checkpoint for `*.comfy_quant` metadata
- At DiT **construction** (meta), inject `comfy.ops.mixed_precision_ops` via `create_object(..., operations=...)`
- On load, keep `QuantizedTensor` storage (packed INT8 / NVFP4) instead of expanding to FP16/BF16

That path lived in:

- `src/optimization/nvfp4_native_ops.py`
- `src/optimization/int8_native_ops.py`
- `src/core/model_loader.py` (`_dit_comfy_quant_ops`, `_dit_needs_comfy_quant_prep`)
- DiT `dit_3b` / `dit_7b` already thread `operations=` the same way

Smoke before the registry fix already showed both 3B packs loading as `QuantizedTensor` (~1957 MB NVFP4 / ~3342 MB INT8 CUDA DiT weights, ~210 quantized Linears) **when** the native ops path ran.

### What was missing

`MODEL_REGISTRY` already listed 7B entries:

- `seedvr2_7b_int8_convrot.safetensors`
- `seedvr2_7b_nvfp4.safetensors`
- sharp variants

It did **not** list the 3B equivalents. Without registry entries:

- UI / download / default listing may omit the packs
- `resolve_dit_config_folder()` cannot use registry `size="3B"` and must fall back to basename heuristics (`"3b" in name`)
- Runner config selection historically used raw `"7b" in dit_model`, which is fragile for names that do not contain `7b`/`3b` tokens

### What this commit adds for 3B

Register the two 3B packs with the same precision tags as 7B:

| Filename | `ModelInfo` |
|---|---|
| `seedvr2_3b_int8_convrot.safetensors` | `size="3B"`, `precision="int8_tensorwise_convrot"` |
| `seedvr2_3b_nvfp4.safetensors` | `size="3B"`, `precision="nvfp4"` |

Also switch `_create_new_runner` to `resolve_dit_config_folder(os.path.basename(dit_model))` so 3B / 7B config folders come from registry size (or basename tokens), not a hard-coded `"7b" in dit_model` substring only.

**Important:** Registry registration alone does not implement quantization. Native VRAM still comes from content detection + construction-time ops. Registry makes 3B packs first-class for listing, config folder resolution, and documentation parity with 7B.

---

## 2. Added / modified filenames

Commit `2f90466` touched **four** source files (no new modules):

| File | Role |
|---|---|
| `src/utils/model_registry.py` | Add 3B INT8 / NVFP4 `MODEL_REGISTRY` entries; hosts `resolve_dit_config_folder` (pre-existing helper, now used by runner creation) |
| `src/core/model_loader.py` | Set durable `runner._dit_is_nvfp4` in `prepare_model_structure` |
| `src/core/generation_phases.py` | Autocast skip reads `_dit_is_nvfp4` only; drop live `checkpoint_is_nvfp4(_dit_checkpoint)` |
| `src/core/model_configuration.py` | Import helpers; use `resolve_dit_config_folder` in `_create_new_runner`; restore `_dit_is_nvfp4` / `_dit_comfy_quant_native` on DiT cache reuse |

Not part of `2f90466` (separate follow-up commits):

- `8848760` — remove root `seedvr2_int8_bench.py`
- `a4a5a96` — track `benchmark/seedvr2_int8_bench.py`, `benchmark/seedvr2_nvfp4_bench.py`; stop ignoring `benchmark/`

---

## 3. Full text of added / modified code

### 3.1 `src/utils/model_registry.py` — new registry lines

```python
    # HSWQ INT8 / NVFP4 (native VRAM path; same as 7B)
    "seedvr2_3b_int8_convrot.safetensors": ModelInfo(size="3B", precision="int8_tensorwise_convrot"),
    "seedvr2_3b_nvfp4.safetensors": ModelInfo(size="3B", precision="nvfp4"),
```

### 3.2 `src/utils/model_registry.py` — `resolve_dit_config_folder` (used by this fix)

```python
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
```

### 3.3 `src/core/model_configuration.py` — imports

```python
from ..utils.model_registry import resolve_dit_config_folder
from ..optimization.nvfp4_native_ops import checkpoint_is_nvfp4
from ..optimization.int8_native_ops import checkpoint_is_hswq_int8
```

### 3.4 `src/core/model_configuration.py` — `_create_new_runner` config path

**Before:**

```python
    config_path = os.path.join(script_directory, 
                              './configs_7b' if "7b" in dit_model else './configs_3b', 
                              'main.yaml')
```

**After:**

```python
    config_folder = resolve_dit_config_folder(os.path.basename(dit_model))
    config_path = os.path.join(script_directory, config_folder, 'main.yaml')
```

### 3.5 `src/core/model_configuration.py` — DiT cache reuse flags

```python
        # Durable native-quant flags (survive _dit_checkpoint clear after materialize)
        cp = runner._dit_checkpoint
        runner._dit_is_nvfp4 = checkpoint_is_nvfp4(cp)
        runner._dit_comfy_quant_native = runner._dit_is_nvfp4 or checkpoint_is_hswq_int8(cp)
```

### 3.6 `src/core/model_loader.py` — durable flag at structure prepare

```python
        runner._dit_comfy_quant_native = bool(create_kwargs)
        # Durable after materialize clears _dit_checkpoint (generation_phases autocast skip).
        runner._dit_is_nvfp4 = bool(create_kwargs) and checkpoint_is_nvfp4(checkpoint_path)
```

### 3.7 `src/core/generation_phases.py` — autocast skip (after)

```python
            # Use durable _dit_is_nvfp4: materialize_model clears _dit_checkpoint.
            nvfp4_native = bool(getattr(runner, "_dit_is_nvfp4", False))
            debug.start_timer(f"dit_inference_{upscale_idx+1}")
            with torch.no_grad():
                use_autocast = (
                    not nvfp4_native
                    and dit_dtype != ctx['compute_dtype']
                    and ctx['dit_device'].type != 'mps'
                )
```

Removed import:

```python
from ..optimization.nvfp4_native_ops import checkpoint_is_nvfp4
```

---

## 4. Meaning of the 3B / registry / config changes

### Registry entries

- Make `seedvr2_3b_int8_convrot.safetensors` and `seedvr2_3b_nvfp4.safetensors` official DiT entries with `size="3B"`.
- Align precision strings with 7B (`int8_tensorwise_convrot`, `nvfp4`) so tooling and docs stay consistent.
- Enable `resolve_dit_config_folder` to return `configs_3b` from registry size even if a future filename omits the `3b` token (as long as the name is registered).

### `resolve_dit_config_folder` in `_create_new_runner`

- Old rule: `"7b" in dit_model` → `configs_7b`, else `configs_3b`.
- New rule: registry size first, then basename `7b`/`3b`, default `configs_3b`.
- Prevents mis-selecting 7B YAML for names that do not contain `"7b"` but are registered as 7B, and makes 3B quantized names resolve cleanly via `size="3B"`.

### Cache reuse flag restore

- When a cached DiT is reused, `_dit_checkpoint` is set again to the on-disk path.
- Immediately recompute `_dit_is_nvfp4` and `_dit_comfy_quant_native` from that path so later phases do not inherit stale or missing flags from a previous run.

### What these changes are **not**

- They are not a second quantization implementation.
- They do not replace content-based `*.comfy_quant` detection.
- They do not by themselves fix float32 → `quantize_nvfp4` failures (that is section 5–7).

---

## 5. NVFP4 bug overview

### Symptom

Native NVFP4 DiT (7B **and** 3B) can fail during upscale inference with a kitchen / CUDA error when `quantize_nvfp4` receives **float32** activations. Comfy Kitchen CUDA dispatch accepts FP16/BF16 only (`DISPATCH_HALF_DTYPE`); float32 is rejected (dtype code 0).

### Intended mitigation (already documented in the NVFP4 guide)

In the DiT upscale phase, **skip `torch.autocast`** for native NVFP4 so LayerNorm / RMSNorm do not promote activations to float32 under autocast before they hit NVFP4 Linear `from_float` / quantize.

ComfyUI UNet/Flux-style NVFP4 paths typically keep activations in FP16/BF16 without wrapping the whole forward in autocast. SeedVR2 previously tried to mirror that by skipping autocast when NVFP4 native was detected.

### Broken detection after materialize

`materialize_model` clears the checkpoint path after weights are loaded:

```python
    if is_dit:
        runner._dit_checkpoint = None
        runner._dit_dtype_override = None
```

The **old** autocast-skip condition was:

```python
            nvfp4_native = (
                bool(getattr(runner, "_dit_comfy_quant_native", False))
                and checkpoint_is_nvfp4(getattr(runner, "_dit_checkpoint", None))
            )
```

Timeline:

1. `prepare_model_structure` sets `_dit_checkpoint` to a real path and may set `_dit_comfy_quant_native = True`.
2. `materialize_model` loads weights, then sets `_dit_checkpoint = None`.
3. Upscale phase runs. `checkpoint_is_nvfp4(None)` is **False**.
4. `nvfp4_native` becomes **False** even when the live DiT is NVFP4 `QuantizedTensor`.
5. Autocast may enable → LayerNorm/RMSNorm → float32 → NVFP4 Linear → kitchen reject.

So the bug is **not** “3B lacks NVFP4 kernels”. It is **flag lifetime**: the skip depended on a path that is intentionally cleared after materialize. That affects **both** 3B and 7B NVFP4.

`_dit_comfy_quant_native` alone is also insufficient as a skip signal because INT8 native uses the same construction-time ops flag; skipping autocast is specifically required for NVFP4 kitchen quantize, not as a blanket rule for all quantized packs.

---

## 6. Full text of the durable fix (autocast-skip durability)

This section is the **durable / permanent** fix for the cleared-checkpoint autocast-skip bug (`_dit_is_nvfp4`).

### 6.1 Set flag when structure is prepared — `model_loader.py`

```python
        runner.dit = model
        runner._dit_checkpoint = checkpoint_path
        runner._dit_block_swap_config = block_swap_config
        runner._dit_comfy_quant_native = bool(create_kwargs)
        # Durable after materialize clears _dit_checkpoint (generation_phases autocast skip).
        runner._dit_is_nvfp4 = bool(create_kwargs) and checkpoint_is_nvfp4(checkpoint_path)
```

### 6.2 Materialize still clears the path (unchanged; intentional)

```python
    if is_dit:
        runner._dit_checkpoint = None
        runner._dit_dtype_override = None
```

`_dit_is_nvfp4` is **not** cleared here.

### 6.3 Inference uses only the durable flag — `generation_phases.py`

**Before (broken after materialize):**

```python
            nvfp4_native = (
                bool(getattr(runner, "_dit_comfy_quant_native", False))
                and checkpoint_is_nvfp4(getattr(runner, "_dit_checkpoint", None))
            )
```

**After:**

```python
            # Use durable _dit_is_nvfp4: materialize_model clears _dit_checkpoint.
            nvfp4_native = bool(getattr(runner, "_dit_is_nvfp4", False))
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
```

### 6.4 Restore on cache reuse — `model_configuration.py`

```python
        runner.dit = cache_context['cached_dit']
        runner._dit_checkpoint = find_model_file(dit_model, base_cache_dir)
        runner._dit_model_name = dit_model
        # Durable native-quant flags (survive _dit_checkpoint clear after materialize)
        cp = runner._dit_checkpoint
        runner._dit_is_nvfp4 = checkpoint_is_nvfp4(cp)
        runner._dit_comfy_quant_native = runner._dit_is_nvfp4 or checkpoint_is_hswq_int8(cp)
```

---

## 7. Meaning of the durable fix

### Design

| Attribute | Lifetime | Purpose |
|---|---|---|
| `_dit_checkpoint` | Cleared after materialize | Was only a temporary path for load; must not be required at inference |
| `_dit_comfy_quant_native` | Kept | “Construction used mixed_precision ops” (INT8 **or** NVFP4) |
| `_dit_is_nvfp4` | Kept across materialize; recomputed on cache reuse | “This DiT is NVFP4 native; skip autocast in upscale” |

### Why a dedicated bool

1. **Survives path clear** — Inference no longer calls `checkpoint_is_nvfp4(None)`.
2. **NVFP4-specific** — Autocast skip targets kitchen NVFP4 float32 rejection, not every quantized DiT.
3. **Cheap** — One `bool` on the runner; no safetensors re-scan at every upscale batch.
4. **Cache-safe** — Reuse path re-derives the flag from the restored checkpoint path before the next materialize clear.

### What success looks like

After `prepare_model_structure` for an NVFP4 pack:

- `_dit_is_nvfp4 is True`
- `_dit_comfy_quant_native is True`

After `materialize_model`:

- `_dit_checkpoint is None`
- `_dit_is_nvfp4` remains `True`

During upscale:

- `nvfp4_native is True` → `use_autocast is False`
- Activations stay in DiT compute dtype (FP16/BF16 path) without autocast promotion to float32 into NVFP4 Linear

### Scope reminder

- This durable flag fixes **autocast skip durability**.
- Packed VRAM still depends on construction-time `operations=` and `*.comfy_quant` content detection (unchanged architecture).
- 3B registry entries fix **product parity / config resolution** for `seedvr2_3b_*` packs; they do not replace the durable flag.

---

## Audit anchors

| Item | Value |
|---|---|
| Fix commit | `2f90466cc78312f21677012eeabfd0ddcb7259d9` |
| Files | `model_registry.py`, `model_loader.py`, `generation_phases.py`, `model_configuration.py` |
| New runner attribute | `runner._dit_is_nvfp4` |
| Cleared after materialize | `runner._dit_checkpoint` |
| 3B INT8 pack | `seedvr2_3b_int8_convrot.safetensors` |
| 3B NVFP4 pack | `seedvr2_3b_nvfp4.safetensors` |

---

## Mapping to the requested outline

| Requested section | This guide |
|---|---|
| ① 3B convrot INT8 / NVFP4 addition | §1 |
| ② Added / modified filenames | §2 |
| ③ Full text of added / modified code | §3 |
| ④ Meaning | §4 |
| ⑤ NVFP4 bug overview | §5 |
| ⑥ Full text of durable fix | §6 |
| ⑦ Meaning of that fix | §7 |
