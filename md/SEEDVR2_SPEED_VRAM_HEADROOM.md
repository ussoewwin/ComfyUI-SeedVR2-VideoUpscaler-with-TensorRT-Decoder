# SeedVR2 Video Upscaler — Speed / VRAM Headroom Beyond max-autotune

Target custom node: `ComfyUI/custom_nodes/seedvr2_videoupscaler`
Code commit: `a14db91b31c08bee62055e17521d4f1537bef03c`
(`feat: NVFP4 DiT native ops and VAE torch.compile inductor fixes`)

This guide catalogs remaining **speed** and **VRAM** levers after abandoning
cudagraphs-based `max-autotune`. The max-autotune experiment was dropped
because the first VAE encode blows past 16 GB: CUDA Graph capture permanently
reserves ~2x the eager peak in private pools (one per shape variant), and
`reclaim_transient_vram_after_torch_compile` fires only *after* the peak and
cannot reclaim live graph pools. Production mode is therefore
`max-autotune-no-cudagraphs`.

Environment assumed: RTX 50-series (sm_120, Blackwell), 16 GB VRAM, WDDM,
`sageattention` installed (`sageattn3` package **not** installed),
`torch.compile` active with cudagraphs disabled.

Related guides: `md/SEEDVR2_NVFP4_AND_TORCH_COMPILE_GUIDE.md`,
`md/SEEDVR2_INT8_NATIVE_OPS_GUIDE.md`.

---

## 1. Settings-only levers (no code changes)

### 1.1 DiT `attention_mode: sageattn_2` — largest speed lever

- Node: `SeedVR2LoadDiTModel`, `attention_mode` combo
  (`src/interfaces/dit_model_loader.py:104-118`). Default is `sdpa`.
- `sageattention` is already installed in `python_embeded`; on sm_120 the
  FP8-PV per-warp path is used. Video latents produce long attention
  sequences, so attention dominates DiT compute — expected speedup is large.
- `sageattn_3` (Blackwell-labeled option) requires the separate `sageattn3`
  package, which is not installed. Use `sageattn_2`.
- Coexistence with `torch.compile`: SageAttention is a custom CUDA kernel, so
  Dynamo inserts graph breaks around it. With `fullgraph=False` (default) this
  is harmless.
- `flash_attn_3` is Hopper-only; not applicable to sm_120.

### 1.2 VAE `encode_tiled=True` / `decode_tiled=True` — largest VRAM lever

- Node: `SeedVR2LoadVAEModel` (`src/interfaces/vae_model_loader.py:58-115`).
  Both default to **False**; schema tile size 1024 / overlap 128.
- Directly removes the first-encode activation peak identified in the
  max-autotune post-mortem (full-resolution conv activations scale with
  ~(tile/H) x (tile/W) instead of the whole frame).
- Temporal slicing is preserved inside the tiled path
  (`src/models/video_vae_v3/modules/attn_video_vae.py:1403`, `slicing_encode`).
- Cost: overlap regions are computed twice and blended — encode/decode get
  somewhat slower, but DiT dominates total runtime, so end-to-end impact is
  small. Edge tiles create a few extra shape variants; with cudagraphs
  disabled this only costs a little extra compile time, no permanent pools.

### 1.3 `uniform_batch_size=True` — fewer shape variants

- Node: main upscaler (`src/interfaces/video_upscaler.py:125`).
- Pads the final batch to `batch_size`, so compiled shape count stays bounded
  (less recompilation, fewer inductor variants). Trades a little wasted
  compute on padded tail frames.

### 1.4 `batch_size` upward (subject to the 4n+1 constraint)

- Once quantized DiT + tiled VAE have freed VRAM, raising `batch_size`
  increases DiT throughput (fewer sampling launches, better utilization).
- Remember `temporal_overlap` frames do not advance the clip — effective new
  frames per batch = `batch_size - temporal_overlap`.

### 1.5 `cache_model=True` + `offload_device=cpu`

- Keeps DiT/VAE resident in system RAM between workflow runs; eliminates
  reload time for batch processing. Costs RAM, not VRAM.

### 1.6 `blocks_to_swap` / `swap_io_components` — only for fp16/int8 DiT

- Node: `SeedVR2LoadDiTModel` (`dit_model_loader.py:57-81`).
- Meaningful when running fp16 (~14 GB) or int8 weights. With NVFP4
  (~3.5 GB) it is usually unnecessary; BlockSwap adds PCIe transfer latency
  during sampling, so treat it as an emergency valve, not a default.

### 1.7 NVFP4 DiT checkpoint — if fp16 is still in use

- `models/SEEDVR2/seedvr2_7b_nvfp4.safetensors` already exists alongside
  fp16 / int8_convrot variants.
- Native path (commit a14db91): construction-time `comfy.ops` injection keeps
  weights packed (~4x smaller than fp16) and runs native NVFP4 matmul on
  sm_120 via `comfy_kitchen.quantize_nvfp4` (`generation_phases.py:717-729`,
  `src/optimization/nvfp4_native_ops.py`). Simultaneous speed + VRAM win.

---

## 2. Code-level levers (require edits)

### 2.1 Enable `cudnn.benchmark` in the ComfyUI path — free VAE speed

- SeedVR2 sets `torch.backends.cudnn.benchmark = True` only in
  `init_torch` (`src/common/distributed/basic.py:64-70`), which is **never
  called** from the ComfyUI runtime path (only re-exported in
  `src/common/distributed/__init__.py`). Dead code for ComfyUI users.
- ComfyUI core enables benchmark only under `--fast AutoTune`
  (`comfy/model_management.py:540`), so with default flags every VAE conv
  runs with heuristic algorithm selection.
- Proposal: set `torch.backends.cudnn.benchmark = True` when the SeedVR2
  runner initializes (and optionally restore it afterwards). With tiling
  enabled, shape count is bounded, so the per-shape autotune cost is paid
  once. Expected: measurable VAE encode/decode speedup, zero quality impact.

### 2.2 Uniform 4-frame temporal slices in the VAE

- `attn_video_vae.py:1257-1266`: the first temporal slice is
  `cat(first_frame, slice)` = 5 frames, subsequent slices are 4 frames
  (`memory_state` INITIALIZING vs ACTIVE). With `dynamic=False` this creates
  two shape families per spatial shape — doubling compile variants and peak
  shape bookkeeping.
- Normalizing to uniform 4-frame slices would halve compile variants and
  first-run compile time.
- Risk: touches causal-conv boundary conditions; needs visual validation of
  first-batch output before adoption.

### 2.3 `channels_last` for the VAE — low priority

- Could help conv tensor-core utilization in fp16, but `cudnn.benchmark`
  (2.1) captures most of the same win without layout risk. Skip unless
  profiling shows conv-bound VAE time after 2.1.

---

## 3. Ruled out / already in place

- **cudagraphs (`reduce-overhead`, `max-autotune`)**: abandoned — private
  graph pools permanently reserve ~2x eager peak per shape variant and cannot
  be reclaimed after the fact (see `generation_phases.py:509-516`,
  `src/optimization/memory_manager.py:581-583`).
- **VAE attention backend switching**: the VAE only has mid-block attention
  at the lowest spatial resolution (`attn_video_vae.py:710-790`,
  `mid_block_add_attention=True`); cost is negligible. Not worth it.
- **TF32 flags**: set in `init_torch` but unreachable from ComfyUI; the VAE
  runs fp16 anyway, so fp32-matmul TF32 is minor here.
- **init_torch** generally: distributed-training leftover; do not call it
  from the ComfyUI path (it would attempt `dist.init_process_group`).

---

## 4. Recommended combination (16 GB, sm_120)

| Lever | Where | Expected effect |
|---|---|---|
| `attention_mode = sageattn_2` | DiT loader node | Large DiT speedup |
| `encode_tiled = True`, `decode_tiled = True` | VAE loader node | Removes VAE activation peak |
| NVFP4 DiT checkpoint | DiT loader node | ~4x weight VRAM cut + native matmul |
| `mode = max-autotune-no-cudagraphs` | Compile settings node | Autotune w/o graph pools (current) |
| `uniform_batch_size = True` | Upscaler node | Bounded compile variants |
| `cudnn.benchmark = True` | Code edit (2.1) | Free VAE conv speedup |

Optional once the above is stable: raise `batch_size` to spend the freed
VRAM on throughput; consider 2.2 only if first-run compile time matters.

---

## 5. Validation checklist (user-run)

1. SageAttention: run one batch with `sageattn_2`, compare output visually
   against `sdpa` (near-lossless expected) and check the startup line
   `SageAttn: v2 ✓` in the debug banner.
2. Tiling: enable encode+decode tiling, watch for tile seams (raise overlap
   if visible); confirm VRAM peak drop in the phase memory report.
3. cudnn.benchmark edit: time VAE encode/decode phase before/after on an
   identical clip and seed.
4. Any 2.2 experiment: validate first-batch frames specifically (causal
   boundary), then full-clip spot check.
