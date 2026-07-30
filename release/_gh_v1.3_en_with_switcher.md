<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/blob/main/zhmd/v1.3.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

**Tag:** `v1.3`  
**Commit:** `0628cb0` (performance work in `5f2a472`)  
**Date:** 2026-07-28

This release focuses on **torch.compile / inductor runtime improvements** for the
ComfyUI SeedVR2 path on Windows. It enables **parallel inductor compile on
win32**, shuts compile workers down after each phase's first batch so they do
not keep holding CUDA contexts, enables **`cudnn.benchmark`** for the duration
of a run, and makes the VAE temporal slices more uniform to reduce compile
shape variants.

**Guides (full write-up):**  
- https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/blob/main/md/SEEDVR2_SPEED_VRAM_HEADROOM.md  
- https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/blob/main/md/SEEDVR2_WINDOWS_PARALLEL_COMPILE_FIX.md

---

## 1. Summary

| Topic | Status |
|-------|--------|
| **Windows compile bottleneck** | Stock torch on `win32` forces `compile_threads = 1`, so cold-cache inductor/Triton compile runs serially and can leave the GPU idle for minutes |
| **Parallel compile fix** | `fix_inductor.py` now switches Windows to `worker_start_method = "spawn"` and `compile_threads = min(8, cpu_count)` unless env overrides are set |
| **Worker VRAM retention** | Spawn workers can hold CUDA contexts after autotune benchmarking |
| **Worker cleanup fix** | `generation_phases.py` now calls `shutdown_compile_workers()` after each phase's first batch, where that phase's compile work is complete |
| **VAE conv speed** | `video_upscaler.py` enables `torch.backends.cudnn.benchmark = True` only for the active run, then restores the caller's previous setting |
| **Compile shape stability** | `attn_video_vae.py` now processes the first frame/latent alone and the rest in uniform ACTIVE chunks, reducing first/rest shape-family splits |

---

## 2. What changed

### 2.1 Parallel inductor compile on Windows

- New `_patch_inductor_parallel_compile_windows()` in
  `src/core/fix_inductor.py`
- Enables parallel compile on Windows without changing Linux behavior
- Respects:
  - `TORCHINDUCTOR_COMPILE_THREADS`
  - `TORCHINDUCTOR_WORKER_START`
- Applied from `_fix_inductor_windows_encoding()` during custom-node import

### 2.2 Release compile workers after the first batch

- New `_shutdown_inductor_compile_workers()` helper in
  `src/core/generation_phases.py`
- Called once per phase after the first batch:
  - encode
  - DiT upscale
  - decode
- Purpose: free CUDA contexts and VRAM held by idle compile workers before the
  remaining batches and the next phase

### 2.3 Enable cuDNN autotuner for the run

- `src/interfaces/video_upscaler.py` now:
  1. saves the current `torch.backends.cudnn.benchmark`
  2. sets it to `True` for the run
  3. restores the previous value in `finally`
- This gives the ComfyUI path the same practical VAE conv autotune benefit
  without requiring a global ComfyUI startup flag

### 2.4 Uniform temporal slices in the VAE

- `src/models/video_vae_v3/modules/attn_video_vae.py`
- The first frame/latent is processed alone with `INITIALIZING`
- Remaining chunks are processed as uniform `ACTIVE` chunks
- This reduces compile shape fragmentation versus the old first/rest split

---

## 3. Code surface in this tag

| Path | Role |
|------|------|
| `src/core/fix_inductor.py` | Parallel inductor compile defaults for Windows |
| `src/core/generation_phases.py` | Compile-worker shutdown hooks after phase-first-batch compile completes |
| `src/interfaces/video_upscaler.py` | Run-scoped `cudnn.benchmark = True` |
| `src/models/video_vae_v3/modules/attn_video_vae.py` | Uniform VAE temporal slices for compile stability |
| `md/SEEDVR2_SPEED_VRAM_HEADROOM.md` | Broader speed / VRAM guidance and implemented levers |
| `md/SEEDVR2_WINDOWS_PARALLEL_COMPILE_FIX.md` | Detailed root cause, implementation, and validation notes |

---

## 4. Expected effect

- Faster cold-cache compile on Windows compared with the stock serial path
- Less VRAM held by idle compile workers after the first batch of each phase
- Better VAE conv performance during normal runs
- Fewer avoidable compile variants from uneven first/rest temporal slicing

This release improves the **runtime behavior around compile**, not the model
weights themselves.

---

## 5. Scope of this GitHub Release

Tags commit **`0628cb0`** on **ComfyUI-SeedVR2_VideoUpscaler**.

Includes:
- Windows parallel inductor compile defaults
- post-first-batch compile-worker shutdown
- run-scoped `cudnn.benchmark`
- uniform VAE temporal slices
- documentation for the compile/speed changes

Does not add binary release assets.
