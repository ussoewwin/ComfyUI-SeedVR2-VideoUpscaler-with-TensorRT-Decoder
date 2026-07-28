# SeedVR2 — Windows Parallel Inductor Compile Fix

Target custom node: `ComfyUI/custom_nodes/seedvr2_videoupscaler`
Code commit: `a14db91b31c08bee62055e17521d4f1537bef03c`
(`feat: NVFP4 DiT native ops and VAE torch.compile inductor fixes`)

Environment: Windows 11 (Japanese locale, cp932), `python_embeded` Python
3.13.13, torch 2.13.0+cu132, 24 logical CPUs, RTX 50-series (sm_120,
Blackwell) 16 GB VRAM, WDDM driver model.

Related guides: `md/SEEDVR2_SPEED_VRAM_HEADROOM.md`,
`md/SEEDVR2_NVFP4_AND_TORCH_COMPILE_GUIDE.md`.

---

## 1. The problem

### 1.1 Symptom

On a cold inductor cache, the pre-decode `torch.compile` took **~9 minutes**
(console: decode batch 1 = 9m28s, batches 2–3 ≈ 70s each). During those 9
minutes VRAM usage was near zero — the GPU sat idle while a single CPU thread
compiled every Triton kernel serially.

All compilation for a phase happens inside that phase's **first batch**: the
tiled VAE decoder compiles every tile-shape family (and its
INITIALIZING/ACTIVE variants) during batch 1, then batches 2+ run purely from
cache. So the wall-clock cost is a one-time, per-cold-cache CPU-bound serial
compile.

### 1.2 Root cause A — PyTorch hard-codes serial compile on win32

`torch/_inductor/config.py:1340-1369` (`decide_compile_threads`):

```python
def decide_compile_threads() -> int:
    # precedence:
    #   1. TORCHINDUCTOR_COMPILE_THREADS env var
    #   2. win32   -> 1          # <── Windows is forced to serial
    #   3. fbcode  -> 1
    #   4. default -> min(32, cpu_count)
```

On Linux this machine would get `min(32, 24) = 24` compile threads. On
Windows it gets **1**, purely because of the OS.

### 1.3 Root cause B — the default worker pool is *broken* on Windows

Raising `compile_threads` alone is not enough. The default worker method
`"subprocess"` (`config.py:1113-1126`, `decide_worker_start_method`) launches
a **SubprocPool sidecar**: a separate `python -m
torch._inductor.compile_worker` process fed jobs over a pipe
(`torch/_inductor/compile_worker/subproc_pool.py`).

`SubprocPool.__init__` defaults its multiprocssing kind to
`SubprocKind.FORK`, and `SubprocMain._start_pool()` calls:

```python
mp_context=multiprocessing.get_context(self.kind.value)  # "fork"
```

Windows has **no `fork` start method** — this raises `ValueError` inside the
sidecar, the sidecar dies before signalling readiness, and the parent's
`wait_pool_ready()` blocks until its ready-timeout (measured: 120 s
`TimeoutError`). The file contains zero Windows-specific handling
(grep-verified). With the stock defaults, parallel compile on Windows is not
merely disabled — the parallel path is guaranteed-dead, which is presumably
why PyTorch forces `compile_threads = 1` there.

### 1.4 The working path — `worker_start_method="spawn"`

`"spawn"` uses an in-parent `TrackedProcessPoolExecutor` with the
`multiprocessing` *spawn* context, which **does** exist on Windows. Measured
on this machine: pool ready in **3.2 s**, compiles complete cleanly.

Spawn re-executes the main module in each worker, so the entry point must be
protected by `if __name__ == "__main__":`. ComfyUI's `main.py` is guarded
(lines 36, 42, 67, 550), so ComfyUI is spawn-safe.

### 1.5 New risk introduced by the fix — workers hold CUDA contexts

Autotune benchmarking runs (partially) inside the workers, so spawn workers
create CUDA contexts. Measured during a parallel run: 9 `python.exe`
processes (1 parent + 8 workers), of which **4 were GPU-attached**. Idle
workers would therefore keep holding VRAM (~1–2 GB total) through the
remaining batches and into the next phase — unacceptable on a 16 GB card
during DiT sampling.

Mitigation: call `torch._inductor.async_compile.shutdown_compile_workers()`
once, right after each phase's first batch (where all compilation provably
happens — see 1.1). The pool is re-created lazily if more compile work
appears later, so this is safe for subsequent phases.

### 1.6 Why this does not invalidate existing caches

Both knobs are in `_cache_config_ignore_prefix`
(`torch/_inductor/config.py:2853-2854`): `"worker_start_method"` and
`"compile_threads"` are explicitly excluded from the FX-graph/autotune cache
key ("not relevant" to cache results — they only affect *scheduling*).
Enabling parallel compile therefore keeps every existing warm cache valid.

---

## 2. Files changed

| File | Change |
|---|---|
| `src/core/fix_inductor.py` | **+36 lines**: new `_patch_inductor_parallel_compile_windows()`, invoked from `_fix_inductor_windows_encoding()` (runs at custom-node import via `__init__.py`) |
| `src/core/generation_phases.py` | **+23 lines**: new `_shutdown_inductor_compile_workers()` helper, plus **3 hook call sites** (one per phase, after the first batch's forward call) |

No other files touched. No dependencies added. Both files pass
`python -m py_compile`.

---

## 3. Full added code

### 3.1 `src/core/fix_inductor.py` — new function

```python
def _patch_inductor_parallel_compile_windows() -> None:
    """
    Enable parallel inductor/Triton compilation on Windows.

    Stock torch hard-codes compile_threads=1 on win32, and its default
    worker_start_method="subprocess" (SubprocPool sidecar) is broken on
    Windows: the sidecar calls multiprocessing.get_context("fork"), which
    does not exist on win32, so the pool never becomes ready and the first
    compile stalls until the ready-timeout.

    worker_start_method="spawn" works on Windows (verified: pool ready in
    seconds, compiles complete). ComfyUI main.py is spawn-safe (guarded by
    `if __name__ == "__main__"`).

    Env overrides are respected: TORCHINDUCTOR_COMPILE_THREADS and
    TORCHINDUCTOR_WORKER_START always win over these defaults.
    """
    if os.name != "nt":
        return
    try:
        import torch._inductor.config as inductor_config

        # win32 eagerly assigns compile_threads=1 at import (config.py:1373),
        # not None — so treat both None (fbcode lazy init) and 1 (win32 forced
        # default) as "not user-chosen". TORCHINDUCTOR_COMPILE_THREADS is the
        # respected opt-out.
        if ("TORCHINDUCTOR_COMPILE_THREADS" not in os.environ
                and getattr(inductor_config, "compile_threads", None) in (None, 1)):
            inductor_config.compile_threads = min(8, os.cpu_count() or 1)
            print(f"[SeedVR2] Enabled parallel inductor compile: {inductor_config.compile_threads} threads")
        if ("TORCHINDUCTOR_WORKER_START" not in os.environ
                and getattr(inductor_config, "worker_start_method", None) == "subprocess"):
            inductor_config.worker_start_method = "spawn"
    except Exception as e:
        print(f"[SeedVR2] Warning: Could not enable parallel inductor compile: {e}")
```

### 3.2 `src/core/fix_inductor.py` — call site (inside `_fix_inductor_windows_encoding()`)

```python
    # --- (0b) bmm make_fallback override (VAE compile assertion) ---
    _patch_inductor_bmm_make_fallback_override()

    # --- (0c) parallel inductor compile (win32 default is serial + broken sidecar) ---
    _patch_inductor_parallel_compile_windows()

    # --- (1) inductor cpp_builder ---
```

### 3.3 `src/core/generation_phases.py` — new helper (module level)

```python
def _shutdown_inductor_compile_workers() -> None:
    """
    Shut down inductor parallel-compile workers after a phase's first batch.

    All torch.compile work for a phase happens during its first batch (every
    shape/graph family is compiled there; later batches hit the cache). With
    worker_start_method="spawn" the workers partially create CUDA contexts
    during autotune benchmarking, so idle workers would keep holding VRAM
    through the remaining batches and the next phase. Shutting the pool down
    releases that memory; it is re-created lazily if more compile work
    appears. No-op when parallel compile is disabled or torch lacks the API.
    """
    try:
        from torch._inductor.async_compile import shutdown_compile_workers
        shutdown_compile_workers()
    except Exception:
        pass
```

### 3.4 Hook 1 — encode phase (`encode_all_batches`, after `runner.vae_encode`)

```python
            # Encode to latents
            cond_latents = runner.vae_encode([transformed_video])

            # First batch carries all torch.compile work for this phase; free
            # parallel-compile workers (they hold CUDA contexts) afterwards.
            if encode_idx == 0:
                _shutdown_inductor_compile_workers()
```

### 3.5 Hook 2 — upscale phase (`upscale_all_batches`, after DiT inference)

```python
            debug.end_timer(f"dit_inference_{upscale_idx+1}", f"DiT inference {upscale_idx+1}")

            # First batch carries all torch.compile work for this phase; free
            # parallel-compile workers (they hold CUDA contexts) afterwards.
            if batch_idx == 0:
                _shutdown_inductor_compile_workers()
```

### 3.6 Hook 3 — decode phase (`decode_all_batches`, after `runner.vae_decode`)

```python
            # Decode latent
            debug.start_timer("vae_decode")
            samples = runner.vae_decode([upscaled_latent])
            debug.end_timer("vae_decode", "VAE decode")

            # First batch carries all torch.compile work for this phase (every
            # tile/graph family compiles here); free parallel-compile workers
            # (they hold CUDA contexts) before the remaining batches.
            if batch_idx == 0:
                _shutdown_inductor_compile_workers()
```

---

## 4. What the code means

### 4.1 `_patch_inductor_parallel_compile_windows()`, line by line

- `if os.name != "nt": return` — the broken defaults are Windows-specific;
  Linux already gets `min(32, cpu_count)` threads with a working pool. Do
  nothing elsewhere.
- `import torch._inductor.config as inductor_config` — imported lazily inside
  the function so module import of `fix_inductor` never pays for inductor
  import.
- First `if` — sets `compile_threads = min(8, cpu_count)` **only if** the
  user did not set `TORCHINDUCTOR_COMPILE_THREADS` and the current value is
  `None` (fbcode lazy-init) or `1` (the win32 forced default — stock torch
  assigns `1` eagerly at import via `decide_compile_threads()`,
  config.py:1373, **not** `None`, which is why the check must be
  `in (None, 1)` rather than `is None`). Capped at 8: beyond ~8 Triton
  processes the returns diminish and each worker costs RAM/VRAM.
  - Verified on the target machine: stock state `1 / subprocess` → after the
    patch `8 / spawn`; with `TORCHINDUCTOR_COMPILE_THREADS=1` set, the value
    stays `1` (opt-out respected).
- Second `if` — switches `worker_start_method` from the broken `"subprocess"`
  default to `"spawn"`, again only when the user has not set
  `TORCHINDUCTOR_WORKER_START`.
- Whole body wrapped in `try/except` — any inductor internals change in a
  future torch version degrades to a printed warning, never a crash; the
  custom node keeps working with stock serial compile.

### 4.2 `_shutdown_inductor_compile_workers()` and the three hooks

- The helper imports `shutdown_compile_workers` lazily and swallows all
  exceptions: on torch builds without the API, or when no pool exists, it is
  a silent no-op. It never disturbs serial-compile setups.
- Each hook fires exactly once per phase, guarded by the loop's first
  iteration (`encode_idx == 0` / `batch_idx == 0`), placed **immediately
  after that phase's first forward call** — the point where every compile for
  the phase has just finished (1.1). Worker processes exit, releasing their
  CUDA contexts and host RAM before the remaining batches and before the next
  phase's model is resident.
- If a later phase triggers genuinely new compilation (new shape family), the
  pool is lazily re-created by inductor; shutdown costs nothing correctness-
  wise.
- Decode gets a slightly longer comment because it is the heaviest case:
  tiled decoding compiles every tile-shape family in batch 1.

### 4.3 Measured effect (honest numbers)

Benchmark harness: 8 conv-heavy graphs (3× conv2d + SiLU + GroupNorm each,
distinct shapes), `mode="max-autotune-no-cudagraphs"`, `dynamic=False`, all
caches disabled:

| Configuration | Total wall time |
|---|---|
| Stock win32 defaults (serial) | **134.8 s** |
| This fix (8 spawn workers) | **111.5 s** (1.21×) |

A second harness with tiny elementwise graphs showed **no** gain (12.6 s vs
12.8 s): small graphs are dominated by serial parent-side lowering, not
parallelizable Triton compilation.

The real pre-decode compile sits between these extremes but leans toward the
conv case: each VAE decode graph family contains dozens of Triton kernels, so
intra-graph autotune parallelism should pay more than 1.21×. **This must be
validated on the next genuine cold run** — the toy numbers are a lower-bound
proof that the machinery works, not a promise of 9 min → X min.

### 4.4 Opt-out / overrides

| Env var | Effect |
|---|---|
| `TORCHINDUCTOR_COMPILE_THREADS=1` | Fully restores stock serial behavior (fix stands down) |
| `TORCHINDUCTOR_COMPILE_THREADS=N` | Uses N workers instead of `min(8, cpu_count)` |
| `TORCHINDUCTOR_WORKER_START=...` | Overrides the spawn choice (e.g. back to `subprocess` once PyTorch fixes the sidecar on win32) |

### 4.5 Cache persistence (when the 9-minute compile comes back)

The FX graph cache and autotune local cache live on disk and **survive
restarts**. They are invalidated only when:

1. traced source code is edited (any change under `src/` that Dynamo traces),
2. torch or triton version changes,
3. compile mode / compile-affecting settings change,
4. a **new input shape** appears under `dynamic=False` (only that shape
   recompiles; existing entries stay warm),
5. the cache directory is deleted or evicted.

`compile_threads` and `worker_start_method` are cache-key-exempt (1.6), so
this fix itself never triggers (1)–(3). Same code + same settings + same
shapes ⇒ the next run is warm and the 9-minute compile does not recur.
