# SeedVR2 — Windows 并行 Inductor 编译修复

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../md/SEEDVR2_WINDOWS_PARALLEL_COMPILE_FIX.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

目标自定义节点：`ComfyUI/custom_nodes/seedvr2_videoupscaler`
代码提交：`a14db91b31c08bee62055e17521d4f1537bef03c`
（`feat: NVFP4 DiT native ops and VAE torch.compile inductor fixes`）

环境：Windows 11（日文区域，cp932）、`python_embeded` Python 3.13.13、torch 2.13.0+cu132、24 逻辑 CPU、RTX 50 系（sm_120，Blackwell）16 GB VRAM、WDDM 驱动模型。

相关指南：`md/SEEDVR2_SPEED_VRAM_HEADROOM.md`、
`md/SEEDVR2_NVFP4_AND_TORCH_COMPILE_GUIDE.md`
（中文版见同名文件于 `zhmd/`）。

---

## 1. 问题

### 1.1 症状

在冷 inductor 缓存下，pre-decode 的 `torch.compile` 花费 **约 9 分钟**
（控制台：decode batch 1 = 9m28s，batch 2–3 各约 70s）。这 9 分钟内 VRAM 接近零——GPU 空闲，而单条 CPU 线程串行编译每个 Triton kernel。

一阶段的全部编译都发生在该阶段的**首批**内：瓦片 VAE decoder 在 batch 1 编译每种 tile-shape 族（及其 INITIALIZING/ACTIVE 变体），之后 batch 2+ 纯走缓存。因此墙钟成本是一次性的、按冷缓存计的、CPU 受限的串行编译。

### 1.2 根因 A — PyTorch 在 win32 上硬编码串行编译

`torch/_inductor/config.py:1340-1369`（`decide_compile_threads`）：

```python
def decide_compile_threads() -> int:
    # precedence:
    #   1. TORCHINDUCTOR_COMPILE_THREADS env var
    #   2. win32   -> 1          # <── Windows is forced to serial
    #   3. fbcode  -> 1
    #   4. default -> min(32, cpu_count)
```

在 Linux 上本机可得 `min(32, 24) = 24` 个编译线程；在 Windows 上仅为 **1**，纯粹因为操作系统。

### 1.3 根因 B — 默认 worker 池在 Windows 上*损坏*

仅提高 `compile_threads` 不够。默认 worker 方法 `"subprocess"`（`config.py:1113-1126`，`decide_worker_start_method`）会启动 **SubprocPool sidecar**：独立的 `python -m torch._inductor.compile_worker` 进程，经管道投递任务（`torch/_inductor/compile_worker/subproc_pool.py`）。

`SubprocPool.__init__` 默认多进程种类为 `SubprocKind.FORK`，且 `SubprocMain._start_pool()` 调用：

```python
mp_context=multiprocessing.get_context(self.kind.value)  # "fork"
```

Windows **没有 `fork` 启动方法**——sidecar 内抛出 `ValueError`，sidecar 在发出就绪信号前死亡，父进程的 `wait_pool_ready()` 阻塞直到就绪超时（实测：120 s `TimeoutError`）。文件内无任何 Windows 专用处理（已 grep 验证）。在官方默认下，Windows 上的并行编译不仅被禁用——并行路径注定死亡，这也大概是 PyTorch 在此强制 `compile_threads = 1` 的原因。

### 1.4 可行路径 — `worker_start_method="spawn"`

`"spawn"` 在父进程内使用带 `multiprocessing` *spawn* 上下文的 `TrackedProcessPoolExecutor`，Windows **确实有** spawn。本机实测：池就绪约 **3.2 s**，编译正常完成。

Spawn 会在每个 worker 中重新执行主模块，因此入口必须受 `if __name__ == "__main__":` 保护。ComfyUI 的 `main.py` 已有保护（约 36、42、67、550 行），故 ComfyUI 对 spawn 安全。

### 1.5 修复引入的新风险 — worker 持有 CUDA 上下文

Autotune 基准（部分）在 worker 内运行，因此 spawn worker 会创建 CUDA 上下文。并行跑次实测：9 个 `python.exe`（1 父 + 8 worker），其中 **4 个挂 GPU**。空闲 worker 会在剩余批次与下一阶段继续占用 VRAM（合计约 1–2 GB）——对 16 GB 卡在 DiT 采样时不可接受。

缓解：在每阶段首批之后（可证明所有编译发生于此——见 1.1）调用一次 `torch._inductor.async_compile.shutdown_compile_workers()`。若之后还有编译工作，池会惰性重建，对后续阶段安全。

### 1.6 为何不会使现有缓存失效

两个旋钮都在 `_cache_config_ignore_prefix` 中
（`torch/_inductor/config.py:2853-2854`）：`"worker_start_method"` 与 `"compile_threads"` 明确排除在 FX-graph/autotune 缓存键之外（「与缓存结果无关」——只影响*调度*）。因此启用并行编译后，所有既有暖缓存仍然有效。

---

## 2. 变更文件

| 文件 | 变更 |
|---|---|
| `src/core/fix_inductor.py` | **+36 行**：新增 `_patch_inductor_parallel_compile_windows()`，由 `_fix_inductor_windows_encoding()` 调用（经 `__init__.py` 在自定义节点导入时运行） |
| `src/core/generation_phases.py` | **+23 行**：新增 `_shutdown_inductor_compile_workers()` 辅助函数，以及 **3 处钩子调用点**（每阶段各一，在首批 forward 之后） |

未改其他文件。未增依赖。两文件均通过 `python -m py_compile`。

---

## 3. 新增代码全文

### 3.1 `src/core/fix_inductor.py` — 新函数

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

### 3.2 `src/core/fix_inductor.py` — 调用点（在 `_fix_inductor_windows_encoding()` 内）

```python
    # --- (0b) bmm make_fallback override (VAE compile assertion) ---
    _patch_inductor_bmm_make_fallback_override()

    # --- (0c) parallel inductor compile (win32 default is serial + broken sidecar) ---
    _patch_inductor_parallel_compile_windows()

    # --- (1) inductor cpp_builder ---
```

### 3.3 `src/core/generation_phases.py` — 新辅助函数（模块级）

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

### 3.4 钩子 1 — encode 阶段（`encode_all_batches`，在 `runner.vae_encode` 之后）

```python
            # Encode to latents
            cond_latents = runner.vae_encode([transformed_video])

            # First batch carries all torch.compile work for this phase; free
            # parallel-compile workers (they hold CUDA contexts) afterwards.
            if encode_idx == 0:
                _shutdown_inductor_compile_workers()
```

### 3.5 钩子 2 — upscale 阶段（`upscale_all_batches`，在 DiT 推理之后）

```python
            debug.end_timer(f"dit_inference_{upscale_idx+1}", f"DiT inference {upscale_idx+1}")

            # First batch carries all torch.compile work for this phase; free
            # parallel-compile workers (they hold CUDA contexts) afterwards.
            if batch_idx == 0:
                _shutdown_inductor_compile_workers()
```

### 3.6 钩子 3 — decode 阶段（`decode_all_batches`，在 `runner.vae_decode` 之后）

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

## 4. 代码含义

### 4.1 `_patch_inductor_parallel_compile_windows()`，逐行

- `if os.name != "nt": return` — 损坏的默认值是 Windows 特有；Linux 已有 `min(32, cpu_count)` 线程与可用池。其他平台不做。
- `import torch._inductor.config as inductor_config` — 函数内惰性导入，避免 `fix_inductor` 模块导入时为 inductor 付费。
- 第一个 `if` — **仅当**用户未设 `TORCHINDUCTOR_COMPILE_THREADS` 且当前值为 `None`（fbcode 惰性初始化）或 `1`（win32 强制默认——官方 torch 在导入时经 `decide_compile_threads()` 急切赋 `1`，config.py:1373，**不是** `None`，因此检查必须是 `in (None, 1)` 而非 `is None`）时，设 `compile_threads = min(8, cpu_count)`。上限 8：超过约 8 个 Triton 进程收益递减，且每个 worker 占 RAM/VRAM。
  - 目标机验证：官方状态 `1 / subprocess` → 补丁后 `8 / spawn`；设 `TORCHINDUCTOR_COMPILE_THREADS=1` 时保持 `1`（退出开关生效）。
- 第二个 `if` — 在用户未设 `TORCHINDUCTOR_WORKER_START` 时，将 `worker_start_method` 从损坏的 `"subprocess"` 默认改为 `"spawn"`。
- 整体包在 `try/except` — 未来 torch 若改内部结构，降级为打印警告、永不崩溃；自定义节点仍可用官方串行编译。

### 4.2 `_shutdown_inductor_compile_workers()` 与三处钩子

- 辅助函数惰性导入 `shutdown_compile_workers` 并吞掉所有异常：无此 API 的 torch 构建、或池不存在时静默 no-op。从不干扰串行编译配置。
- 每阶段钩子恰好触发一次，由循环首轮守卫（`encode_idx == 0` / `batch_idx == 0`），放在**该阶段首次 forward 调用之后**——该阶段所有编译刚结束的点（1.1）。worker 进程退出，在剩余批次与下一阶段模型驻留前释放 CUDA 上下文与主机 RAM。
- 若后续阶段触发真正新的编译（新 shape 族），inductor 会惰性重建池；shutdown 不影响正确性。
- Decode 注释略长，因其最重：瓦片解码在 batch 1 编译每种 tile-shape 族。

### 4.3 实测效果（如实数字）

基准：8 个偏 conv 的图（各含 3× conv2d + SiLU + GroupNorm，不同 shape），`mode="max-autotune-no-cudagraphs"`，`dynamic=False`，全部禁用缓存：

| 配置 | 总墙钟时间 |
|---|---|
| 官方 win32 默认（串行） | **134.8 s** |
| 本修复（8 个 spawn worker） | **111.5 s**（1.21×） |

另一套极小 elementwise 图**无**收益（12.6 s vs 12.8 s）：小图由父侧串行 lowering 主导，而非可并行的 Triton 编译。

真实 pre-decode 编译介于两者之间，但更接近 conv 情形：每个 VAE decode 图族含数十个 Triton kernel，图内 autotune 并行应高于 1.21×。**必须在下次真正冷跑次上验证**——玩具数字只是机制可用的下界证明，不是 9 分钟 → X 分钟的承诺。

### 4.4 退出 / 覆盖

| 环境变量 | 效果 |
|---|---|
| `TORCHINDUCTOR_COMPILE_THREADS=1` | 完全恢复官方串行（补丁让步） |
| `TORCHINDUCTOR_COMPILE_THREADS=N` | 使用 N 个 worker，而非 `min(8, cpu_count)` |
| `TORCHINDUCTOR_WORKER_START=...` | 覆盖 spawn 选择（例如 PyTorch 修好 win32 sidecar 后再改回 `subprocess`） |

### 4.5 缓存持久性（何时会再次出现 9 分钟编译）

FX graph 缓存与 autotune 本地缓存在磁盘上，**重启后仍在**。仅在以下情况失效：

1. 被追踪的源码被编辑（Dynamo 追踪到的 `src/` 下任何变更），
2. torch 或 triton 版本变更，
3. 编译模式 / 影响编译的设置变更，
4. 在 `dynamic=False` 下出现**新输入 shape**（仅该 shape 重编译；既有条目仍暖），
5. 缓存目录被删除或淘汰。

`compile_threads` 与 `worker_start_method` 免缓存键（1.6），故本修复本身永不触发 (1)–(3)。相同代码 + 相同设置 + 相同 shape ⇒ 下次为暖缓存，9 分钟编译不会复发。
