# SeedVR2 Video Upscaler — 超越 max-autotune 的速度 / 显存余量

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../md/SEEDVR2_SPEED_VRAM_HEADROOM.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

目标自定义节点：`ComfyUI/custom_nodes/seedvr2_videoupscaler`
代码提交：`a14db91b31c08bee62055e17521d4f1537bef03c`
（`feat: NVFP4 DiT native ops and VAE torch.compile inductor fixes`）

本指南梳理在放弃基于 cudagraphs 的 `max-autotune` 之后，仍可用的 **速度** 与 **显存（VRAM）** 杠杆。放弃 max-autotune 实验的原因是：首次 VAE encode 就会冲破 16 GB——CUDA Graph 捕获会在私有池中永久预留约 2 倍于 eager 峰值的显存（每种 shape 变体一份），而 `reclaim_transient_vram_after_torch_compile` 只在峰值**之后**触发，无法回收仍存活的 graph 池。因此生产模式为 `max-autotune-no-cudagraphs`。

假定环境：RTX 50 系（sm_120，Blackwell）、16 GB VRAM、WDDM、已安装 `sageattention`（**未**安装 `sageattn3` 包）、`torch.compile` 已启用且关闭 cudagraphs。

相关指南：`md/SEEDVR2_NVFP4_AND_TORCH_COMPILE_GUIDE.md`、
`md/SEEDVR2_INT8_NATIVE_OPS_GUIDE.md`、
`md/SEEDVR2_WINDOWS_PARALLEL_COMPILE_FIX.md`
（中文版见同名文件于 `zhmd/`）。

---

## 1. 仅改设置的杠杆（无需改代码）

### 1.1 DiT `attention_mode: sageattn_2` — 最大速度杠杆

- 节点：`SeedVR2LoadDiTModel`，`attention_mode` 下拉
  （`src/interfaces/dit_model_loader.py:104-118`）。默认 `sdpa`。
- `sageattention` 已在 `python_embeded` 中安装；在 sm_120 上走 FP8-PV per-warp 路径。视频 latent 的 attention 序列很长，attention 主导 DiT 算力——预期加速明显。
- `sageattn_3`（标注 Blackwell 的选项）需要单独的 `sageattn3` 包，当前未安装。请用 `sageattn_2`。
- 与 `torch.compile` 共存：SageAttention 是自定义 CUDA kernel，Dynamo 会在其周围插入 graph break。在 `fullgraph=False`（默认）下无害。
- `flash_attn_3` 仅 Hopper；不适用于 sm_120。

### 1.2 VAE `encode_tiled=True` / `decode_tiled=True` — 最大显存杠杆

- 节点：`SeedVR2LoadVAEModel`（`src/interfaces/vae_model_loader.py:58-115`）。
  两者默认均为 **False**；schema 瓦片尺寸 1024 / overlap 128。
- 直接消除 max-autotune 事后分析中指出的首次 encode 激活峰值（全分辨率 conv 激活按整帧缩放；瓦片路径按 ~(tile/H)×(tile/W) 缩放）。
- 瓦片路径内仍保留时间切片
  （`src/models/video_vae_v3/modules/attn_video_vae.py:1403`，`slicing_encode`）。
- 代价：重叠区会算两次并混合——encode/decode 略慢，但总耗时由 DiT 主导，端到端影响较小。边缘瓦片会多几种 shape；关闭 cudagraphs 时只多一点编译时间，不会永久占池。

### 1.3 `uniform_batch_size=True` — 更少 shape 变体

- 节点：主放大节点（`src/interfaces/video_upscaler.py:125`）。
- 将末批 pad 到 `batch_size`，使已编译 shape 数量有界（更少重编译、更少 inductor 变体）。代价是末尾 pad 帧上少量浪费算力。

### 1.4 上调 `batch_size`（遵守 4n+1 约束）

- 量化 DiT + 瓦片 VAE 腾出显存后，提高 `batch_size` 可提升 DiT 吞吐（更少采样启动、更好利用率）。
- 注意 `temporal_overlap` 帧不会推进片段——每批有效新帧 = `batch_size - temporal_overlap`。

### 1.5 `cache_model=True` + `offload_device=cpu`

- 工作流运行之间把 DiT/VAE 留在系统内存，批量处理时省去重载时间。占 RAM，不占 VRAM。

### 1.6 `blocks_to_swap` / `swap_io_components` — 仅适用于 fp16/int8 DiT

- 节点：`SeedVR2LoadDiTModel`（`dit_model_loader.py:57-81`）。
- 在跑 fp16（约 14 GB）或 int8 权重时有意义。NVFP4（约 3.5 GB）通常不需要；BlockSwap 会在采样时增加 PCIe 传输延迟，应视为应急阀而非默认项。

### 1.7 NVFP4 DiT 权重 — 若仍在用 fp16

- `models/SEEDVR2/seedvr2_7b_nvfp4.safetensors` 已与 fp16 / int8_convrot 变体并存。
- 原生路径（提交 a14db91）：构造期注入 `comfy.ops`，权重保持打包（约比 fp16 小 4 倍），并在 sm_120 上经 `comfy_kitchen.quantize_nvfp4` 做原生 NVFP4 matmul（`generation_phases.py:717-729`，`src/optimization/nvfp4_native_ops.py`）。同时获得速度与显存收益。

---

## 2. 代码级杠杆（需改代码）

### 2.1 在 ComfyUI 路径启用 `cudnn.benchmark` — 免费的 VAE 加速

- SeedVR2 仅在 `init_torch`（`src/common/distributed/basic.py:64-70`）里设置 `torch.backends.cudnn.benchmark = True`，而 ComfyUI 运行时路径**从不调用**它（只在 `src/common/distributed/__init__.py` 再导出）。对 ComfyUI 用户是死代码。
- ComfyUI 核心仅在 `--fast AutoTune` 下启用 benchmark（`comfy/model_management.py:540`），默认标志下每个 VAE conv 都用启发式算法选择。
- **已实现（2026-07-28，提交 `5f2a472`）：** `video_upscaler.py` 保存调用方原值，运行期间设 `torch.backends.cudnn.benchmark = True`，并在 `finally` 中恢复。开启 tiling 后 shape 数量有界，每种 shape 的 autotune 成本只付一次。预期：可测的 VAE encode/decode 加速，零画质影响。

### 2.2 VAE 均匀时间切片（已实现）

- 原先：第一个时间块是 `cat(first, slice0)`（INITIALIZING），其余切片均匀（ACTIVE）——例如 5 帧头块 + 4 帧块。在 `dynamic=False` 下每种空间 shape 多出一个 shape 族，编译变体翻倍。
- **已实现（2026-07-28，提交 `5f2a472`）：** `attn_video_vae.py` 现在单独 encode/decode 第一帧/latent（INITIALIZING），其余块均为均匀的 `slicing_sample_min_size` / `slicing_latent_min_size`（ACTIVE）——一个均匀 shape 族，而非 first/rest 成对。
- 正确性：对因果 k=3/s=2 时间卷积逐窗追踪——kernel−stride 缓存携带使 `[1, N, N, …]` 分块与 `[1+N, N, …]` 逐元素相同；总量一致（4n+1 帧 → 2n+1 latent）。与上游 diffusers CogVideoX VAE 约定相同。
- 剩余验证：在真实跑次上目视检查首批输出（因果边界）；数学上应相同，若有可见接缝应视为别处实现 bug，而非设计缺陷。

### 2.3 VAE 的 `channels_last` — 低优先级

- 可能有助于 fp16 下 conv tensor-core 利用率，但 `cudnn.benchmark`（2.1）已吃到大部分同等收益且无布局风险。除非 profiling 显示 2.1 之后 VAE 仍受 conv 束缚，否则跳过。

### 2.4 Windows 上并行 inductor 编译（已实现）

- 原先：在 win32 上，官方 torch 强制 `compile_threads = 1`，且默认 `worker_start_method = "subprocess"` 池损坏（sidecar 调用不存在于 Windows 的 `multiprocessing.get_context("fork")`），因此每次冷缓存编译都串行——pre-decode 编译约 9 分钟且 GPU 空闲。
- **已实现（2026-07-28，提交 `5f2a472`）：** `fix_inductor.py` 设 `compile_threads = min(8, cpu_count)` 与 `worker_start_method = "spawn"`（spawn 在 Windows 可用；ComfyUI `main.py` 有 `__main__` 保护），并尊重 `TORCHINDUCTOR_COMPILE_THREADS` / `TORCHINDUCTOR_WORKER_START` 作为退出开关。两旋钮均排除在 inductor 缓存键之外，暖缓存仍有效。
- Spawn worker 在 autotune 时会创建部分 CUDA 上下文，因此 `generation_phases.py` 在每阶段首批之后（可证明所有编译发生于此）调用一次 `shutdown_compile_workers()`，在剩余批次与下一阶段前释放其 VRAM。
- 完整说明：`md/SEEDVR2_WINDOWS_PARALLEL_COMPILE_FIX.md`（中文：`zhmd/SEEDVR2_WINDOWS_PARALLEL_COMPILE_FIX.md`）。

---

## 3. 已排除 / 已就位

- **cudagraphs（`reduce-overhead`、`max-autotune`）**：已放弃——私有 graph 池按每种 shape 变体永久预留约 2 倍 eager 峰值，事后无法回收（见 `generation_phases.py:509-516`，`src/optimization/memory_manager.py:581-583`）。
- **VAE attention 后端切换**：VAE 仅在最低空间分辨率有 mid-block attention（`attn_video_vae.py:710-790`，`mid_block_add_attention=True`）；成本可忽略。不值得。
- **TF32 标志**：在 `init_torch` 中设置但从 ComfyUI 不可达；VAE 本身跑 fp16，此处 fp32-matmul 的 TF32 影响很小。
- **一般不要从 ComfyUI 路径调用 init_torch**：分布式训练残留；调用会尝试 `dist.init_process_group`。

---

## 4. 推荐组合（16 GB，sm_120）

| 杠杆 | 位置 | 预期效果 |
|---|---|---|
| `attention_mode = sageattn_2` | DiT loader 节点 | 大幅 DiT 加速 |
| `encode_tiled = True`，`decode_tiled = True` | VAE loader 节点 | 消除 VAE 激活峰值 |
| NVFP4 DiT 权重 | DiT loader 节点 | 权重显存约减 4 倍 + 原生 matmul |
| `mode = max-autotune-no-cudagraphs` | Compile 设置节点 | 无 graph 池的 autotune（当前） |
| `uniform_batch_size = True` | Upscaler 节点 | 有界编译变体 |
| `cudnn.benchmark = True` | 已实现（2.1） | 免费 VAE conv 加速 |

上述稳定后可选：提高 `batch_size`，把腾出的显存换成吞吐。

---

## 5. 验证清单（用户跑次）

1. SageAttention：用 `sageattn_2` 跑一批，与 `sdpa` 目视对比（预期近无损），并确认调试横幅中有 `SageAttn: v2 ✓`。
2. Tiling：开启 encode+decode tiling，观察瓦片接缝（可见则提高 overlap）；在阶段显存报告中确认峰值下降。
3. cudnn.benchmark（已实现，2.1）：在相同片段与种子下对比 VAE encode/decode 阶段耗时。
4. 均匀时间切片（已实现，2.2）：重点验证首批帧（因果边界），再做全片抽检。
5. 并行编译（已实现，2.4）：下次冷缓存跑次确认出现 `[SeedVR2] Enabled parallel inductor compile` 横幅，且首批编译时间相对约 9 分钟串行基线下降。
