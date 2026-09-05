<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder/blob/main/md/changelog.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

# 更新日志

Fork 发行历史。

## v1.8.1 — 2026-09-05

- **摘要：** 安装程序与运行时稳定性全面改进：
  - **全自动零干预安装：** 修复 `install.py` 无法自动装全依赖的问题，无需手动运行批处理文件即可在目标 Python 环境中自动完成 `requirements.txt` 完整安装。
  - **统一注意力机制与 SDPA 标准：** 彻底废除 `install.py` 与 `scripts/install.ps1` 中脆弱的 FlashAttention 2 / SageAttention 2 外部 wheel 强制下载与安装逻辑；未安装自定义注意力加速库时，统一安全回退至 PyTorch 原生 SDPA（`attention_mode: sdpa`）（[#1](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder/issues/1)）。
  - **全链路 FFmpeg 路径自动解析：** 在节点初始化（`__init__.py`）、安装器（`install.py`）、环境验证（`scripts/verify_install.py`）及 CLI 中全面引入多候选路径扫描与 `imageio_ffmpeg` 自动兜底机制，彻底杜绝视频合成与导出时的 PATH 缺失异常。
  - **解码器引擎规范明示：** 在文档中明确规定构建 TensorRT VAE 解码器引擎（`kind: decoder`）时必须使用 `tile_size: 256`，彻底杜绝推理时的空间维度不匹配问题。
- **技术详情：** 请参阅 [v1.8.1 发行说明](v1.8.1.md) 获取完整说明

## v1.5 — 2026-09-03

- **摘要：** 新增 TensorRT VAE 解码器支持与专用加载节点（`SeedVR2LoadTensorRTVAEDecoder`），支持多 Tile 引擎（256px/512px，4n+1 帧长规格）、执行上下文缓存复用、编码/解码独立解耦配置，以及引擎缺失时自动安全回退至 PyTorch VAE。
- **技术详情：** 请参阅 [v1.5 发行说明](v1.5.md) 获取完整说明

## v1.4 — 2026-07-31

- **摘要：** 在 `MODEL_REGISTRY` 中登记 3B HSWQ INT8 ConvRot 与 NVFP4 DiT 权重包（与 7B 相同的原生显存路径）。
- **技术详情：** 请参阅 [v1.4 发行说明](v1.4.md) 获取完整说明

## v1.3 — 2026-07-28

- **摘要：** Windows 上 torch.compile / inductor 运行时改进：在 win32 上启用并行 inductor 编译；在每个阶段首个 batch 之后关闭 compile worker 以释放 CUDA 上下文；运行期间启用 `cudnn.benchmark`；以及更均匀的 VAE 时序切片，以减少编译形状变体。
- **技术详情：** 请参阅 [v1.3 发行说明](v1.3.md) 获取完整说明

## v1.2 — 2026-07-28

- **摘要：** 通过构建时 `comfy.ops.mixed_precision_ops` 为 SeedVR2 DiT 增加原生 NVFP4 加载；并修复 Windows / inductor，使 FP16 VAE 的 `torch.compile` 不再因 cp932 解码或 `aten.bmm` 的 fallback+decomp 断言而失败。v1.1 的 INT8 路径仍然可用。
- **技术详情：** 请参阅 [v1.2 发行说明](v1.2.md) 获取完整说明

## v1.1 — 2026-07-27

- **摘要：** 通过构建时 `comfy.ops.mixed_precision_ops` 为 SeedVR2 DiT 增加原生 INT8 加载（`int8_tensorwise` + `comfy_quant` / `weight_scale`），使 INT8 权重包在 `load_state_dict` 过程中保持量化，而不是展开为完整 FP16（降低显存）。仅限 DiT；VAE 仍为 FP16。
- **技术详情：** 请参阅 [v1.1 发行说明](v1.1.md) 获取完整说明

## v1.0 — 2026-04-05

- **摘要：** 节点加载时将缺失的 SeedVR2 依赖自动安装到当前 ComfyUI 所用 Python（`sys.executable`），以解决 Vast.ai / RunPod 等云模板上终端 `pip` 与 ComfyUI 虚拟环境不一致导致的 `ModuleNotFoundError`（例如 `diffusers`、`rotary_embedding_torch`）。
- **技术详情：** 请参阅 [v1.0 发行说明](v1.0.md) 获取完整说明
