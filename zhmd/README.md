# ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../README.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

[![View Code](https://img.shields.io/badge/📂_View_Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)

[SeedVR2](https://github.com/ByteDance-Seed/SeedVR) 的 ComfyUI 官方发布版本，支持高质量视频和图像放大。

本仓库是官方仓库（[https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)）的 Fork，基于 Apache 2.0 许可证创建。独立实现了 ConvRot INT8 与 NVFP4 量化模型支持，以及显存（VRAM）节省功能。

[![SeedVR2 v2.5 Deep Dive Tutorial](https://img.youtube.com/vi/MBtWYXq_r60/maxresdefault.jpg)](https://youtu.be/MBtWYXq_r60)

## 工作流与节点示例

### 完整工作流概览（TensorRT VAE 与量化模型）

- 工作流 JSON：[`example_workflows/SeedVR2_tensorrt_decode.json`](../example_workflows/SeedVR2_tensorrt_decode.json)

![Usage Example - Full Workflow](https://raw.githubusercontent.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder/main/docs/usage_01.png)

### TensorRT VAE 解码器节点

![Usage Example - TensorRT VAE Decoder](https://raw.githubusercontent.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder/main/docs/usage_02.png)

### TensorRT VAE 引擎构建节点

![TensorRT VAE Engine Builder Node](https://raw.githubusercontent.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder/main/docs/build.png)

**`SeedVR2 Build TensorRT VAE Engines`** 节点允许用户在 ComfyUI 中直接按需显式构建专用的 TensorRT RTX VAE 引擎（`.rtxplan`）。该节点在后台调用 GPU 追踪（`tools/cloud_export_gpu.py`）与 TensorRT 编译（`tools/cloud_build_engine.py`）。

构建完成的引擎会自动保存至 `tensorrt_backend/artifacts/` 目录中。重启 ComfyUI 后，TensorRT VAE 解码器加载节点的 `engine_frames` 下拉菜单中将自动出现新构建的帧长规格。

#### 节点参数与设置详解

- **`model`**：原始 PyTorch VAE 模型权重（例如 `ema_vae_fp16.safetensors`）。
- **`frames`**：引擎目标帧长。自动规范化为所需的 **4n+1** 格式（例如 `5`、`21`、`29`、`61`、`89`、`101`、`185`、`205` 等）。
- **`tile_size`**：空间分块尺寸（`256` 或 `512`）：
  - **`256`**：适合在 16GB–24GB 显存显卡上构建长帧序列（60f–185f+）引擎。
  - **`512`**：空间分块更大、细节还原更优，但编译时显存占用较高（建议在 16GB 显卡上用于 21f–29f 规格）。
- **`kind`**：选择构建的引擎类型：
  - **`both`**：同时构建编码器与解码器引擎。
  - **`decoder`**：仅构建 VAE 解码器引擎（推荐用于 Phase 3 解码加速）。
  - **`encoder`**：仅构建 VAE 编码器引擎。
- **`workspace_gb`**：TensorRT 编译时的最大工作区显存上限（单位：GB，默认 `8.0`–`16.0` GB）。
- **`min_ws`**：启用（`True`）后，通过二分搜索查找可成功构建的最小工作区尺寸，进一步降低运行时显存占用（编译时间略有增加）。
- **`force_rebuild`**：启用（`True`）后，即使引擎文件已存在也会强制重新构建并覆盖。
- **输出（`STRING`）**：输出构建状态、生成引擎文件名、文件大小及总耗时。可连接 `Show Text` 节点实时查看。

#### 使用与构建步骤

1. 在工作流中添加 **`SeedVR2 Build TensorRT VAE Engines`** 节点。
2. 配置所需的帧长（`frames`）、空间尺寸 `tile_size`（256 或 512）以及 `kind`（如 `decoder`）。
3. 点击 **Queue Prompt** 启动构建，系统将在后台自动完成 ONNX 导出与 TensorRT 引擎构建。
4. 构建完成后重启 ComfyUI，**`SeedVR2 Load TensorRT VAE Decoder`** 节点的 `engine_frames` 下拉列表中将显示新构建的帧数，即可开启极速解码。

## 文档

详细说明请参阅官方仓库：

https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler

本 Fork 技术指南（中文）：

- [3B INT8 / NVFP4 模型支持](SEEDVR2_3B_INT8_NVFP4_REGISTRY_GUIDE.md)
- [INT8 原生推理](SEEDVR2_INT8_NATIVE_OPS_GUIDE.md)
- [NVFP4 与 torch.compile](SEEDVR2_NVFP4_AND_TORCH_COMPILE_GUIDE.md)
- [速度 / 显存余量](SEEDVR2_SPEED_VRAM_HEADROOM.md)
- [Windows 并行编译修复](SEEDVR2_WINDOWS_PARALLEL_COMPILE_FIX.md)
- [云环境依赖错误](vastai_dependency_guide.md)

## 更新日志

- [changelogzh.md](changelogzh.md)

## 🙏 致谢

本 ComfyUI 实现由 **[NumZ](https://github.com/numz)** 与 **[AInVFX](https://www.youtube.com/@AInVFX)**（Adrien Toupet）协作完成，基于 ByteDance Seed Team 的原始 [SeedVR2](https://github.com/ByteDance-Seed/SeedVR)。

特别感谢社区贡献者 [naxci1](https://github.com/naxci1)、[thehhmdb](https://github.com/thehhmdb)、[s-cerevisiae](https://github.com/s-cerevisiae)、[benjaminherb](https://github.com/benjaminherb)、[cmeka](https://github.com/cmeka)、[FurkanGozukara](https://github.com/FurkanGozukara)、[JohnAlcatraz](https://github.com/JohnAlcatraz)、[lihaoyun6](https://github.com/lihaoyun6)、[Luchuanzhao](https://github.com/Luchuanzhao)、[Luke2642](https://github.com/Luke2642)、[proxyid](https://github.com/proxyid)、[q5sys](https://github.com/q5sys) 以及许多其他人，在官方仓库（[https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)）中的改进、错误修复与测试。

## 📜 许可证

本仓库中的代码按 Apache 2.0 许可证发布，详见 [LICENSE](../LICENSE) 文件。
