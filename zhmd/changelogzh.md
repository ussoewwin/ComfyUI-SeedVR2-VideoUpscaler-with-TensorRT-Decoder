<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/blob/main/md/changelog.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

# 更新日志

[ussoewwin/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler) 的 Fork 发行历史。

## v1.4 — 2026-07-31

- **摘要：** 在 `MODEL_REGISTRY` 中登记 3B HSWQ INT8 ConvRot 与 NVFP4 DiT 权重包（与 7B 相同的原生显存路径）；按 registry 规格选择 3B/7B 配置目录；并保留 durable `_dit_is_nvfp4`，使 materialize 清空 `_dit_checkpoint` 之后 NVFP4 仍可跳过 autocast。
- **Release (GitHub):** https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/releases/tag/v1.4

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
