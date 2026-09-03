# ComfyUI-SeedVR2_VideoUpscaler

<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="zhmd/README.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

[![View Code](https://img.shields.io/badge/📂_View_Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)

Official release of [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) for ComfyUI that enables high-quality video and image upscaling.

This repository is a fork of the official repository ([https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)), created under the Apache 2.0 license. It independently implements support for ConvRot INT8 and NVFP4 quantized models, along with VRAM-saving features.

[![SeedVR2 v2.5 Deep Dive Tutorial](https://img.youtube.com/vi/MBtWYXq_r60/maxresdefault.jpg)](https://youtu.be/MBtWYXq_r60)

## Workflow & Node Examples

### Complete Workflow Overview (TensorRT VAE & Quantized Models)

![Usage Example - Full Workflow](https://raw.githubusercontent.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/main/docs/usage_01.png)

### TensorRT VAE Decoder Node

![Usage Example - TensorRT VAE Decoder](https://raw.githubusercontent.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/main/docs/usage_02.png)

## Documentation

For details, refer to the official repository:

https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler

### Technical Guides (This Fork)

- [3B INT8 / NVFP4 Model Registry Guide](md/SEEDVR2_3B_INT8_NVFP4_REGISTRY_GUIDE.md)
- [INT8 Native Ops Inference Guide](md/SEEDVR2_INT8_NATIVE_OPS_GUIDE.md)
- [NVFP4 & torch.compile Guide](md/SEEDVR2_NVFP4_AND_TORCH_COMPILE_GUIDE.md)
- [Speed & VRAM Headroom Analysis](md/SEEDVR2_SPEED_VRAM_HEADROOM.md)
- [Windows Parallel Compile Fix](md/SEEDVR2_WINDOWS_PARALLEL_COMPILE_FIX.md)
- [Cloud Environment Dependency Guide](md/vastai_dependency_guide.md)

## Changelog

- [md/changelog.md](md/changelog.md)

## 🙏 Credits

This ComfyUI implementation is a collaborative project by **[NumZ](https://github.com/numz)** and **[AInVFX](https://www.youtube.com/@AInVFX)** (Adrien Toupet), based on the original [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) by ByteDance Seed Team.

Special thanks to our community contributors including [naxci1](https://github.com/naxci1), [thehhmdb](https://github.com/thehhmdb), [s-cerevisiae](https://github.com/s-cerevisiae), [benjaminherb](https://github.com/benjaminherb), [cmeka](https://github.com/cmeka), [FurkanGozukara](https://github.com/FurkanGozukara), [JohnAlcatraz](https://github.com/JohnAlcatraz), [lihaoyun6](https://github.com/lihaoyun6), [Luchuanzhao](https://github.com/Luchuanzhao), [Luke2642](https://github.com/Luke2642), [proxyid](https://github.com/proxyid), [q5sys](https://github.com/q5sys), and many others for their improvements, bug fixes, and testing in the official repository ([https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)).

### TensorRT VAE backend

The TensorRT VAE encode/decode engine in this repository was inspired by [VRGDG-SeedVR2-TensorRT-Studio](https://github.com/vrgamegirl19/VRGDG-SeedVR2-TensorRT-Studio) (Apache 2.0). The author of that project had considered porting the DiT to TensorRT, but gave up due to the many difficulties involved and instead focused on improving performance by creating a ComfyUI node that supports ConvRot INT8/NVFP4 quantized models. The idea of porting the VAE encode/decode to TensorRT, however, came from this project — without that work, this approach would never have been conceived. Sincere respect and gratitude to the original author.

## 📜 License

The code in this repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

The TensorRT VAE backend is inspired by [VRGDG-SeedVR2-TensorRT-Studio](https://github.com/vrgamegirl19/VRGDG-SeedVR2-TensorRT-Studio), which is also released under the Apache 2.0 license. Attribution and copyright notices are retained in accordance with Apache 2.0 requirements.
