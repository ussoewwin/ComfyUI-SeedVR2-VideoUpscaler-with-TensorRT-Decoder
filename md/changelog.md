<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/blob/main/zhmd/changelogzh.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# Changelog

Fork release history for [ussoewwin/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler).

## v1.3 — 2026-07-28

- **Summary:** Windows torch.compile / inductor runtime improvements: parallel inductor compile on win32, shut down compile workers after each phase’s first batch to free CUDA contexts, run-scoped `cudnn.benchmark`, and more uniform VAE temporal slices to reduce compile shape variants.
- **Technical Details:** See [v1.3 Release Notes](../zhmd/v1.3.md) for complete explanation
- **Release (GitHub):** https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/releases/tag/v1.3

## v1.2 — 2026-07-28

- **Summary:** Native NVFP4 loading for SeedVR2 DiT via construction-time `comfy.ops.mixed_precision_ops`, plus Windows / inductor fixes so FP16 VAE `torch.compile` no longer fails on cp932 decode or `aten.bmm` fallback+decomp asserts. INT8 path from v1.1 remains available.
- **Technical Details:** See [v1.2 Release Notes](../zhmd/v1.2.md) for complete explanation
- **Release (GitHub):** https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/releases/tag/v1.2

## v1.1 — 2026-07-27

- **Summary:** Native INT8 loading for SeedVR2 DiT (`int8_tensorwise` + `comfy_quant` / `weight_scale`) via construction-time `comfy.ops.mixed_precision_ops`, so INT8 packs stay quantized through `load_state_dict` instead of expanding to full FP16 (VRAM reduction). DiT only; VAE remains FP16.
- **Technical Details:** See [v1.1 Release Notes](../zhmd/v1.1.md) for complete explanation
- **Release (GitHub):** https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/releases/tag/v1.1

## v1.0 — 2026-04-05

- **Summary:** Auto-install missing SeedVR2 dependencies into the active ComfyUI Python (`sys.executable`) at node load, addressing `ModuleNotFoundError` (e.g. `diffusers`, `rotary_embedding_torch`) on cloud templates such as Vast.ai / RunPod where terminal `pip` and ComfyUI’s venv diverge.
- **Technical Details:** See [v1.0 Release Notes](../zhmd/v1.0.md) for complete explanation
- **Release (GitHub):** https://github.com/ussoewwin/ComfyUI-SeedVR2_VideoUpscaler/releases/tag/v1.0
