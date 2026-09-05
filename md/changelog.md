<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder/blob/main/zhmd/changelogzh.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# Changelog

Fork release history.

## v1.8.1 — 2026-09-05

- **Summary:** Comprehensive installer and runtime reliability overhaul:
  - **Zero-Intervention Automated Installation:** Resolved dependency installation failures in `install.py` by automatically installing `requirements.txt` into the host Python environment without requiring external batch files.
  - **Attention Backend Unification & PyTorch SDPA Standard:** Removed brittle wheel auto-downloaders for FlashAttention 2 and SageAttention 2, standardizing on PyTorch native SDPA fallback (`attention_mode: sdpa`) when custom attention packages are absent ([#1](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder/issues/1)).
  - **Comprehensive FFmpeg PATH Discovery:** Implemented proactive multi-candidate directory search and `imageio_ffmpeg` fallback across runtime module loading (`__init__.py`), `install.py`, `scripts/verify_install.py`, and CLI to eliminate video export crashes.
  - **Decoder Engine Tile Size Specification:** Documented mandatory `tile_size: 256` constraint for TensorRT VAE Decoder engine compilation to prevent spatial dimension mismatch errors during inference.
- **Technical Details:** See [v1.8.1 Release Notes](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder/releases/tag/v1.8.1) for complete explanation

## v1.5 — 2026-09-03

- **Summary:** Added TensorRT VAE Decoder support with dedicated loader node (`SeedVR2LoadTensorRTVAEDecoder`), multi-tile engine support (256px/512px, 4n+1 frame sizes), cached execution context reuse, decoupled encode/decode configurations, and automatic safe fallback to PyTorch VAE on missing engines.
- **Technical Details:** See [v1.5 Release Notes](../zhmd/v1.5.md) for complete explanation

## v1.4 — 2026-07-31

- **Summary:** Register 3B HSWQ INT8 ConvRot and NVFP4 DiT packs in `MODEL_REGISTRY` (same native VRAM path as 7B).
- **Technical Details:** See [v1.4 Release Notes](../zhmd/v1.4.md) for complete explanation

## v1.3 — 2026-07-28

- **Summary:** Windows torch.compile / inductor runtime improvements: parallel inductor compile on win32, shut down compile workers after each phase’s first batch to free CUDA contexts, run-scoped `cudnn.benchmark`, and more uniform VAE temporal slices to reduce compile shape variants.
- **Technical Details:** See [v1.3 Release Notes](../zhmd/v1.3.md) for complete explanation

## v1.2 — 2026-07-28

- **Summary:** Native NVFP4 loading for SeedVR2 DiT via construction-time `comfy.ops.mixed_precision_ops`, plus Windows / inductor fixes so FP16 VAE `torch.compile` no longer fails on cp932 decode or `aten.bmm` fallback+decomp asserts. INT8 path from v1.1 remains available.
- **Technical Details:** See [v1.2 Release Notes](../zhmd/v1.2.md) for complete explanation

## v1.1 — 2026-07-27

- **Summary:** Native INT8 loading for SeedVR2 DiT (`int8_tensorwise` + `comfy_quant` / `weight_scale`) via construction-time `comfy.ops.mixed_precision_ops`, so INT8 packs stay quantized through `load_state_dict` instead of expanding to full FP16 (VRAM reduction). DiT only; VAE remains FP16.
- **Technical Details:** See [v1.1 Release Notes](../zhmd/v1.1.md) for complete explanation

## v1.0 — 2026-04-05

- **Summary:** Auto-install missing SeedVR2 dependencies into the active ComfyUI Python (`sys.executable`) at node load, addressing `ModuleNotFoundError` (e.g. `diffusers`, `rotary_embedding_torch`) on cloud templates such as Vast.ai / RunPod where terminal `pip` and ComfyUI’s venv diverge.
- **Technical Details:** See [v1.0 Release Notes](../zhmd/v1.0.md) for complete explanation
