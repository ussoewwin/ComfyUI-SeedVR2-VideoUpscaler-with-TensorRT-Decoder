<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder/blob/main/zhmd/v1.5.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

**Tag:** `v1.5`  
**Scope:** TensorRT VAE Decoder Integration & Execution Pipeline  
**Date:** 2026-09-03  

This release introduces dedicated **TensorRT VAE Decoder** acceleration for SeedVR2 in ComfyUI, delivering significant speedups (2x–5x) during the VAE decoding phase (Phase 3) while maintaining exact mathematical parity and robust memory safety.

The TensorRT VAE backend architecture in this repository is inspired by [VRGDG-SeedVR2-TensorRT-Studio](https://github.com/vrgamegirl19/VRGDG-SeedVR2-TensorRT-Studio) (Apache 2.0).

---

## 1. Summary of Changes

| Component / Feature | Implementation Details |
|---|---|
| **Dedicated Decoder Node** | New `SeedVR2LoadTensorRTVAEDecoder` node under `SEEDVR2` category, decoupling encoder and decoder engine configurations |
| **Separated Upscaler Inputs** | `SeedVR2VideoUpscaler` now accepts independent `vae_encode` and `vae_decode` custom inputs |
| **Engine Frame Auto-Discovery** | Dynamically scans `tensorrt_backend/artifacts/` for available `vae_decoder_tile_*_*f.rtxplan` engines and populates 4n+1 frame options |
| **Multi-Tile & Studio Overlap Parity** | Supports 512px tiles (tile=64, overlap=24) and 256px tiles (tile=32, overlap=12) with Studio-compatible spatial feathered blending and `clamp(-2.0, 2.0)` |
| **Execution Context Caching** | Caches deserialized CUDA engines, execution contexts, and CUDA streams across chunks, eliminating ~4.5GB VRAM re-allocation overhead per chunk |
| **Arbitrary Batch Chunking** | Automatically chunks arbitrary long latent videos into engine-sized frame windows with 1 latent-frame (4 video frames) overlap |
| **Graceful Fallback Mechanism** | Seamlessly falls back to PyTorch standard VAE with configurable tiled decoding (`Fallback: Decode Tiled`, tile size, overlap) when engines are missing |
| **Complete VRAM Teardown** | Immediate VRAM reclamation via `release()` (`gc.collect()` + `torch.cuda.synchronize()` + double `empty_cache()`) |

---

## 2. Technical Architecture & Code Details

### 2.1 Dedicated Decoder Loader Node (`SeedVR2LoadTensorRTVAEDecoder`)
Located in `src/interfaces/trt_vae_model_loader.py`:
- **Node ID:** `SeedVR2LoadTensorRTVAEDecoder`
- **Category:** `SEEDVR2`
- **Output:** `SEEDVR2_VAE` (configured specifically for Phase 3 decoding)
- **Inputs & Parameters:**
  - `model`: Standard VAE weights (e.g. `ema_vae_fp16.safetensors`) for fallback initialization
  - `device`: GPU device for execution (`cuda:0`)
  - `Fallback: Decode Tiled`, `Fallback: Decode Tile Size`, `Fallback: Decode Tile Overlap`: Dedicated fallback settings applied only when the TRT engine is unavailable
  - `engine_frames`: Auto-populated dropdown listing available 4n+1 frame sizes (`auto`, `29`, `21`, `5`, etc.)

### 2.2 TensorRT Decoder Execution Pipeline (`src/core/trt_decoder.py`)
- **Engine Search & Selection (`find_engine_path`)**:
  Searches `tensorrt_backend/artifacts/` and `ComfyUI/models/tensorrt/seedvr2/`.
  Prefers 512px tile engine (`vae_decoder_tile_512_{N}f.rtxplan`) and falls back to 256px tile engine (`vae_decoder_tile_256_{N}f.rtxplan`).
- **Context Caching (`_engine`)**:
  Deserializes the engine once and maintains the execution context in `_DECODER_ENGINES`. Context reuse prevents expensive CUDA driver allocations during batch processing.
- **Tiled Spatial Blending (`_decode_single_chunk`)**:
  Pads latent spatial dimensions to tile multiples, processes tiles asynchronously via dedicated CUDA streams, applies directional linear feathering (`_feather`), normalizes by overlapping weights, and clamps output values to `[-2.0, 2.0]` before casting to target dtype.
- **Chunked Temporal Decoding (`_decode_chunked` / `src/core/infer.py`)**:
  When latent frame length exceeds the engine's fixed frame capacity, the latent tensor is sliced into overlapping sub-chunks (`lat_stride = engine_latent - 1`). Output video segments are seamlessly joined along temporal boundaries.

### 2.3 Workflow Decoupling (`src/interfaces/video_upscaler.py`)
`SeedVR2VideoUpscaler` now exposes dedicated `vae_encode` and `vae_decode` sockets. This allows workflows to use standard PyTorch VAE encoding in Phase 1 and high-speed TensorRT decoding in Phase 3, avoiding unnecessary encoder overhead while maximizing decoding throughput.

---

## 3. Code Surface in This Release

| File Path | Role & Changes |
|---|---|
| `src/core/trt_decoder.py` | Core TensorRT VAE decode engine, caching, tiling, chunking, and memory release |
| `src/interfaces/trt_vae_model_loader.py` | Implementation of `SeedVR2LoadTensorRTVAEDecoder` node with auto engine discovery |
| `src/interfaces/trt_vae_builder.py` | Explicit TensorRT engine builder interface |
| `src/interfaces/video_upscaler.py` | Dual-socket `vae_encode` / `vae_decode` routing and execution control |
| `src/core/infer.py` | Stride alignment, chunk dispatching, and spatial padding for TRT VAE batches |
| `__init__.py` | Node registration for `SeedVR2LoadTensorRTVAEDecoder` and startup engine readiness checks |
| `example_workflows/SeedVR2_tensorrt_decode.json` | Complete example workflow with TensorRT VAE Decoder and 2nd upscale pipeline |

---

## 4. Verification & Parity

- **Mathematical Consistency**: Tiling and blending algorithms preserve exact pixel reconstruction parity with Studio TensorRT engines.
- **VRAM Stability**: Context caching prevents GPU memory leaks; `release()` guarantees zero residual VRAM footprint between generation runs.
- **Fallback Integrity**: In environments without TensorRT engines or NVIDIA GPUs, workflows execute safely via standard PyTorch VAE fallback.
