<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder/blob/main/zhmd/v1.5.1.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

**Tag:** `v1.5.1`  
**Scope:** Installer Reliability Overhaul, Attention Backend Unification (PyTorch Native SDPA), Universal FFmpeg PATH Resolution, and TensorRT VAE Decoder Engine Constraints  
**Date:** 2026-09-05  

This maintenance and stabilization release addresses critical installation issues, eliminates fragile external attention wheel dependencies, implements universal FFmpeg discovery across ComfyUI environments, and clarifies spatial tile size constraints for TensorRT VAE Decoder engine compilation.

Addresses and resolves: [#1](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder/issues/1).

---

## 1. Summary of Changes

| Area | Component | Implementation Details |
|---|---|---|
| **Zero-Intervention Setup** | `install.py` | Automatically installs all dependencies defined in `requirements.txt` into the active Python environment without requiring users to manually run batch scripts (`Install TensorRT SeedVR2.bat`). |
| **Attention Backend** | `install.py`<br>`scripts/install.ps1`<br>`scripts/verify_install.py` | Completely removed custom wheel auto-downloaders for FlashAttention 2 and SageAttention 2. Standardized on PyTorch native Scaled Dot-Product Attention (SDPA, `attention_mode: sdpa`) when specialized attention kernels are not installed. If users wish to use FA or SA, they must manually build/install them for their specific environment. |
| **FFmpeg Discovery** | `__init__.py`<br>`install.py`<br>`inference_cli.py`<br>`scripts/verify_install.py` | Implemented proactive multi-candidate directory search and `imageio_ffmpeg` fallback across runtime module loading, installation, and CLI, resolving missing `ffmpeg` binaries in isolated ComfyUI portable environments. |
| **TRT VAE Engine Spec** | `README.md`<br>`zhmd/README.md` | Documented mandatory `tile_size: 256` constraint for TensorRT VAE Decoder engine builds (`kind: decoder` or `kind: both`), preventing runtime spatial dimension mismatches during inference. |
| **Verification Cleanliness** | `scripts/verify_install.py` | Missing FlashAttention / SageAttention kernels or optional pre-built engines are now reported cleanly as informational notices (`not available (using SDPA)`) rather than raising fatal exit codes. |

---

## 2. Deep Dive & Architectural Improvements

### 2.1 Zero-Intervention Automated Dependency Installation (`install.py`)
Previously, installing this custom node purely through ComfyUI Manager or `install.py` left core requirements uninstalled (such as `diffusers`, `peft`, `rotary_embedding_torch`, `onnxscript`, and `polygraphy`) because `install.py` only validated pre-existing wheels and TensorRT engines, requiring users to manually execute `Install TensorRT SeedVR2.bat`.

In `v1.5.1`, `install.py` incorporates an automated dependency resolution phase:
```python
def install_requirements():
    req_file = os.path.join(REPO_ROOT, "requirements.txt")
    if not os.path.isfile(req_file):
        return True
    cmd = [sys.executable, "-s", "-m", "pip", "install", "-r", req_file, "--no-warn-script-location"]
    result = subprocess.run(cmd)
    return result.returncode == 0
```
This guarantees that all required packages are present upon first run in both virtual environments and standalone embedded Python distributions.

### 2.2 Attention Backend Architecture Unification & SDPA Standard
Hardcoding wheel URLs for FlashAttention 2 and SageAttention 2 caused continuous installation failures across differing Python versions (Python 3.11, 3.12, 3.13) and platform ABI differences (Issue [#1](https://github.com/ussoewwin/ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder/issues/1)).

- **Elimination of Wheel Auto-Downloaders:**
  - Removed `FLASH_ATTN_WHEELS`, `SAGE_ATTN_WHEELS`, and `install_wheels()` from `install.py`.
  - Removed `Install-CachedWheel` and automated wheel acquisition from `scripts/install.ps1`.
- **Manual Installation Policy for FA/SA:**
  - FlashAttention and SageAttention are optional acceleration backends. Users wishing to use them must manually compile or install appropriate wheels compatible with their exact Python and CUDA ABI into ComfyUI.
- **Seamless PyTorch SDPA Standard:**
  - When specialized acceleration kernels (FlashAttention or SageAttention) are not installed, the pipeline smoothly utilizes PyTorch native SDPA (`torch.nn.functional.scaled_dot_product_attention`).
  - Users operating without compiled attention kernels can simply set `attention_mode` to `sdpa` in their node parameters for full functionality without crashes or missing symbols.
  - `scripts/verify_install.py` reports attention status as `not available (using SDPA)` with clean exit code `0`.

### 2.3 Universal Multi-Candidate FFmpeg PATH Discovery
SeedVR2 video processing pipelines require an accessible `ffmpeg` executable for frame extraction and video encoding. In standard Windows portable ComfyUI setups, FFmpeg is frequently located in portable parent directories or bundled inside Python site-packages (`imageio_ffmpeg`), but absent from `os.environ["PATH"]`.

`_ensure_ffmpeg_path()` has been integrated into `__init__.py`, `install.py`, `scripts/verify_install.py`, and `inference_cli.py`:
1. **Existing PATH Verification:** Checks `shutil.which("ffmpeg")`.
2. **Directory Candidate Search:** Proactively inspects ComfyUI installation hierarchies, searching up to two directory levels up and looking into common relative folders (`ffmpeg/bin`, `bin`).
3. **`imageio_ffmpeg` Dynamic Fallback:** If still not found, automatically queries `imageio_ffmpeg.get_ffmpeg_exe()`, derives its directory, and prepends it to `os.environ["PATH"]`.

Because this logic runs in `__init__.py` upon custom node loading, ComfyUI processes automatically acquire the correct FFmpeg environment without requiring manual Windows system PATH modifications.

### 2.4 TensorRT VAE Decoder Spatial Tile Size Constraints
The TensorRT VAE Decoder backend (`src/core/trt_decoder.py`) is engineered to process 256x256 spatial patches (`tile_size: 256`). Compiling a decoder engine with `tile_size: 512` produces an execution plan with mismatched spatial dimensions, resulting in tensor shape mismatch errors during reconstruction.

`README.md` and `zhmd/README.md` have been updated with explicit notices:
- **`tile_size: 256`**: **Mandatory for Decoder engines (`kind: decoder` or `kind: both`)**.
- **`tile_size: 512`**: Reserved exclusively for VAE Encoder engines (`kind: encoder`).

---

## 3. Verification & Diagnostic Matrix

| Test Scenario | Prior Behavior (`v1.5`) | Updated Behavior (`v1.5.1`) | Status |
|---|---|---|---|
| Clean ComfyUI Install via Manager | Missing `diffusers` / `rotary_embedding_torch` | `install.py` automatically installs `requirements.txt` | **Resolved** |
| Missing FlashAttention / SageAttention | Pip build crashes or wheel version mismatch | Standardizes on native PyTorch SDPA (`attention_mode: sdpa`) | **Resolved** |
| FFmpeg in Portable ComfyUI Root | Video export fails with `ffmpeg not found` | Multi-directory discovery + `imageio_ffmpeg` resolves path | **Resolved** |
| `scripts/verify_install.py` Execution | Exited with error `1` on missing optional wheels | Informational pass with `not available (using SDPA)` | **Resolved** |
| Decoder Engine Build | User confusion leading to `tile_size: 512` shape errors | Explicit `[!IMPORTANT]` warnings enforce `tile_size: 256` | **Documented** |

---

## 4. Associated Commits

- `17aa225`: Multi-version install fixes and automatic requirements installation
- `4de4d2f`: Remove FA/SA wheel management from installers and rely on standard PyTorch SDPA
- `d58c66a`: Comprehensive FFmpeg PATH resolution across runtime, installer, and CLI
- `a5d2b1a`: Specify mandatory tile_size 256 for TensorRT VAE decoder engines in README and zhmd
- `4b4a55b`: Add changelog entries in English and Chinese
- `49f7c3d`: Correct changelog version to v1.5.1 in English and Chinese
