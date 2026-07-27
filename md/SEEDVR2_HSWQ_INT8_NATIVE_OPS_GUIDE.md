# SeedVR2 Video Upscaler — HSWQ INT8 Native Inference Guide

対象カスタムノード: `ComfyUI/custom_nodes/seedvr2_videoupscaler`  
（本解説は当該カスタムノードツリーの修正について述べる。別リポジトリのミラーは対象外。）

---

## ① 対策の概要

### 問題

HSWQ 形式の SeedVR2 DiT INT8 重み（`int8_tensorwise` + `comfy_quant` / `weight_scale`）を、従来どおり **ロード後に Linear を差し替える**（GGUF 系と同様の post-load replace）だけでは、ComfyUI の `comfy.ops._load_quantized_module` が走らない。結果として INT8 を **推論前に FP16 へ全面展開**する経路に落ち、VRAM 削減の目的を失う。

### 方針

1. **構築時（construction-time）** に `comfy.ops.mixed_precision_ops` を NaDiT へ注入する。  
2. DiT 内の `nn.Linear` を、`operations` 引数経由で **`operations.Linear`（= comfy.ops の Linear）** に差し替え可能にする。  
3. safetensors ロード直前に  
   - `*.comfy_quant` を CPU へ移す（`.numpy()` 要件）  
   - meta 構築された Linear の `factory_kwargs["device"]` を実デバイスへ直す  
4. チェックポイント検出は `*.comfy_quant` 内 JSON の `format == "int8_tensorwise"`。  
5. ベンチ用に `seedvr2_int8_bench.py` を同梱し、実 ComfyUI 本体 + 本カスタムノード上で FP16 vs native INT8 を比較できるようにする。

### データ流（要約）

```
HSWQ INT8 .safetensors
  → checkpoint_is_hswq_int8()
  → create_object(..., operations=mixed_precision_ops(...))  # meta 上で構築
  → prepare_hswq_state_dict_for_comfy_ops() + patch_ops_factory_device()
  → load_state_dict → comfy.ops._load_quantized_module
  → 推論時は QuantizedTensor を保持したまま matmul（VRAM 節約）
```

---

## ② 追加・修正したファイル名

### 新規追加

| パス |
|------|
| `src/optimization/int8_native_ops.py` |
| `seedvr2_int8_bench.py` |

### 修正

| パス | 役割 |
|------|------|
| `src/common/config.py` | `create_object` が `**extra_kwargs`（`operations`）をコンストラクタへ渡す |
| `src/core/model_loader.py` | HSWQ 検出・ops 注入・load 前 prep |
| `src/utils/model_registry.py` | INT8 モデル名登録と 7B 設定解決 |
| `src/models/dit_3b/nadit.py` | `operations` 伝播 |
| `src/models/dit_3b/mlp.py` | `ops.Linear` |
| `src/models/dit_3b/embedding.py` | `ops.Linear` |
| `src/models/dit_3b/patch/patch_v1.py` | `ops.Linear` |
| `src/models/dit_3b/nablocks/mmsr_block.py` | `operations` 伝播 |
| `src/models/dit_3b/nablocks/attention/mmattn.py` | `ops.Linear` |
| `src/models/dit_7b/nadit.py` | `operations` 伝播 |
| `src/models/dit_7b/mlp.py` | `ops.Linear` |
| `src/models/dit_7b/embedding.py` | `ops.Linear` |
| `src/models/dit_7b/patch.py` | `ops.Linear` |
| `src/models/dit_7b/nablocks/mmsr_block.py` | `operations` 伝播 |
| `src/models/dit_7b/blocks/mmdit_window_block.py` | `ops.Linear` / 伝播 |

---

## ③ 追加・修正したコード全文

以下はカスタムノード上の **現行ファイル全文**（当該修正を含む完成形）である。

### 新規追加ファイル

### `src/optimization/int8_native_ops.py`

```python
"""
HSWQ INT8 native inference via ComfyUI comfy.ops construction-time injection.

HSWQ safetensors carry ``comfy_quant`` + ``weight_scale``. Native VRAM-saving
load requires Linear modules that already implement
``_load_from_state_dict`` → ``comfy.ops._load_quantized_module`` at
``load_state_dict`` time. That is provided by
``comfy.ops.mixed_precision_ops``.

Post-load Linear replace (GGUF-style) does not interpret ``comfy_quant`` and
is the wrong path for this format.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch


def checkpoint_is_hswq_int8(checkpoint_path: Optional[str]) -> bool:
    """True if safetensors has at least one ``*.comfy_quant`` with format int8_tensorwise."""
    if not checkpoint_path:
        return False
    path = str(checkpoint_path)
    if not (path.endswith(".safetensors") or path.endswith(".sft")):
        return False
    if not os.path.isfile(path):
        return False
    try:
        from safetensors import safe_open
    except ImportError:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.endswith(".comfy_quant"):
                    continue
                raw = f.get_tensor(key)
                if raw.dtype != torch.uint8:
                    continue
                conf = json.loads(raw.numpy().tobytes())
                if conf.get("format") == "int8_tensorwise":
                    return True
    except Exception:
        return False
    return False


def get_hswq_mixed_precision_ops(compute_dtype: torch.dtype = torch.float16) -> Any:
    """
    Return ``comfy.ops.mixed_precision_ops`` with empty quant_config.

    Empty config is intentional: layers that carry ``comfy_quant`` become
    QuantizedTensor; layers without markers load as plain compute_dtype Parameters.
    """
    import comfy.ops as comfy_ops

    return comfy_ops.mixed_precision_ops(
        quant_config={},
        compute_dtype=compute_dtype,
        full_precision_mm=False,
        disabled=[],
    )


def resolve_linear_ops(operations: Optional[Any] = None) -> Any:
    """Return an object with ``.Linear`` (operations or ``torch.nn``)."""
    if operations is None:
        return torch.nn
    return operations


def prepare_hswq_state_dict_for_comfy_ops(state: dict) -> dict:
    """
    Move ``*.comfy_quant`` tensors to CPU in-place.

    ``comfy.ops._load_quantized_module`` does ``layer_conf.numpy().tobytes()``.
    That requires host memory; SeedVR2 often loads safetensors straight to CUDA,
    which raises: can't convert cuda device type tensor to numpy.
    Weight / scale tensors may stay on the target device.
    """
    for key, value in list(state.items()):
        if not key.endswith("comfy_quant"):
            continue
        if torch.is_tensor(value) and value.device.type != "cpu":
            state[key] = value.cpu()
    return state


def patch_ops_factory_device(model: torch.nn.Module, device: torch.device) -> int:
    """
    Point ``factory_kwargs["device"]`` at the materialization device.

    NaDiT is built under ``torch.device("meta")`` so Linear ``device`` is often
    ``None``/meta. ``_load_quantized_module`` places QuantizedTensor via that
    field; without this patch weights can remain on meta after assign load.
    """
    patched = 0
    for module in model.modules():
        fk = getattr(module, "factory_kwargs", None)
        if not isinstance(fk, dict) or "device" not in fk:
            continue
        fk["device"] = device
        patched += 1
    return patched
```

### `seedvr2_int8_bench.py`

```python
#!/usr/bin/env python3
"""
SeedVR2 Native INT8 Benchmark (construction-time comfy.ops injection)
=====================================================================
Compare community FP16 SeedVR2 DiT vs HSWQ native INT8 (int8_tensorwise,
optional ConvRot) through numz SeedVR2_VideoUpscaler.

HSWQ INT8 safetensors keep comfy_quant + weight_scale. The videoupscaler path
injects comfy.ops.mixed_precision_ops at DiT construction so load_state_dict
hits _load_quantized_module (QuantizedTensor stays INT8 in VRAM).

This bench does NOT dequantize INT8 to a temporary FP16 safetensors.

Primary metric: FP16 output vs native INT8 output (MSE / SSIM / diff PNG).

Path layout (no hardcoded drive letters — works for any install):

  Layout A — ComfyUI custom node (recommended for end users)
    <ComfyUI>/custom_nodes/seedvr2_videoupscaler/seedvr2_int8_bench.py
      seedvr2 root  = this script's directory
      ComfyUI root  = nearest ancestor that contains comfy/ops.py
      model_dir     = <ComfyUI>/models/SEEDVR2  (default)

  Layout B — HSWQ repository twin
    <hswq>/seedvr2_videoupscaler/seedvr2_int8_bench.py
    or <hswq>/benchmark/seedvr2_int8_bench.py
      seedvr2 root  = <hswq>/seedvr2_videoupscaler
      ComfyUI root  = <hswq>/ComfyUI-master
      model_dir     = <ComfyUI>/models/SEEDVR2 when present

Example (from custom_nodes/seedvr2_videoupscaler, filenames under models/SEEDVR2):

  python seedvr2_int8_bench.py ^
    --fp16 seedvr2_ema_7b_fp16.safetensors ^
    --int8 seedvr2_7b_int8_convrot.safetensors ^
    --vae  ema_vae_fp16.safetensors

--image is optional: when omitted, a synthetic RGB pattern is used.
Default resolution=1080 / color_correction=lab match videoupscaler CLI defaults.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import types
from pathlib import Path

# Windows cp932 consoles choke on seedvr2 emoji prints during import.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim


SCRIPT_DIR = Path(__file__).resolve().parent


def _clean_path(p: str) -> str:
    """PowerShell trailing \\\" leaves a final backslash; strip it."""
    return os.path.normpath(str(p).rstrip("\\/"))


def _find_comfy_root(start: Path) -> Path | None:
    """Walk ancestors for a directory that contains comfy/ops.py."""
    for parent in [start, *start.parents]:
        if (parent / "comfy" / "ops.py").is_file():
            return parent
    return None


def _discover_defaults() -> tuple[Path, Path, Path | None, str]:
    """
    Resolve (seedvr2_root, comfy_root, default_model_dir, layout_name)
    without any absolute/drive-hardcoded paths.
    """
    # Layout A: this file lives inside the SeedVR2 package root.
    if (SCRIPT_DIR / "inference_cli.py").is_file() and (SCRIPT_DIR / "src").is_dir():
        seed = SCRIPT_DIR
        # Prefer real ComfyUI ancestor (custom_nodes/... layout).
        comfy = _find_comfy_root(SCRIPT_DIR)
        layout = "comfyui_custom_node"
        if comfy is None:
            # HSWQ twin: ComfyUI-master sits next to seedvr2_videoupscaler/.
            sibling = SCRIPT_DIR.parent / "ComfyUI-master"
            if (sibling / "comfy" / "ops.py").is_file():
                comfy = sibling
                layout = "hswq_seedvr2_package"
        if comfy is None:
            raise RuntimeError(
                "Could not find ComfyUI root (comfy/ops.py) above "
                f"{SCRIPT_DIR}, and sibling ComfyUI-master is missing. "
                "Install as custom_nodes/seedvr2_videoupscaler under a "
                "ComfyUI tree, or pass --comfy_path."
            )
        model_dir = comfy / "models" / "SEEDVR2"
        if not model_dir.is_dir() and layout == "hswq_seedvr2_package":
            host = _find_comfy_root(Path.cwd())
            if host is not None:
                host_models = host / "models" / "SEEDVR2"
                if host_models.is_dir():
                    model_dir = host_models
        return seed, comfy, (model_dir if model_dir.is_dir() else None), layout

    # Layout B: this file lives under hswq/benchmark/ (or similar).
    repo = SCRIPT_DIR.parent
    seed = repo / "seedvr2_videoupscaler"
    comfy = repo / "ComfyUI-master"
    if seed.is_dir() and (comfy / "comfy" / "ops.py").is_file():
        model_dir = comfy / "models" / "SEEDVR2"
        if not model_dir.is_dir():
            # Prefer the host ComfyUI models folder when twin has none.
            host = _find_comfy_root(Path.cwd())
            if host is not None:
                host_models = host / "models" / "SEEDVR2"
                if host_models.is_dir():
                    model_dir = host_models
        return seed, comfy, (model_dir if model_dir.is_dir() else None), "hswq_repo"

    raise RuntimeError(
        "Cannot discover SeedVR2 / ComfyUI layout from "
        f"{SCRIPT_DIR}. Place this script in "
        "custom_nodes/seedvr2_videoupscaler/ or pass --seedvr2_path / --comfy_path."
    )


_DEFAULT_SEED, _DEFAULT_COMFY, _DEFAULT_MODEL_DIR, _LAYOUT = _discover_defaults()
DEFAULT_SEEDVR2_PATH = _DEFAULT_SEED
DEFAULT_COMFY_PATH = _DEFAULT_COMFY
DEFAULT_MODEL_DIR = _DEFAULT_MODEL_DIR
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "seedvr2_out"


def _dit_size_tag(*names: str) -> str:
    """
    SeedVR2 configure_runner selects configs_7b iff '7b' is in dit_model
    filename (else configs_3b). INT8 filename must carry the same marker.
    """
    joined = " ".join(Path(n).name.lower() for n in names if n)
    if "7b" in joined:
        return "7b"
    if "3b" in joined:
        return "3b"
    raise ValueError(
        "Cannot infer SeedVR2 DiT size (3b/7b) from filenames: "
        + ", ".join(repr(Path(n).name) for n in names if n)
        + ". Rename sources to include 3b or 7b, e.g. seedvr2_ema_7b_fp16.safetensors."
    )


def make_synthetic_rgb(short_edge: int = 512) -> Image.Image:
    h = short_edge
    w = int(round(short_edge * 16 / 9))
    w = max(w, short_edge)
    img = Image.new("RGB", (w, h), (32, 40, 56))
    draw = ImageDraw.Draw(img)
    for i in range(0, w, 32):
        draw.line([(i, 0), (i, h)], fill=(i % 255, 90, 140), width=1)
    for j in range(0, h, 32):
        draw.line([(0, j), (w, j)], fill=(80, j % 255, 160), width=1)
    draw.ellipse(
        [w // 4, h // 4, 3 * w // 4, 3 * h // 4],
        outline=(220, 180, 60),
        width=4,
    )
    draw.rectangle(
        [w // 8, h // 8, w // 3, h // 3],
        fill=(180, 60, 90),
        outline=(255, 255, 255),
    )
    return img


def pil_to_thwc_f16(img: Image.Image) -> torch.Tensor:
    """[T=1, H, W, C] float16 in [0,1] — videoupscaler image tensor layout."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...].to(torch.float16)


def thwc_to_pil(t: torch.Tensor) -> Image.Image:
    x = t.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    arr = (x.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return Image.fromarray(arr, mode="RGB")


def calculate_metrics(img1, img2):
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    mse = np.mean((arr1 - arr2) ** 2)
    score_ssim = ssim(arr1, arr2, win_size=3, channel_axis=2, data_range=255)
    return mse, score_ssim


def _resolve_weight(path_or_name: str, model_dir: Path | None, flag: str) -> Path:
    """
    Accept either a plain filename (resolved under model_dir) or any filesystem path.
    Never requires a hardcoded absolute default.
    """
    raw = Path(_clean_path(path_or_name))
    if raw.is_file():
        return raw.resolve()
    if model_dir is not None:
        candidate = (model_dir / raw.name).resolve()
        if candidate.is_file():
            return candidate
        # Also allow relative subpaths under model_dir.
        candidate2 = (model_dir / raw).resolve()
        if candidate2.is_file():
            return candidate2
    raise FileNotFoundError(
        f"{flag} not found: {path_or_name}"
        + (f" (also looked under {model_dir})" if model_dir is not None else "")
    )


def _stub_comfy_aimdo() -> None:
    try:
        import comfy_aimdo  # noqa: F401
    except Exception:
        m = types.ModuleType("comfy_aimdo")
        m.__file__ = "<stub>"
        m.__path__ = []
        sys.modules["comfy_aimdo"] = m
        filt = types.ModuleType("comfy_aimdo.filter")
        filt.filter_modules = lambda *a, **k: None
        sys.modules["comfy_aimdo.filter"] = filt
        model_vbar = types.ModuleType("comfy_aimdo.model_vbar")
        sys.modules["comfy_aimdo.model_vbar"] = model_vbar
        ta = types.ModuleType("comfy_aimdo.torch")
        sys.modules["comfy_aimdo.torch"] = ta


def _install_package_paths(*, seedvr2_path: str, comfy_path: str) -> tuple[str, str]:
    """
    Put package roots on sys.path:
      1) seedvr2_videoupscaler (src / inference_cli)
      2) ComfyUI root (comfy.ops)
    """
    seed_root = Path(seedvr2_path).resolve()
    comfy_root = Path(comfy_path).resolve()
    if not seed_root.is_dir():
        raise FileNotFoundError(f"--seedvr2_path not found: {seed_root}")
    if not (seed_root / "inference_cli.py").is_file():
        raise FileNotFoundError(
            f"--seedvr2_path missing inference_cli.py: {seed_root}"
        )
    if not comfy_root.is_dir():
        raise FileNotFoundError(f"--comfy_path not found: {comfy_root}")
    if not (comfy_root / "comfy" / "ops.py").is_file():
        raise FileNotFoundError(f"comfy.ops missing under --comfy_path: {comfy_root}")

    allowed = {seed_root, comfy_root}
    prepend = [str(seed_root), str(comfy_root)]
    sys.path = prepend + [
        p for p in sys.path if Path(p).resolve() not in allowed
    ]
    os.environ["PYTHONPATH"] = (
        os.pathsep.join(prepend) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )

    # Same pattern as krea2_int8_bench: keep cli_args from swallowing bench argv.
    import comfy.options

    comfy.options.enable_args_parsing(False)
    _stub_comfy_aimdo()
    return str(seed_root), str(comfy_root)


def _build_cli_args(
    *,
    dit_model: str,
    model_dir: str,
    resolution: int,
    seed: int,
    color_correction: str,
    batch_size: int,
    attention_mode: str,
    blocks_to_swap: int,
    dit_offload_device: str,
    vae_offload_device: str,
    tensor_offload_device: str,
) -> argparse.Namespace:
    """Minimal Namespace matching inference_cli._process_frames_core expectations."""
    return argparse.Namespace(
        dit_model=dit_model,
        model_dir=model_dir,
        resolution=resolution,
        max_resolution=0,
        batch_size=batch_size,
        uniform_batch_size=False,
        seed=seed,
        skip_first_frames=0,
        load_cap=0,
        chunk_size=0,
        prepend_frames=0,
        temporal_overlap=0,
        color_correction=color_correction,
        input_noise_scale=0.0,
        latent_noise_scale=0.0,
        dit_offload_device=dit_offload_device,
        vae_offload_device=vae_offload_device,
        tensor_offload_device=tensor_offload_device,
        blocks_to_swap=blocks_to_swap,
        swap_io_components=False,
        vae_encode_tiled=False,
        vae_encode_tile_size=1024,
        vae_encode_tile_overlap=128,
        vae_decode_tiled=False,
        vae_decode_tile_size=1024,
        vae_decode_tile_overlap=128,
        tile_debug="false",
        attention_mode=attention_mode,
        compile_dit=False,
        compile_vae=False,
        compile_backend="inductor",
        compile_mode="default",
        compile_fullgraph=False,
        compile_dynamic=False,
        compile_dynamo_cache_size_limit=64,
        compile_dynamo_recompile_limit=128,
        cache_dit=False,
        cache_vae=False,
        debug=False,
    )


def run_branch(
    *,
    label: str,
    dit_model: str,
    model_dir: str,
    frames: torch.Tensor,
    args_ns: argparse.Namespace,
) -> tuple[Image.Image, float, float]:
    from src.utils.debug import Debug
    from inference_cli import _process_frames_core

    print(f"\n=== {label}: SeedVR2 videoupscaler ===")
    print(f"  dit_model: {dit_model}")
    print(f"  model_dir: {model_dir}")

    ns = types.SimpleNamespace(**vars(args_ns))
    ns.dit_model = dit_model
    ns.model_dir = model_dir

    debug = Debug(enabled=False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    result = _process_frames_core(
        frames_tensor=frames,
        args=ns,
        device_id="0",
        debug=debug,
        runner_cache=None,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )
    print(f"  wall: {elapsed:.2f}s  peak_vram={peak_gb:.2f} GiB  out={tuple(result.shape)}")

    img = thwc_to_pil(result)
    del result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return img, elapsed, peak_gb


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SeedVR2 native INT8 bench (FP16 vs HSWQ INT8 via construction-time ops)"
    )
    parser.add_argument(
        "--fp16",
        required=True,
        help="FP16 SeedVR2 DiT safetensors (filename or path)",
    )
    parser.add_argument(
        "--int8",
        required=True,
        help="HSWQ INT8 SeedVR2 DiT safetensors (filename or path)",
    )
    parser.add_argument(
        "--vae",
        required=True,
        help="SeedVR2 VAE safetensors (basename should be ema_vae_fp16.safetensors)",
    )
    parser.add_argument(
        "--seedvr2_path",
        default=str(DEFAULT_SEEDVR2_PATH),
        help=f"seedvr2_videoupscaler root (default: {DEFAULT_SEEDVR2_PATH})",
    )
    parser.add_argument(
        "--comfy_path",
        default=str(DEFAULT_COMFY_PATH),
        help=f"ComfyUI root for comfy.ops (default: {DEFAULT_COMFY_PATH})",
    )
    parser.add_argument(
        "--model_dir",
        default=str(DEFAULT_MODEL_DIR) if DEFAULT_MODEL_DIR is not None else None,
        help=(
            "Directory containing DiT/VAE filenames "
            f"(default: {DEFAULT_MODEL_DIR or 'directory of --fp16'})"
        ),
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional input image. When omitted, a synthetic RGB pattern is used.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1080,
        help="Target short-side resolution (videoupscaler default: 1080)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1, help="Frames per batch (4n+1; image=1)")
    parser.add_argument(
        "--color",
        default="lab",
        choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
        help="color_correction (default: lab)",
    )
    parser.add_argument("--attention_mode", default="sdpa")
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--dit_offload_device", default="none")
    parser.add_argument("--vae_offload_device", default="none")
    parser.add_argument("--tensor_offload_device", default="cpu")
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    args.seedvr2_path = _clean_path(args.seedvr2_path)
    args.comfy_path = _clean_path(args.comfy_path)
    args.output_dir = _clean_path(args.output_dir)
    if args.model_dir is not None:
        args.model_dir = _clean_path(args.model_dir)
    if args.image is not None:
        args.image = _clean_path(args.image)

    model_dir_path = Path(args.model_dir).resolve() if args.model_dir else None
    if model_dir_path is not None and not model_dir_path.is_dir():
        raise FileNotFoundError(f"--model_dir not found: {model_dir_path}")

    fp16_path = _resolve_weight(args.fp16, model_dir_path, "--fp16")
    int8_path = _resolve_weight(args.int8, model_dir_path, "--int8")
    vae_path = _resolve_weight(args.vae, model_dir_path, "--vae")
    if args.image is not None and not Path(args.image).is_file():
        raise FileNotFoundError(f"--image not found: {args.image}")

    # Enforce matching 3b/7b tags between FP16 and INT8 filenames.
    tag = _dit_size_tag(str(fp16_path), str(int8_path))
    print(f"[BENCH] DiT size tag: {tag}")

    model_dir = str(model_dir_path) if model_dir_path is not None else str(fp16_path.parent)
    model_dir_p = Path(model_dir)
    vae_name = vae_path.name
    int8_name = int8_path.name
    fp16_name = fp16_path.name

    for src, name in ((vae_path, vae_name), (int8_path, int8_name), (fp16_path, fp16_name)):
        target = model_dir_p / name
        if src.resolve() != target.resolve():
            if not target.is_file():
                raise FileNotFoundError(
                    f"{name} must live under --model_dir: expected {target}"
                )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[BENCH] python: {sys.executable}")
    print(f"[BENCH] layout: {_LAYOUT}")
    print(f"[BENCH] script_dir: {SCRIPT_DIR}")
    print(f"[BENCH] seedvr2_path: {args.seedvr2_path}")
    print(f"[BENCH] comfy_path: {args.comfy_path}")
    print(f"[BENCH] model_dir: {model_dir}")
    print("[BENCH] mode: native INT8 (construction-time mixed_precision_ops)")
    seed_root, comfy_root = _install_package_paths(
        seedvr2_path=args.seedvr2_path,
        comfy_path=args.comfy_path,
    )
    print(f"[BENCH] sys.path package roots: {seed_root} | {comfy_root}")

    from src.optimization.int8_native_ops import checkpoint_is_hswq_int8
    from src.utils.model_registry import DEFAULT_VAE as _DEFAULT_VAE

    if not checkpoint_is_hswq_int8(str(int8_path)):
        raise RuntimeError(
            f"--int8 does not look like HSWQ int8_tensorwise: {int8_path}"
        )
    print(f"  [BENCH] HSWQ INT8 marker OK: {int8_name}")

    if vae_name != _DEFAULT_VAE:
        print(
            f"  [BENCH] WARNING: videoupscaler CLI hardcodes VAE={_DEFAULT_VAE}; "
            f"--vae basename is {vae_name}. Ensure {_DEFAULT_VAE} exists under {model_dir}."
        )
        default_vae_path = model_dir_p / _DEFAULT_VAE
        if not default_vae_path.is_file():
            raise FileNotFoundError(
                f"Place {_DEFAULT_VAE} in model_dir ({model_dir}) "
                f"or rename --vae to match (found {vae_name})"
            )

    # --- input ---
    if args.image:
        print(f"Loading image: {args.image}")
        pil_in = Image.open(args.image).convert("RGB")
    else:
        print("No --image: using synthetic RGB pattern")
        pil_in = make_synthetic_rgb(short_edge=max(256, args.resolution // 2))

    frames = pil_to_thwc_f16(pil_in)
    print(f"  input tensor: {tuple(frames.shape)} dtype={frames.dtype}")

    ns = _build_cli_args(
        dit_model=fp16_name,
        model_dir=model_dir,
        resolution=args.resolution,
        seed=args.seed,
        color_correction=args.color,
        batch_size=args.batch_size,
        attention_mode=args.attention_mode,
        blocks_to_swap=args.blocks_to_swap,
        dit_offload_device=args.dit_offload_device,
        vae_offload_device=args.vae_offload_device,
        tensor_offload_device=args.tensor_offload_device,
    )

    img_fp16, t_fp16, v_fp16 = run_branch(
        label="FP16",
        dit_model=fp16_name,
        model_dir=model_dir,
        frames=frames,
        args_ns=ns,
    )
    out_fp16 = Path(args.output_dir) / "seedvr2_fp16.png"
    img_fp16.save(out_fp16)
    print(f"  saved: {out_fp16}")

    img_int8, t_int8, v_int8 = run_branch(
        label="INT8 (native QuantizedTensor)",
        dit_model=int8_name,
        model_dir=model_dir,
        frames=frames,
        args_ns=ns,
    )
    out_int8 = Path(args.output_dir) / "seedvr2_int8.png"
    img_int8.save(out_int8)
    print(f"  saved: {out_int8}")

    if img_fp16.size != img_int8.size:
        print(
            f"  [BENCH] size mismatch FP16={img_fp16.size} INT8={img_int8.size}; "
            "resizing INT8 to FP16 for metrics"
        )
        img_int8 = img_int8.resize(img_fp16.size, Image.Resampling.LANCZOS)

    mse, score = calculate_metrics(img_fp16, img_int8)
    diff = Image.fromarray(
        np.abs(np.asarray(img_fp16).astype(np.int16) - np.asarray(img_int8).astype(np.int16))
        .clip(0, 255)
        .astype(np.uint8)
    )
    out_diff = Path(args.output_dir) / "seedvr2_diff.png"
    diff.save(out_diff)

    print("\n=== Results (FP16 vs native INT8, same videoupscaler pipeline) ===")
    print(f"  MSE:  {mse:.6f}")
    print(f"  SSIM: {score:.6f}")
    print(f"  FP16 wall: {t_fp16:.2f}s  peak_vram={v_fp16:.2f} GiB")
    print(f"  INT8 wall: {t_int8:.2f}s  peak_vram={v_int8:.2f} GiB")
    print(f"  outputs: {out_fp16} | {out_int8} | {out_diff}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 修正ファイル

### `src/common/config.py`

```python
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

"""
Configuration utility functions
"""

import importlib
from typing import Any, Callable, List, Union
from omegaconf import DictConfig, ListConfig, OmegaConf
from ..utils.model_registry import MODEL_CLASSES

try:
    OmegaConf.register_new_resolver("eval", eval)
except Exception as e:
    if "already registered" not in str(e):
        raise



def load_config(path: str, argv: List[str] = None) -> Union[DictConfig, ListConfig]:
    """
    Load a configuration. Will resolve inheritance.
    """
    
    #print(path)
    config = OmegaConf.load(path)
    if argv is not None:
        config_argv = OmegaConf.from_dotlist(argv)
        config = OmegaConf.merge(config, config_argv)
    config = resolve_recursive(config, resolve_inheritance)
    return config


def resolve_recursive(
    config: Any,
    resolver: Callable[[Union[DictConfig, ListConfig]], Union[DictConfig, ListConfig]],
) -> Any:
    config = resolver(config)
    if isinstance(config, DictConfig):
        for k in config.keys():
            v = config.get(k)
            if isinstance(v, (DictConfig, ListConfig)):
                config[k] = resolve_recursive(v, resolver)
    if isinstance(config, ListConfig):
        for i in range(len(config)):
            v = config.get(i)
            if isinstance(v, (DictConfig, ListConfig)):
                config[i] = resolve_recursive(v, resolver)
    return config


def resolve_inheritance(config: Union[DictConfig, ListConfig]) -> Any:
    """
    Recursively resolve inheritance if the config contains:
    __inherit__: path/to/parent.yaml or a ListConfig of such paths.
    """
    if isinstance(config, DictConfig):
        inherit = config.pop("__inherit__", None)

        if inherit:
            inherit_list = inherit if isinstance(inherit, ListConfig) else [inherit]

            parent_config = None
            for parent_path in inherit_list:
                assert isinstance(parent_path, str)
                parent_config = (
                    load_config(parent_path)
                    if parent_config is None
                    else OmegaConf.merge(parent_config, load_config(parent_path))
                )

            if len(config.keys()) > 0:
                config = OmegaConf.merge(parent_config, config)
            else:
                config = parent_config
    return config


def import_item(path: str, name: str) -> Any:
    """
    Import a python item, checking model registry first.
    
    Args:
        path: Module path
        name: Class/function name to import
        
    Returns:
        Imported object
    """
    # Simple lookup with path as key
    if path in MODEL_CLASSES:
        return MODEL_CLASSES[path]
    
    # Fallback to dynamic import for everything else
    try:
        return getattr(importlib.import_module(path), name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import '{name}' from '{path}': {e}")


def create_object(config: DictConfig, **extra_kwargs) -> Any:
    """
    Create an object from config.
    The config is expected to contains the following:
    __object__:
      path: path.to.module
      name: MyClass
      args: as_config | as_params (default to as_config)

    ``extra_kwargs`` are merged at construction time only (e.g. ``operations``
    for ComfyUI ``comfy.ops`` HSWQ INT8 injection). Not stored in YAML.
    """
    
    item = import_item(
        path=config.__object__.path,
        name=config.__object__.name,
    )
    args = config.__object__.get("args", "as_config")
    if args == "as_config":
        return item(config, **extra_kwargs)
    if args == "as_params":
        config = OmegaConf.to_object(config)
        config.pop("__object__")
        config.update(extra_kwargs)
        return item(**config)
    raise NotImplementedError(f"Unknown args type: {args}")
```

### `src/core/model_loader.py`

```python
"""
Model Weight Loading for SeedVR2

This module handles all weight loading operations for DiT and VAE models:
- Loading state dictionaries from multiple formats (SafeTensors, PyTorch, GGUF)
- Materializing models from meta device to target device
- Applying weights with dtype conversion
- GGUF quantized model support with dequantization
- Meta buffer initialization for non-persistent buffers

Key Features:
- Multi-format support: .safetensors, .pth, .gguf files
- Memory-efficient loading with meta device initialization
- Native FP8 weight handling with optimal performance
- GGUF quantization support (Q4_K_M, Q8_0, etc.)
- Automatic dtype conversion for compatibility
- Meta buffer initialization post-materialization

Main Functions:
- load_quantized_state_dict: Load state dict from checkpoint file
- materialize_model: Move model from meta device and load weights
- prepare_model_structure: Create model structure on meta device

GGUF Support:
- apply_gguf_parameters: Apply GGUF weights to model (handles meta and materialized)
- _load_gguf_state: Load GGUF quantized weights from file
- _load_gguf_weights: Apply GGUF weights to model with validation
- _validate_gguf_architecture: Validate GGUF model architecture
- _create_dequantize_method: Create dequantization callable
- _create_gguf_parameter: Create parameter preserving quantization info
- _set_parameter_on_meta_model: Set parameter on meta device model
- _set_parameter_on_materialized_model: Set parameter on materialized model
- _navigate_to_parameter: Navigate to module containing parameter
- _get_tensor_shape: Get logical shape of tensor (handling GGUF)
- _is_quantized_tensor: Check if tensor is GGUF quantized
- _report_parameter_mismatches: Report parameter mismatches

Meta Buffer Initialization:
- initialize_meta_buffers: Initialize meta buffers with timing wrapper
- initialize_meta_buffers_impl: Initialize non-persistent buffers on target device

Standard Loading:
- _load_model_weights: Orchestrate weight loading process
- _load_standard_weights: Apply SafeTensors/PyTorch weights
- _convert_state_dtype: Convert weight dtypes
- _log_weight_stats: Log weight statistics

This module is used by model_configuration for weight loading during materialization.
"""

import os
import torch
from omegaconf import OmegaConf
from typing import Dict, Any, Optional, Tuple, Union, Callable

# Import SafeTensors with fallback
try:
    from safetensors.torch import load_file as load_safetensors_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

from .infer import VideoDiffusionInfer
from ..common.config import create_object
from ..optimization.int8_native_ops import (
    checkpoint_is_hswq_int8,
    get_hswq_mixed_precision_ops,
    patch_ops_factory_device,
    prepare_hswq_state_dict_for_comfy_ops,
)
from ..optimization.compatibility import (
    GGUF_AVAILABLE,
    GGMLQuantizationType,
    validate_gguf_availability
)

# GGUF-specific imports (only when available)
if GGUF_AVAILABLE:
    import gguf
    import traceback
    from ..optimization.gguf_dequant import dequantize_tensor
    from ..optimization.gguf_ops import replace_linear_with_quantized

from ..utils.constants import get_script_directory, suppress_tensor_warnings

# Get script directory for config paths
script_directory = get_script_directory()


def load_quantized_state_dict(checkpoint_path: str, device: torch.device = torch.device("cpu"),
                              debug: Optional['Debug'] = None) -> Dict[str, torch.Tensor]:
    """
    Load model state dict from checkpoint with support for multiple formats.
    
    Handles .safetensors, .gguf, and .pth files. GGUF models support quantization
    for memory-efficient loading. Validates required libraries are installed.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Target device for tensor placement (torch.device object, defaults to CPU)
        debug: Optional Debug instance for logging
        
    Returns:
        dict: State dictionary loaded with appropriate format handler
        
    Notes:
        - SafeTensors files use optimized loading with direct device placement
        - PyTorch files use memory-mapped loading to reduce RAM usage
    """
    device_str = str(device)
    
    if checkpoint_path.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            error_msg = (
                f"Cannot load {os.path.basename(checkpoint_path)}\n"
                f"SafeTensors library is required but not installed.\n"
                f"Please install it with: pip install safetensors"
            )
            if debug:
                debug.log(error_msg, level="ERROR", category="dit", force=True)
                debug.log("This is a one-time installation that will enable loading of .safetensors files", 
                         level="INFO", category="info", force=True)
            raise ImportError(error_msg)
        
        # Try direct device loading first (optimal path)
        try:
            state = load_safetensors_file(checkpoint_path, device=device_str)
        except RuntimeError as e:
            # MPS allocator fallback: some PyTorch/macOS versions have issues with
            # direct MPS loading (allocation failures, watermark errors, etc.)
            error_msg = str(e).lower()
            is_mps_alloc_error = device.type == "mps" and any(
                keyword in error_msg for keyword in ["watermark", "allocat", "memory"]
            )
            
            if is_mps_alloc_error:
                # Transparent fallback - only log if debug enabled
                if debug:
                    debug.log("Using CPU intermediate loading for MPS compatibility", 
                            category="info", indent_level=1)
                state = load_safetensors_file(checkpoint_path, device="cpu")
                # Tensors will be moved to MPS during model.load_state_dict()
            else:
                # Re-raise if it's a different error (file corruption, etc.)
                raise

    elif checkpoint_path.endswith('.gguf'):
        validate_gguf_availability(f"load {os.path.basename(checkpoint_path)}", debug)
        state = _load_gguf_state(
                    checkpoint_path=checkpoint_path, 
                    device=device, 
                    debug=debug, 
                    handle_prefix="model.diffusion_model."
                )
    elif checkpoint_path.endswith('.pth'):
        state = torch.load(checkpoint_path, map_location=device_str, mmap=True, weights_only=True)
    else:
        raise ValueError(f"Unsupported checkpoint format. Expected .safetensors or .pth, got: {checkpoint_path}")
    
    return state


def _load_gguf_state(checkpoint_path: str, device: torch.device, debug: Optional['Debug'] = None,
                    handle_prefix: str = "model.diffusion_model.") -> Dict[str, torch.Tensor]:
    """
    Load GGUF state dict
    
    Args:
        checkpoint_path: Path to GGUF file
        device: Target device (torch.device object)
        debug: Debug instance
        handle_prefix: Prefix to strip from tensor names
        
    Returns:
        State dictionary with loaded tensors
    """
    reader = gguf.GGUFReader(checkpoint_path)

    # Filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)
        
    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    state_dict = {}
    total_tensors = len(reader.tensors)
    
    device_str = str(device)
    debug.log(f"Loading {total_tensors} tensors to {str(device_str)}...", category="dit")
    
    # Suppress expected warnings: GGUF tensors are read-only numpy arrays that trigger warnings when converted
    suppress_tensor_warnings()
    
    for i, (sd_key, tensor) in enumerate(tensors):
        tensor_name = tensor.name
        
        # Create tensor directly on target device to avoid CPU->GPU copy overhead
        # For meta-initialized models, this directly materializes to the target device
        torch_tensor = torch.from_numpy(tensor.data).to(device, non_blocking=False)
            
        # Get original shape from metadata or infer from tensor shape
        shape = _get_tensor_logical_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            
        # Handle tensors based on quantization type
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # For unquantized tensors, just reshape
            torch_tensor = torch_tensor.view(*shape)
        else:
            # For quantized tensors, keep them quantized but track original shape
            torch_tensor = GGUFTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape, debug=debug)
            
        state_dict[sd_key] = torch_tensor
        
        # Progress reporting
        if (i + 1) % 100 == 0:
            debug.log(f"Loaded {i+1}/{total_tensors} tensors...", category="dit", indent_level=1)

    debug.log(f"Successfully loaded {len(state_dict)} tensors to {device_str}", category="success")

    return state_dict


def _get_tensor_logical_shape(reader: 'gguf.GGUFReader', tensor_name: str) -> Optional[torch.Size]:
    """
    Extract the logical (unquantized) shape from GGUF metadata
    """
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


class GGUFTensor(torch.Tensor):
    """
    Tensor wrapper for GGUF quantized tensors that preserves quantization info
    """
    def __init__(self, *args, tensor_type, tensor_shape, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        
    def __new__(cls, *args, tensor_type, tensor_shape, debug, **kwargs):
        # Create tensor with requires_grad=False to avoid gradient issues
        tensor = super().__new__(cls, *args, **kwargs)
        tensor.requires_grad_(False)
        tensor.tensor_type = tensor_type
        tensor.tensor_shape = tensor_shape
        tensor.debug = debug
        return tensor
    
    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", self.tensor_shape if hasattr(self, "tensor_shape") else new.shape)
        new.debug = getattr(self, "debug", None)
        new.requires_grad_(False)  # Ensure no gradients
        return new
    
    @property
    def shape(self):
        # Always return the logical tensor shape, not the quantized data shape
        if hasattr(self, "tensor_shape"):
            return self.tensor_shape
        else:
            # Fallback to actual data shape if tensor_shape is not available
            return self.size()
        
    def size(self, *args):
        # Override size() to also return logical shape
        if hasattr(self, "tensor_shape") and len(args) == 0:
            return self.tensor_shape
        elif hasattr(self, "tensor_shape") and len(args) == 1:
            return self.tensor_shape[args[0]]
        else:
            return super().size(*args)
        
    def dequantize(self, device=None, dtype=torch.float16, dequant_dtype=None):
        """Dequantize this tensor to its original shape"""
        if device is None:
            device = self.device
            
        # Suppress expected warning when converting from GGUFTensor subclass to regular tensor
        suppress_tensor_warnings()

        # Check if already unquantized
        if self.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # Return regular tensor, not GGUFTensor
            result = self.to(device, dtype)
            if isinstance(result, GGUFTensor):
                # Convert to regular tensor to avoid __torch_function__ calls
                result = torch.tensor(result, dtype=dtype, device=device, requires_grad=False)
            return result
        
        # Try fast dequantization with crash protection
        try:
            result = dequantize_tensor(self, dtype, dequant_dtype)
            final_result = result.to(device)
            
            # Ensure we return a regular tensor, not GGUFTensor
            if isinstance(final_result, GGUFTensor):
                final_result = torch.tensor(final_result.data, dtype=dtype, device=device, requires_grad=False)
                
            return final_result
        except Exception as e:
            self.debug.log(f"Fast dequantization failed: {e}", level="WARNING", category="dit", force=True)
            self.debug.log(f"Falling back to numpy dequantization", level="WARNING", category="dit", force=True)
            
        # Fallback to numpy (slower but reliable)
        try:
            numpy_data = self.cpu().numpy()
            dequantized = gguf.quants.dequantize(numpy_data, self.tensor_type)
            result = torch.from_numpy(dequantized).to(device, dtype)
            result.requires_grad_(False)
            final_result = result.reshape(self.tensor_shape)
            # from_numpy already returns a regular tensor, no conversion needed
            return final_result
        except Exception as e:
            self.debug.log(f"Numpy fallback also failed: {e}", level="WARNING", category="dit", force=True)
            self.debug.log(f"Tensor type: {self.tensor_type}", level="WARNING", category="dit", force=True, indent_level=1)
            self.debug.log(f"Shape: {self.shape}", level="WARNING", category="dit", force=True, indent_level=1)
            self.debug.log(f"Target shape: {self.tensor_shape}", level="WARNING", category="dit", force=True, indent_level=1)
            traceback.print_exc()
            
            # Return regular tensor as last resort
            result = self.to(device, dtype)
            if isinstance(result, GGUFTensor):
                result = torch.tensor(result.data, dtype=dtype, device=device, requires_grad=False)
            return result
        
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Override torch function calls to automatically dequantize"""
        if kwargs is None:
            kwargs = {}
        
        # Find the GGUFTensor instance(s) in args
        gguf_tensors = [arg for arg in args if isinstance(arg, cls)]
        if not gguf_tensors:
            return super().__torch_function__(func, types, args, kwargs)
        
        # Use the first GGUFTensor instance for attribute access
        self = gguf_tensors[0]
        
        # Check if the tensor is fully constructed and still quantized
        tensor_type = getattr(self, 'tensor_type', None)
        if tensor_type is None:
            # Tensor is either being constructed or already dequantized
            return super().__torch_function__(func, types, args, kwargs)
        
        # Check if tensor is already unquantized (F32/F16)
        if tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            return super().__torch_function__(func, types, args, kwargs)
        
        # Check if debug exists before using it
        debug = getattr(self, 'debug', None)
        
        # Handle linear operations specially
        if func == torch.nn.functional.linear:
            if len(args) >= 2 and isinstance(args[1], cls):  # weight is the second argument
                try:
                    weight_tensor = args[1]
                    dequantized_weight = weight_tensor.dequantize(device=args[0].device, dtype=args[0].dtype)
                    new_args = (args[0], dequantized_weight) + args[2:]
                    return func(*new_args, **kwargs)
                except Exception as e:
                    if debug:
                        debug.log(f"Error in linear dequantization: {e}", level="WARNING", category="dit", force=True)
                        debug.log(f"Function: {func}", level="WARNING", category="dit", force=True, indent_level=1)
                        debug.log(f"Args: {[arg.shape if hasattr(arg, 'shape') else type(arg) for arg in args]}", level="WARNING", category="dit", force=True, indent_level=1)
                    raise
        
        # Handle matrix multiplication operations that need dequantization
        if func in {torch.matmul, torch.mm, torch.bmm, torch.addmm, torch.addmv,
                    torch.addr, torch.baddbmm, torch.chain_matmul}:
            try:
                new_args = []
                for arg in args:
                    if isinstance(arg, cls):
                        new_args.append(arg.dequantize())
                    else:
                        new_args.append(arg)
                return func(*tuple(new_args), **kwargs)
            except Exception as e:
                if debug:
                    debug.log(f"Error in {func.__name__} dequantization: {e}", level="WARNING", category="dit", force=True)
                raise

        # Handle conv2d/conv3d operations (critical for GGUF VAE models)
        # Conv3d layers (InflatedCausalConv3d) are not replaced by layer replacement
        if func in {torch.nn.functional.conv2d, torch.nn.functional.conv3d}:
            if len(args) >= 2 and isinstance(args[1], cls):  # weight is second arg
                try:
                    weight_tensor = args[1]
                    dequantized_weight = weight_tensor.dequantize(device=args[0].device, dtype=args[0].dtype)
                    new_args = (args[0], dequantized_weight) + args[2:]
                    return func(*new_args, **kwargs)
                except Exception as e:
                    if debug:
                        debug.log(f"Error in conv dequantization: {e}", level="WARNING", category="dit", force=True)
                    raise
        
        # For ALL other operations, delegate to parent WITHOUT dequantization
        # This includes .cpu(), .to(), .device, .dtype, .shape, etc.
        return super().__torch_function__(func, types, args, kwargs)


def prepare_model_structure(
    runner: VideoDiffusionInfer,
    model_type: str,
    checkpoint_path: str,
    config: OmegaConf,
    debug: 'Debug',
    block_swap_config: Optional[Dict[str, Any]] = None
) -> VideoDiffusionInfer:
    """
    Prepare model structure on meta device without loading weights.
    This uses zero memory as meta device doesn't allocate real memory.
    
    Args:
        runner: VideoDiffusionInfer instance
        model_type: "dit" or "vae"
        checkpoint_path: Path to checkpoint (stored for later loading)
        config: Model configuration
        debug: Debug instance for logging (required)
        block_swap_config: BlockSwap config (stored for DiT, optional)
        
    Returns:
        runner: Updated runner with model structure on meta device
    """
    if debug is None:
        raise ValueError(f"Debug instance required for prepare_model_structure")
    
    is_dit = (model_type == "dit")
    model_type_upper = "DiT" if is_dit else "VAE"
    model_config = config.dit.model if is_dit else config.vae.model
    
    # Always create on meta device for zero memory usage
    debug.log(f"Creating {model_type_upper} model structure on meta device", 
             category=model_type, force=True)
    debug.start_timer(f"{model_type}_structure")

    # HSWQ INT8 safetensors need construction-time comfy.ops injection so
    # load_state_dict can hit _load_quantized_module (not post-load replace).
    create_kwargs = {}
    if is_dit and checkpoint_is_hswq_int8(checkpoint_path):
        create_kwargs["operations"] = get_hswq_mixed_precision_ops(torch.float16)
        debug.log(
            "HSWQ INT8 detected: injecting comfy.ops.mixed_precision_ops at DiT construction",
            category=model_type,
            force=True,
        )
    
    with torch.device("meta"):
        model = create_object(model_config, **create_kwargs)
    
    debug.end_timer(f"{model_type}_structure", f"{model_type_upper} structure created")
    
    # Store model and config for later materialization
    if is_dit:
        runner.dit = model
        runner._dit_checkpoint = checkpoint_path
        runner._dit_block_swap_config = block_swap_config
        runner._dit_hswq_int8_native = bool(create_kwargs)
    else:
        runner.vae = model  
        runner._vae_checkpoint = checkpoint_path
    
    return runner


def materialize_model(runner: VideoDiffusionInfer, model_type: str, device: torch.device, 
                     config: OmegaConf, debug: 'Debug') -> None:
    """
    Materialize model weights from checkpoint to memory.
    Call this right before the model is needed.
    
    Args:
        runner: Runner with model structure on meta device
        model_type: "dit" or "vae"
        device: Target device for inference (torch.device object)
        config: Full configuration
        debug: Debug instance
    """
    if debug is None:
        raise ValueError(f"Debug instance required for materialize_model")
        
    is_dit = (model_type == "dit")
    model_type_upper = "DiT" if is_dit else "VAE"
    
    # Get model and checkpoint path
    if is_dit:
        model = runner.dit
        checkpoint_path = runner._dit_checkpoint
        block_swap_config = runner._dit_block_swap_config
        override_dtype = getattr(runner, '_dit_dtype_override', None)
    else:
        model = runner.vae
        checkpoint_path = runner._vae_checkpoint
        block_swap_config = None
        override_dtype = getattr(runner, '_vae_dtype_override', None)
    
    # Check if already materialized
    if model is None:
        debug.log(f"No {model_type_upper} model structure found", level="WARNING", category=model_type, force=True)
        return
    param_device = next(model.parameters()).device
    if param_device.type != 'meta':
        debug.log(f"{model_type_upper} already materialized on {model.device}", category=model_type)
        return
    
    # Determine target device for materialization
    offload_device_str = None
    if hasattr(runner, f'_{model_type}_offload_device'):
        offload_device_str = getattr(runner, f'_{model_type}_offload_device')

    # If offload_device is set and not "none", materialize to offload device
    if offload_device_str and offload_device_str != "none":
        target_device = torch.device(offload_device_str)
        offload_reason = " (offload device)"
    else:
        # Otherwise materialize to inference device
        target_device = device
        offload_reason = ""
    
    # Start materialization
    debug.start_timer(f"{model_type}_materialize")
    
    # Load weights (this materializes from meta to target device)
    model = _load_model_weights(model, checkpoint_path, target_device, True,
                               model_type_upper, offload_reason, debug, override_dtype) 
   
    # Apply model-specific configurations (includes BlockSwap and torch.compile)
    # Import here to avoid circular dependency 
    from .model_configuration import apply_model_specific_config
    model = apply_model_specific_config(model, runner, config, is_dit, debug)
    
    debug.end_timer(f"{model_type}_materialize", f"{model_type_upper} materialized")
    
    # Clean up checkpoint paths (no longer needed after weights are loaded)
    # Note: Config attributes (_dit_block_swap_config, _dit_compile_args) are preserved
    # for configuration change detection on subsequent runs
    if is_dit:
        runner._dit_checkpoint = None
        runner._dit_dtype_override = None
    else:
        runner._vae_checkpoint = None
        runner._vae_dtype_override = None


def _load_model_weights(model: torch.nn.Module, checkpoint_path: str, target_device: torch.device, 
                        used_meta: bool, model_type: str, cpu_reason: str, 
                        debug: Optional['Debug'] = None, override_dtype: Optional[torch.dtype] = None) -> torch.nn.Module:
    """
    Load model weights from checkpoint file with optimized GGUF support.
    
    For meta-initialized models, materializes to target device.
    For standard models, loads weights and applies state dict.
    
    Args:
        model: Model instance (may be on meta device)
        checkpoint_path: Path to checkpoint file
        target_device: Target device for weights (torch.device object)
        used_meta: Whether model was created on meta device
        model_type: Model type string for logging
        cpu_reason: Reason string if using CPU
        debug: Debug instance
        override_dtype: Optional dtype override for weights
        
    Returns:
        Model with loaded weights
    """
    model_type_lower = model_type.lower()
    
    # Log loading action
    action = "Materializing" if used_meta else "Loading"
    target_device_str = str(target_device).upper()
    debug.log(f"{action} {model_type} weights to {target_device_str}{cpu_reason}: {checkpoint_path}", 
             category=model_type_lower, force=True)
    
    # Load state dict from file
    debug.start_timer(f"{model_type_lower}_weights_load")
    state = load_quantized_state_dict(checkpoint_path, target_device, debug)
    debug.end_timer(f"{model_type_lower}_weights_load", f"{model_type} weights loaded from file")

    # HSWQ INT8: comfy.ops parses comfy_quant via .numpy() (CPU only), and
    # meta-built mixed_precision Linear needs factory_kwargs["device"] set to
    # the materialization target so QuantizedTensor is not left on meta.
    if (
        model_type_lower == "dit"
        and not checkpoint_path.endswith(".gguf")
        and checkpoint_is_hswq_int8(checkpoint_path)
    ):
        prepare_hswq_state_dict_for_comfy_ops(state)
        n_patch = patch_ops_factory_device(model, target_device)
        debug.log(
            f"HSWQ INT8 load prep: comfy_quant→CPU, factory_kwargs device={target_device} "
            f"({n_patch} modules)",
            category=model_type_lower,
            force=True,
        )
    
    # Apply dtype conversion if requested
    if override_dtype is not None:
        state = _convert_state_dtype(state, override_dtype, model_type, debug)
    
    # Log weight statistics
    _log_weight_stats(state, used_meta, model_type, debug)
    
    # Handle GGUF or standard loading
    if checkpoint_path.endswith('.gguf'):
        model = _load_gguf_weights(model, state, used_meta, model_type_lower, debug)
    else:
        model = _load_standard_weights(model, state, used_meta, model_type, model_type_lower, debug)
    
    # Clean up state dict
    del state
    
    # Initialize meta buffers if needed
    if used_meta:
        initialize_meta_buffers(model, target_device, debug)
    
    return model


def _convert_state_dtype(state: Dict[str, torch.Tensor], target_dtype: torch.dtype, 
                        model_type: str, debug: Optional['Debug'] = None) -> Dict[str, torch.Tensor]:
    """Convert floating point tensors in state dict to target dtype."""
    debug.log(f"Converting {model_type} weights to {target_dtype} during loading", category="precision")
    debug.start_timer(f"{model_type.lower()}_dtype_convert")
    
    for key in state:
        if torch.is_tensor(state[key]) and state[key].is_floating_point():
            state[key] = state[key].to(target_dtype)
    
    debug.end_timer(f"{model_type.lower()}_dtype_convert", f"{model_type} weights converted to {target_dtype}")
    return state


def _log_weight_stats(state: Dict[str, torch.Tensor], used_meta: bool, model_type: str, debug: Optional['Debug'] = None) -> None:
    """Log statistics about loaded weights."""
    num_params = len(state)
    total_size_mb = sum(p.nelement() * p.element_size() for p in state.values()) / (1024 * 1024)
    action = "Materializing" if used_meta else "Applying"
    debug.log(f"{action} {model_type}: {num_params} parameters, {total_size_mb:.2f}MB total", 
             category=model_type.lower())


def apply_gguf_parameters(model: torch.nn.Module, state: Dict[str, torch.Tensor], 
                           model_state: Dict[str, torch.Tensor], debug: Optional['Debug'] = None) -> Dict[str, Any]:
    """
    Apply GGUF parameters to model, handling both meta and materialized models.
    
    Returns:
        Statistics dictionary with loaded count, quantized count, and parameter names
    """
    loaded_names = set()
    quantized_count = 0
    
    for name, param in state.items():
        if name not in model_state:
            continue
            
        model_param = model_state[name]
        param_shape = _get_tensor_shape(param)
        
        if param_shape != model_param.shape:
            debug.log(f"Unexpected shape mismatch for {name}: {param_shape} vs {model_param.shape}", 
                     level="ERROR", category="dit", force=True)
            raise ValueError(f"Shape mismatch for parameter {name}")
        
        # Apply parameter based on device type
        with torch.no_grad():
            if model_param.device.type == 'meta':
                _set_parameter_on_meta_model(model, name, param, debug)
            else:
                _set_parameter_on_materialized_model(model, name, param, debug)
        
        loaded_names.add(name)
        if _is_quantized_tensor(param):
            quantized_count += 1
    
    return {
        'loaded': len(loaded_names), 
        'quantized': quantized_count, 
        'loaded_names': loaded_names
    }


def _set_parameter_on_meta_model(model: torch.nn.Module, param_name: str, 
                                 param_value: torch.Tensor, debug: Optional['Debug'] = None) -> None:
    """Set parameter on meta device model."""
    module, attr_name = _navigate_to_parameter(model, param_name)
    new_param = _create_gguf_parameter(param_value, debug)
    setattr(module, attr_name, new_param)


def _set_parameter_on_materialized_model(model: torch.nn.Module, param_name: str, 
                                         param_value: torch.Tensor, debug: Optional['Debug'] = None) -> None:
    """Set parameter on already materialized model."""
    module, attr_name = _navigate_to_parameter(model, param_name)
    
    if _is_quantized_tensor(param_value):
        # For quantized tensors, replace with wrapped parameter
        new_param = _create_gguf_parameter(param_value, debug)
        setattr(module, attr_name, new_param)
    else:
        # For regular tensors, just copy
        existing_param = getattr(module, attr_name)
        existing_param.copy_(param_value)


def _navigate_to_parameter(model: torch.nn.Module, param_path: str) -> Tuple[torch.nn.Module, str]:
    """
    Navigate to the module containing a parameter.
    
    Args:
        model: Root model
        param_path: Dot-separated path to parameter
        
    Returns:
        Tuple of (parent module, parameter name)
    """
    path_parts = param_path.split('.')
    module = model
    
    # Navigate to parent module
    for part in path_parts[:-1]:
        module = getattr(module, part)
    
    return module, path_parts[-1]


def _create_gguf_parameter(tensor: torch.Tensor, debug: Optional['Debug'] = None) -> torch.nn.Parameter:
    """
    Create a parameter from a GGUF tensor, preserving quantization info.
    
    Args:
        tensor: GGUF tensor (may be quantized)
        debug: Debug instance for logging
        
    Returns:
        Parameter with GGUF attributes and dequantize method if quantized
    """
    param = torch.nn.Parameter(tensor, requires_grad=False)
    
    # Preserve GGUF attributes if present
    if hasattr(tensor, 'tensor_type'):
        param.tensor_type = tensor.tensor_type
        param.tensor_shape = tensor.tensor_shape
        
        # Add dequantize method for runtime dequantization
        param.gguf_dequantize = _create_dequantize_method(tensor, debug)
    
    return param


def _get_tensor_shape(tensor: torch.Tensor) -> torch.Size:
    """Get the logical shape of a tensor (handling GGUF quantized tensors)."""
    if hasattr(tensor, 'tensor_shape'):
        return tensor.tensor_shape
    return tensor.shape


def _is_quantized_tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is GGUF quantized."""
    return hasattr(tensor, 'tensor_type') and hasattr(tensor, 'tensor_shape')


def _report_parameter_mismatches(state: Dict[str, torch.Tensor], 
                                 model_state: Dict[str, torch.Tensor], 
                                 loaded_names: set, debug: Optional['Debug'] = None) -> None:
    """Report any parameter mismatches between GGUF and model."""
    # Check for unmatched GGUF parameters
    unmatched = [name for name in state if name not in model_state]
    if unmatched:
        debug.log(f"Warning: {len(unmatched)} parameters from GGUF not found in model", 
                 level="WARNING", category="dit", force=True)
        debug.log(f"First few unmatched: {unmatched[:5]}", level="WARNING", category="dit", force=True, indent_level=1)
    
    # Check for missing model parameters  
    missing = [name for name in model_state if name not in loaded_names]
    if missing:
        debug.log(f"Warning: {len(missing)} model parameters not loaded from GGUF", 
                 level="WARNING", category="dit", force=True)
        debug.log(f"First few missing: {missing[:5]}", level="WARNING", category="dit", force=True, indent_level=1)


def initialize_meta_buffers(model: torch.nn.Module, target_device: torch.device, debug: Optional['Debug'] = None) -> None:
    """Initialize meta buffers with timing."""
    debug.start_timer("buffer_init")
    initialized = initialize_meta_buffers_impl(model, target_device, debug)
    if initialized > 0:
        debug.log(f"Initialized {initialized} non-persistent buffers", category="success")
    debug.end_timer("buffer_init", "Buffer initialization")


def initialize_meta_buffers_impl(model: torch.nn.Module, target_device: torch.device, debug: Optional['Debug'] = None) -> int:
    """
    Initialize any buffers still on meta device after materialization.
    
    Non-persistent buffers aren't included in state_dict and remain on meta
    device after load_state_dict. This function moves them to the target device.
    
    Args:
        model: Model potentially containing meta device buffers
        target_device: Target device for initialization (torch.device object)
        debug: Debug instance for logging
        
    Returns:
        Number of buffers initialized
    """
    initialized_count = 0
    
    # Simply initialize all meta device buffers to zeros on target device
    for name, buffer in model.named_buffers():
        if buffer is not None and buffer.device.type == 'meta':
            # Get the module that owns this buffer
            module_path = name.rsplit('.', 1)[0] if '.' in name else ''
            buffer_name = name.rsplit('.', 1)[1] if '.' in name else name
            
            # Get the actual module
            if module_path:
                module = model
                for part in module_path.split('.'):
                    module = getattr(module, part)
            else:
                module = model
            
            # Create a zero tensor of the same shape on target device
            # This is safe for all non-persistent buffers (caches, dummy tensors, etc.)
            initialized_buffer = torch.zeros_like(buffer, device=target_device)
            module.register_buffer(buffer_name, initialized_buffer, persistent=False)
            initialized_count += 1
    
    return initialized_count


def _load_standard_weights(model: torch.nn.Module, state: Dict[str, torch.Tensor], 
                          used_meta: bool, model_type: str, model_type_lower: str,
                          debug: Optional['Debug'] = None) -> torch.nn.Module:
    """Load standard (non-GGUF) weights into model."""
    debug.start_timer(f"{model_type_lower}_state_apply")
    model.load_state_dict(state, strict=False, assign=True)
    
    action = "materialized" if used_meta else "applied"
    debug.end_timer(f"{model_type_lower}_state_apply", f"{model_type} weights {action}")
    
    if used_meta:
        debug.log(f"{model_type} materialized directly from meta with loaded weights", category=model_type_lower)
    else:
        debug.log(f"{model_type} weights applied", category=model_type_lower)
    
    return model


def _load_gguf_weights(model: torch.nn.Module, state: Dict[str, torch.Tensor], 
                      used_meta: bool, model_type_lower: str, debug: Optional['Debug'] = None) -> torch.nn.Module:
    """
    Load GGUF quantized weights into model with architecture validation.
    
    Args:
        model: Target model
        state: GGUF state dict with quantized tensors
        used_meta: Whether model was initialized on meta device
        model_type_lower: Lowercase model type for logging
        debug: Debug instance
        
    Returns:
        Model with GGUF weights loaded
    """
    debug.log("Loading GGUF weights", category="dit")
    
    # Get model state dict for validation
    model_state = model.state_dict()
    
    # Validate architecture compatibility
    _validate_gguf_architecture(state, model_state, debug)
    
    # Load GGUF parameters
    stats = apply_gguf_parameters(model, state, model_state, debug)
    
    # Log results
    debug.log(f"GGUF loading complete: {stats['loaded']} parameters loaded", category="success")
    debug.log(f"Quantized parameters: {stats['quantized']}", category="info")
    
    # Report any mismatches
    _report_parameter_mismatches(state, model_state, stats['loaded_names'], debug)
    
    # Replace Linear/Conv2d layers with quantized versions for optimal precision handling
    if stats['quantized'] > 0:
        debug.log("Replacing layers with GGUF-optimized versions for precision handling", category="dit")
        
        replacements, quant_types = replace_linear_with_quantized(model, debug=debug)
        
        if replacements > 0:
            debug.log(f"Replaced {replacements} layers with GGUF-optimized versions", category="success")
            
            # Show actual quantization types found and precision strategy
            if quant_types:
                qtypes_str = ', '.join([f"{qtype}:{count}" for qtype, count in quant_types.items()])
                debug.log(
                    f"GGUF precision path: {qtypes_str} → FP16 (preserve) → BF16/FP32 (compute)", 
                    category="precision"
                )
            else:
                debug.log(
                    "GGUF precision: Dequantizing to FP16 first, then converting to compute dtype", 
                    category="precision"
                )
        else:
            debug.log("Warning: No layers were replaced despite having quantized parameters", 
                     level="WARNING", category="dit", force=True)
    
    return model


def _validate_gguf_architecture(state: Dict[str, torch.Tensor], 
                                model_state: Dict[str, torch.Tensor], debug: Optional['Debug'] = None) -> None:
    """
    Validate GGUF model architecture matches target model.
    
    Raises:
        ValueError: If architecture mismatch is detected
    """
    key_params = [
        "blocks.0.attn.proj_qkv.vid.weight",
        "blocks.0.attn.proj_qkv.txt.weight", 
        "blocks.0.mlp.vid.proj_in.weight"
    ]

    for key in key_params:
        if key in state and key in model_state:
            model_shape = model_state[key].shape
            gguf_shape = _get_tensor_shape(state[key])
            
            if model_shape != gguf_shape:
                # Check if it's just a quantization difference
                if hasattr(state[key], 'tensor_shape') and state[key].tensor_shape == model_shape:
                    continue
                    
                raise ValueError(
                    f"GGUF model architecture mismatch: This GGUF model is incompatible with the current architecture.\n\n"
                    f"Detected mismatch:\n"
                    f"  Parameter: {key}\n"
                    f"  Expected shape: {model_shape}\n"
                    f"  GGUF shape: {gguf_shape}\n\n"
                    f"Possible solutions:\n"
                    f"1. Use a GGUF model that matches the current architecture\n"
                    f"2. Try using a regular FP16 model instead\n"
                    f"3. Verify you're using the correct model variant (3B vs 7B)"
                )
    
    debug.log(f"Architecture check complete, no shape mismatch", category="success")


def _create_dequantize_method(tensor: torch.Tensor, debug: Optional['Debug'] = None) -> callable:
    """
    Create a dequantization method for a GGUF tensor.
    
    Args:
        tensor: GGUF quantized tensor with tensor_type and tensor_shape attributes
        debug: Debug instance
        
    Returns:
        Callable dequantization method
    """
    def dequantize(device: Optional[torch.device] = None, 
                   dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Dequantize GGUF tensor on demand."""
        if hasattr(tensor, 'dequantize'):
            return tensor.dequantize(device, dtype)
        
        try:
            # Fallback to manual dequantization using gguf library
            numpy_data = tensor.cpu().numpy()
            dequantized = gguf.quants.dequantize(numpy_data, tensor.tensor_type)
            result = torch.from_numpy(dequantized).to(device or tensor.device, dtype)
            result.requires_grad_(False)
            return result.reshape(tensor.tensor_shape)
        except Exception as e:
            if debug:
                debug.log(f"Warning: Could not dequantize tensor: {e}", level="WARNING", category="dit", force=True)
            return tensor.to(device or tensor.device, dtype)
    
    return dequantize
```

### `src/utils/model_registry.py`

```python
"""
Model Registry for SeedVR2
Central registry for model definitions, repositories, and metadata
"""

import os
from typing import List, Optional
from dataclasses import dataclass
from .constants import get_all_model_files

# Model class imports using relative imports
from ..models.dit_3b.nadit import NaDiT as NaDiT3B
from ..models.dit_7b.nadit import NaDiT as NaDiT7B
from ..models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper

# Model classes - simple registry with clear keys
MODEL_CLASSES = {
    "dit_3b.nadit": NaDiT3B,
    "dit_7b.nadit": NaDiT7B,
    "video_vae_v3.modules.attn_video_vae": VideoAutoencoderKLWrapper,
}

@dataclass
class ModelInfo:
    """Model metadata"""
    repo: str = "numz/SeedVR2_comfyUI"
    category: str = "dit" # 'model' or 'vae'
    precision: str = "fp16" # 'fp16', 'fp8_e4m3fn', 'Q4_K_M', etc.
    size: str = "3B" # '3B', '7B', etc.
    variant: Optional[str] = None # 'sharp', etc.
    sha256: Optional[str] = None # Cached hash

# Model registry with metadata
MODEL_REGISTRY = {
    # 3B models
    "seedvr2_ema_3b-Q4_K_M.gguf": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="3B", precision="Q4_K_M", sha256="e665e3909de1a8c88a69c609bca9d43ff5a134647face2ce4497640cc3597f0e"),
    "seedvr2_ema_3b-Q8_0.gguf": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="3B", precision="Q8_0", sha256="be0d60083a2051a265eb4b77f28edf494e6db67ffc250216f32b72292e5cbd96"),
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors": ModelInfo(size="3B", precision="fp8_e4m3fn", sha256="3bf1e43ebedd570e7e7a0b1b60d6a02e105978f505c8128a241cde99a8240cff"),
    "seedvr2_ema_3b_fp16.safetensors": ModelInfo(size="3B", precision="fp16", sha256="2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304"),
    
    # 7B models
    "seedvr2_ema_7b-Q4_K_M.gguf": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="Q4_K_M", sha256="db9cb2ad90ebd40d2e8c29da2b3fc6fd03ba87cd58cbadceccca13ad27162789"),
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="fp8_e4m3fn_mixed_block35_fp16", sha256="3d68b5ec0b295ae28092e355c8cad870edd00b817b26587d0cb8f9dd2df19bb2"),
    "seedvr2_ema_7b_fp16.safetensors": ModelInfo(size="7B", precision="fp16", sha256="7b8241aa957606ab6cfb66edabc96d43234f9819c5392b44d2492d9f0b0bbe4a"),
    # HSWQ INT8 (int8_tensorwise + ConvRot) — native INT8 inference target (VRAM-saving path)
    "seedvr2_7b_int8_convrot.safetensors": ModelInfo(size="7B", precision="int8_tensorwise_convrot"),
    
    # 7B sharp variants
    "seedvr2_ema_7b_sharp-Q4_K_M.gguf": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="Q4_K_M", variant="sharp", sha256="7aed800ac4eb8e0d18569a954c0ff35f5a1caa3ed5d920e66cc31405f75b6e69"),
    "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors": ModelInfo(repo="AInVFX/SeedVR2_comfyUI", size="7B", precision="fp8_e4m3fn_mixed_block35_fp16", variant="sharp", sha256="0d2c5b8be0fda94351149c5115da26aef4f4932a7a2a928c6f184dda9186e0be"),
    "seedvr2_ema_7b_sharp_fp16.safetensors": ModelInfo(size="7B", precision="fp16", variant="sharp", sha256="20a93e01ff24beaeebc5de4e4e5be924359606c356c9c51509fba245bd2d77dd"),
    "seedvr2_7b_sharp_int8_convrot.safetensors": ModelInfo(size="7B", precision="int8_tensorwise_convrot", variant="sharp"),
    
    # VAE models
    "ema_vae_fp16.safetensors": ModelInfo(category="vae", precision="fp16", sha256="20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"),
}

# Configuration constants
DEFAULT_DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
DEFAULT_VAE = "ema_vae_fp16.safetensors"

def get_default_models(category: str) -> List[str]:
    """Get list of default models"""
    return [name for name, info in MODEL_REGISTRY.items() if info.category == category]

def get_model_repo(model_name: str) -> str:
    """Get repository for a specific model"""
    return MODEL_REGISTRY.get(model_name, ModelInfo()).repo

def resolve_dit_config_folder(dit_model: str) -> str:
    """
    Resolve configs_7b vs configs_3b from registry size and/or filename.

    Filename substring \"7b\"/\"3b\" is the historical rule. Registry size is used
    when the model is registered (including HSWQ INT8 names). Prefer explicit
    7b/3b tokens in the basename so untagged temp names do not silently pick 3B.
    """
    info = MODEL_REGISTRY.get(dit_model)
    if info is not None and info.category == "dit":
        size = (info.size or "").upper()
        if size == "7B":
            return "configs_7b"
        if size == "3B":
            return "configs_3b"

    name = dit_model.lower()
    if "7b" in name:
        return "configs_7b"
    if "3b" in name:
        return "configs_3b"
    return "configs_3b"

def get_available_dit_models() -> List[str]:
    """Get all available DiT models including those discovered on disk"""
    model_list = get_default_models("dit")
    
    try:
        # Get all model files from all paths
        model_files = get_all_model_files()
        
        # Add files not in registry
        discovered_models = [
            filename for filename in model_files
            if filename not in MODEL_REGISTRY
        ]
        
        # Add discovered models to the list
        model_list.extend(sorted(discovered_models))
    except:
        pass
    
    return model_list

def get_available_vae_models() -> List[str]:
    """Get all available VAE models from the registry"""
    model_list = get_default_models("vae")
    return model_list
```

### `src/models/dit_3b/nadit.py`

```python
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Callable
import torch
from torch import nn

from ...common.cache import Cache
from ...common.distributed.ops import slice_inputs

from . import na
from .embedding import TimeEmbedding
from .modulation import get_ada_layer
from .nablocks import get_nablock
from .normalization import get_norm_layer
from .patch import get_na_patch_layers

# Fake func, no checkpointing is required for inference
def gradient_checkpointing(module: Union[Callable, nn.Module], *args, enabled: bool, **kwargs):
    return module(*args, **kwargs)

@dataclass
class NaDiTOutput:
    vid_sample: torch.Tensor


class NaDiT(nn.Module):
    """
    Native Resolution Diffusion Transformer (NaDiT)
    """

    gradient_checkpointing = False

    def __init__(
        self,
        vid_in_channels: int,
        vid_out_channels: int,
        vid_dim: int,
        txt_in_dim: Union[int, List[int]],
        txt_dim: Optional[int],
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: Optional[str],
        norm_eps: float,
        ada: str,
        qk_bias: bool,
        qk_norm: Optional[str],
        patch_size: Union[int, Tuple[int, int, int]],
        num_layers: int,
        block_type: Union[str, Tuple[str]],
        mm_layers: Union[int, Tuple[bool]],
        mlp_type: str = "normal",
        patch_type: str = "v1",
        rope_type: Optional[str] = "rope3d",
        rope_dim: Optional[int] = None,
        window: Optional[Tuple] = None,
        window_method: Optional[Tuple[str]] = None,
        msa_type: Optional[Tuple[str]] = None,
        mca_type: Optional[Tuple[str]] = None,
        txt_in_norm: Optional[str] = None,
        txt_in_norm_scale_factor: int = 0.01,
        txt_proj_type: Optional[str] = "linear",
        vid_out_norm: Optional[str] = None,
        attention_mode: str = 'sdpa',
        operations=None,
        **kwargs,
    ):
        ada = get_ada_layer(ada)
        norm = get_norm_layer(norm)
        qk_norm = get_norm_layer(qk_norm)
        rope_dim = rope_dim if rope_dim is not None else head_dim // 2
        if isinstance(block_type, str):
            block_type = [block_type] * num_layers
        elif len(block_type) != num_layers:
            raise ValueError("The ``block_type`` list should equal to ``num_layers``.")
        super().__init__()
        ops = operations if operations is not None else nn
        NaPatchIn, NaPatchOut = get_na_patch_layers(patch_type)
        self.vid_in = NaPatchIn(
            in_channels=vid_in_channels,
            patch_size=patch_size,
            dim=vid_dim,
            operations=operations,
        )
        if not isinstance(txt_in_dim, int):
            self.txt_in = nn.ModuleList([])
            for in_dim in txt_in_dim:
                txt_norm_layer = get_norm_layer(txt_in_norm)(txt_dim, norm_eps, True)
                if txt_proj_type == "linear":
                    txt_proj_layer = ops.Linear(in_dim, txt_dim)
                else:
                    txt_proj_layer = nn.Sequential(
                        ops.Linear(in_dim, in_dim), nn.GELU("tanh"), ops.Linear(in_dim, txt_dim)
                    )
                torch.nn.init.constant_(txt_norm_layer.weight, txt_in_norm_scale_factor)
                self.txt_in.append(
                    nn.Sequential(
                        txt_proj_layer,
                        txt_norm_layer,
                    )
                )
        else:
            self.txt_in = (
                ops.Linear(txt_in_dim, txt_dim)
                if txt_in_dim and txt_in_dim != txt_dim
                else nn.Identity()
            )
        self.emb_in = TimeEmbedding(
            sinusoidal_dim=256,
            hidden_dim=max(vid_dim, txt_dim),
            output_dim=emb_dim,
            operations=operations,
        )

        if window is None or isinstance(window[0], int):
            window = [window] * num_layers
        if window_method is None or isinstance(window_method, str):
            window_method = [window_method] * num_layers

        if msa_type is None or isinstance(msa_type, str):
            msa_type = [msa_type] * num_layers
        if mca_type is None or isinstance(mca_type, str):
            mca_type = [mca_type] * num_layers

        self.blocks = nn.ModuleList(
            [
                get_nablock(block_type[i])(
                    vid_dim=vid_dim,
                    txt_dim=txt_dim,
                    emb_dim=emb_dim,
                    heads=heads,
                    head_dim=head_dim,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    norm_eps=norm_eps,
                    ada=ada,
                    qk_bias=qk_bias,
                    qk_norm=qk_norm,
                    shared_weights=not (
                        (i < mm_layers) if isinstance(mm_layers, int) else mm_layers[i]
                    ),
                    mlp_type=mlp_type,
                    window=window[i],
                    window_method=window_method[i],
                    msa_type=msa_type[i],
                    mca_type=mca_type[i],
                    rope_type=rope_type,
                    rope_dim=rope_dim,
                    is_last_layer=(i == num_layers - 1),
                    attention_mode=attention_mode,
                    operations=operations,
                    **kwargs,
                )
                for i in range(num_layers)
            ]
        )

        self.vid_out_norm = None
        if vid_out_norm is not None:
            self.vid_out_norm = get_norm_layer(vid_out_norm)(
                dim=vid_dim,
                eps=norm_eps,
                elementwise_affine=True,
            )
            self.vid_out_ada = ada(
                dim=vid_dim,
                emb_dim=emb_dim,
                layers=["out"],
                modes=["in"],
            )

        self.vid_out = NaPatchOut(
            out_channels=vid_out_channels,
            patch_size=patch_size,
            dim=vid_dim,
            operations=operations,
        )

    def set_gradient_checkpointing(self, enable: bool):
        self.gradient_checkpointing = enable

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: Union[torch.FloatTensor, List[torch.FloatTensor]],  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: Union[torch.LongTensor, List[torch.LongTensor]],  # b 1
        timestep: Union[int, float, torch.IntTensor, torch.FloatTensor],  # b
        disable_cache: bool = False,  # for test
    ):
        cache = Cache(disable=disable_cache)

        # slice vid after patching in when using sequence parallelism
        if isinstance(txt, list):
            assert isinstance(self.txt_in, nn.ModuleList)
            txt = [
                na.unflatten(fc(i), s) for fc, i, s in zip(self.txt_in, txt, txt_shape)
            ]  # B L D
            txt, txt_shape = na.flatten([torch.cat(t, dim=0) for t in zip(*txt)])
            txt = slice_inputs(txt, dim=0)
        else:
            txt = slice_inputs(txt, dim=0)
            txt = self.txt_in(txt)

        # Video input.
        # Sequence parallel slicing is done inside patching class.
        vid, vid_shape = self.vid_in(vid, vid_shape, cache)

        # Embedding input.
        emb = self.emb_in(timestep, device=vid.device, dtype=vid.dtype)

        # Body
        for i, block in enumerate(self.blocks):
            vid, txt, vid_shape, txt_shape = gradient_checkpointing(
                enabled=(self.gradient_checkpointing and self.training),
                module=block,
                vid=vid,
                txt=txt,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                emb=emb,
                cache=cache,
            )

        # Video output norm.
        if self.vid_out_norm:
            vid = self.vid_out_norm(vid)
            vid = self.vid_out_ada(
                vid,
                emb=emb,
                layer="out",
                mode="in",
                hid_len=cache("vid_len", lambda: vid_shape.prod(-1)),
                cache=cache,
                branch_tag="vid",
            )

        # Video output.
        vid, vid_shape = self.vid_out(vid, vid_shape, cache)
        return NaDiTOutput(vid_sample=vid)
```

### `src/models/dit_3b/mlp.py`

```python
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn


def get_mlp(mlp_type: Optional[str] = "normal"):
    if mlp_type == "normal":
        return MLP
    elif mlp_type == "swiglu":
        return SwiGLUMLP


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int,
        operations=None,
    ):
        super().__init__()
        ops = operations if operations is not None else nn
        self.proj_in = ops.Linear(dim, dim * expand_ratio)
        self.act = nn.GELU("tanh")
        self.proj_out = ops.Linear(dim * expand_ratio, dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.proj_in(x)
        x = self.act(x)
        x = self.proj_out(x)
        return x


class SwiGLUMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int,
        multiple_of: int = 256,
        operations=None,
    ):
        super().__init__()
        ops = operations if operations is not None else nn
        hidden_dim = int(2 * dim * expand_ratio / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.proj_in_gate = ops.Linear(dim, hidden_dim, bias=False)
        self.proj_out = ops.Linear(hidden_dim, dim, bias=False)
        self.proj_in = ops.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.proj_out(F.silu(self.proj_in_gate(x)) * self.proj_in(x))
        return x
```

### `src/models/dit_3b/embedding.py`

```python
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Optional, Union
import torch
from diffusers.models.embeddings import get_timestep_embedding
from torch import nn


def emb_add(emb1: torch.Tensor, emb2: Optional[torch.Tensor]):
    return emb1 if emb2 is None else emb1 + emb2


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        sinusoidal_dim: int,
        hidden_dim: int,
        output_dim: int,
        operations=None,
    ):
        super().__init__()
        ops = operations if operations is not None else nn
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = ops.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = ops.Linear(hidden_dim, hidden_dim)
        self.proj_out = ops.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()

    def forward(
        self,
        timestep: Union[int, float, torch.IntTensor, torch.FloatTensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.FloatTensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=dtype)
        if timestep.ndim == 0:
            timestep = timestep[None]

        emb = get_timestep_embedding(
            timesteps=timestep,
            embedding_dim=self.sinusoidal_dim,
            flip_sin_to_cos=False,
            downscale_freq_shift=0,
        )
        emb = emb.to(dtype)
        emb = self.proj_in(emb)
        emb = self.act(emb)
        emb = self.proj_hid(emb)
        emb = self.act(emb)
        emb = self.proj_out(emb)
        return emb
```

### `src/models/dit_3b/patch/patch_v1.py`

```python
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Tuple, Union
import torch
from einops import rearrange
from torch import nn
from torch.nn.modules.utils import _triple

from ....common.cache import Cache
from ....common.distributed.ops import gather_outputs, slice_inputs

from .. import na


class PatchIn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: Union[int, Tuple[int, int, int]],
        dim: int,
        operations=None,
    ):
        super().__init__()
        ops = operations if operations is not None else nn
        t, h, w = _triple(patch_size)
        self.patch_size = t, h, w
        self.proj = ops.Linear(in_channels * t * h * w, dim)

    def forward(
        self,
        vid: torch.Tensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        if t > 1:
            assert vid.size(2) % t == 1
            vid = torch.cat([vid[:, :, :1]] * (t - 1) + [vid], dim=2)
        vid = rearrange(vid, "b c (T t) (H h) (W w) -> b T H W (t h w c)", t=t, h=h, w=w)
        vid = self.proj(vid)
        return vid


class PatchOut(nn.Module):
    def __init__(
        self,
        out_channels: int,
        patch_size: Union[int, Tuple[int, int, int]],
        dim: int,
        operations=None,
    ):
        super().__init__()
        ops = operations if operations is not None else nn
        t, h, w = _triple(patch_size)
        self.patch_size = t, h, w
        self.proj = ops.Linear(dim, out_channels * t * h * w)

    def forward(
        self,
        vid: torch.Tensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        vid = self.proj(vid)
        vid = rearrange(vid, "b T H W (t h w c) -> b c (T t) (H h) (W w)", t=t, h=h, w=w)
        if t > 1:
            vid = vid[:, :, (t - 1) :]
        return vid


class NaPatchIn(PatchIn):
    def forward(
        self,
        vid: torch.Tensor,  # l c
        vid_shape: torch.LongTensor,
        cache: Cache = Cache(disable=True),  # for test
    ) -> torch.Tensor:
        cache = cache.namespace("patch")
        vid_shape_before_patchify = cache("vid_shape_before_patchify", lambda: vid_shape)
        t, h, w = self.patch_size
        if not (t == h == w == 1):
            vid = na.unflatten(vid, vid_shape)
            for i in range(len(vid)):
                if t > 1 and vid_shape_before_patchify[i, 0] % t != 0:
                    vid[i] = torch.cat([vid[i][:1]] * (t - vid[i].size(0) % t) + [vid[i]], dim=0)
                vid[i] = rearrange(vid[i], "(T t) (H h) (W w) c -> T H W (t h w c)", t=t, h=h, w=w)
            vid, vid_shape = na.flatten(vid)

        # slice vid after patching in when using sequence parallelism
        vid = slice_inputs(vid, dim=0)
        vid = self.proj(vid)
        return vid, vid_shape


class NaPatchOut(PatchOut):
    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,
        cache: Cache = Cache(disable=True),  # for test
    ) -> Tuple[
        torch.FloatTensor,
        torch.LongTensor,
    ]:
        cache = cache.namespace("patch")
        vid_shape_before_patchify = cache.get("vid_shape_before_patchify")

        t, h, w = self.patch_size
        vid = self.proj(vid)
        # gather vid before patching out when enabling sequence parallelism
        vid = gather_outputs(
            vid, gather_dim=0, padding_dim=0, unpad_shape=vid_shape, cache=cache.namespace("vid")
        )
        if not (t == h == w == 1):
            vid = na.unflatten(vid, vid_shape)
            for i in range(len(vid)):
                vid[i] = rearrange(vid[i], "T H W (t h w c) -> (T t) (H h) (W w) c", t=t, h=h, w=w)
                if t > 1 and vid_shape_before_patchify[i, 0] % t != 0:
                    vid[i] = vid[i][(t - vid_shape_before_patchify[i, 0] % t) :]
            vid, vid_shape = na.flatten(vid)

        return vid, vid_shape
```

### `src/models/dit_3b/nablocks/mmsr_block.py`

```python
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Tuple
import torch
import torch.nn as nn

# from ..cache import Cache
from ....common.cache import Cache

from .attention.mmattn import NaSwinAttention
from ..mm import MMArg
from ..modulation import ada_layer_type
from ..normalization import norm_layer_type
from ..mm import MMArg, MMModule
from ..mlp import get_mlp
    

class NaMMSRTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: norm_layer_type,
        norm_eps: float,
        ada: ada_layer_type,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        mlp_type: str,
        shared_weights: bool,
        rope_type: str,
        rope_dim: int,
        is_last_layer: bool,
        attention_mode: str = 'sdpa',
        operations=None,
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        self.attn_norm = MMModule(norm, dim=dim, eps=norm_eps, elementwise_affine=False, shared_weights=shared_weights,)

        self.attn = NaSwinAttention(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_norm=qk_norm,
            qk_norm_eps=norm_eps,
            rope_type=rope_type,
            rope_dim=rope_dim,
            shared_weights=shared_weights,
            attention_mode=attention_mode,
            operations=operations,
            window=kwargs.pop("window", None),
            window_method=kwargs.pop("window_method", None),
        )

        self.mlp_norm = MMModule(norm, dim=dim, eps=norm_eps, elementwise_affine=False, shared_weights=shared_weights, vid_only=is_last_layer)
        self.mlp = MMModule(
            get_mlp(mlp_type),
            dim=dim,
            expand_ratio=expand_ratio,
            shared_weights=shared_weights,
            vid_only=is_last_layer,
            operations=operations,
        )
        self.ada = MMModule(ada, dim=dim, emb_dim=emb_dim, layers=["attn", "mlp"], shared_weights=shared_weights, vid_only=is_last_layer)
        self.is_last_layer = is_last_layer

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        emb: torch.FloatTensor,
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
    ]:
        hid_len = MMArg(
            cache("vid_len", lambda: vid_shape.prod(-1)),
            cache("txt_len", lambda: txt_shape.prod(-1)),
        )
        ada_kwargs = {
            "emb": emb,
            "hid_len": hid_len,
            "cache": cache,
            "branch_tag": MMArg("vid", "txt"),
        }

        vid_attn, txt_attn = self.attn_norm(vid, txt)

        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="in", **ada_kwargs)
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, vid_shape, txt_shape, cache)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="out", **ada_kwargs)
        vid_attn, txt_attn = (vid_attn + vid), (txt_attn + txt)

        vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
        # ADD BY NUMZ
        if vid_mlp.dtype != vid_attn.dtype:
            vid_mlp = vid_mlp.to(vid_attn.dtype)
        if txt_mlp.dtype != txt_attn.dtype:
            txt_mlp = txt_mlp.to(txt_attn.dtype)
        # END BY NUMZ
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="in", **ada_kwargs)
        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="out", **ada_kwargs)
        vid_mlp, txt_mlp = (vid_mlp + vid_attn), (txt_mlp + txt_attn)

        return vid_mlp, txt_mlp, vid_shape, txt_shape
```

### `src/models/dit_3b/nablocks/attention/mmattn.py`

```python
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Optional, Tuple, Union
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _triple

from .....common.cache import Cache
from .....common.distributed.ops import gather_heads_scatter_seq, gather_seq_scatter_heads_qkv
from .....common.half_precision_fixes import safe_pad_operation

from ... import na
from ...attention import FlashAttentionVarlen
from ...mm import MMArg, MMModule
from ...normalization import norm_layer_type
from ...rope import get_na_rope
from ...window import get_window_op
from itertools import chain


class NaMMAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        rope_type: Optional[str],
        rope_dim: int,
        shared_weights: bool,
        attention_mode: str = 'sdpa',
        operations=None,
        **kwargs,
    ):
        super().__init__()
        ops = operations if operations is not None else nn
        dim = MMArg(vid_dim, txt_dim)
        inner_dim = heads * head_dim
        qkv_dim = inner_dim * 3
        self.head_dim = head_dim
        self.proj_qkv = MMModule(
            ops.Linear, dim, qkv_dim, bias=qk_bias, shared_weights=shared_weights
        )
        self.proj_out = MMModule(ops.Linear, inner_dim, dim, shared_weights=shared_weights)
        self.norm_q = MMModule(
            qk_norm,
            dim=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
        )
        self.norm_k = MMModule(
            qk_norm,
            dim=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
        )

        self.rope = get_na_rope(rope_type=rope_type, dim=rope_dim)
        self.attn = FlashAttentionVarlen(attention_mode=attention_mode)

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_qkv = gather_seq_scatter_heads_qkv(
            vid_qkv,
            seq_dim=0,
            qkv_shape=vid_shape,
            cache=cache.namespace("vid"),
        )
        txt_qkv = gather_seq_scatter_heads_qkv(
            txt_qkv,
            seq_dim=0,
            qkv_shape=txt_shape,
            cache=cache.namespace("txt"),
        )
        vid_qkv = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        if self.rope:
            if self.rope.mm:
                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache
                )
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, vid_shape, cache)

        vid_len = cache("vid_len", lambda: vid_shape.prod(-1))
        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))
        all_len = cache("all_len", lambda: vid_len + txt_len)

        concat, unconcat = cache("mm_pnp", lambda: na.concat_idx(vid_len, txt_len))

        # Attention handles dtype conversion internally using pipeline compute_dtype
        attn = self.attn(
            q=concat(vid_q, txt_q),
            k=concat(vid_k, txt_k),
            v=concat(vid_v, txt_v),
            cu_seqlens_q=cache("mm_seqlens", lambda: safe_pad_operation(all_len.cumsum(0), (1, 0)).int()),
            cu_seqlens_k=cache("mm_seqlens", lambda: safe_pad_operation(all_len.cumsum(0), (1, 0)).int()),
            max_seqlen_q=cache("mm_maxlen", lambda: all_len.max()),
            max_seqlen_k=cache("mm_maxlen", lambda: all_len.max()),
        ).type_as(vid_q)

        attn = rearrange(attn, "l h d -> l (h d)")
        vid_out, txt_out = unconcat(attn)
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out


class NaSwinAttention(NaMMAttention):
    def __init__(
        self,
        *args,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        attention_mode: str = 'sdpa',
        **kwargs,
    ):
        super().__init__(*args, attention_mode=attention_mode, **kwargs)
        self.window = _triple(window)
        self.window_method = window_method
        assert all(map(lambda v: isinstance(v, int) and v >= 0, self.window))

        self.window_op = get_window_op(window_method)

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:

        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_qkv = gather_seq_scatter_heads_qkv(
            vid_qkv,
            seq_dim=0,
            qkv_shape=vid_shape,
            cache=cache.namespace("vid"),
        )
        txt_qkv = gather_seq_scatter_heads_qkv(
            txt_qkv,
            seq_dim=0,
            qkv_shape=txt_shape,
            cache=cache.namespace("txt"),
        )

        # re-org the input seq for window attn
        cache_win = cache.namespace(f"{self.window_method}_{self.window}_sd3")

        def make_window(x: torch.Tensor):
            t, h, w, _ = x.shape
            window_slices = self.window_op((t, h, w), self.window)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        window_partition, window_reverse, window_shape, window_count = cache_win(
            "win_transform",
            lambda: na.window_idx(vid_shape, make_window),
        )
        vid_qkv_win = window_partition(vid_qkv)

        vid_qkv_win = rearrange(vid_qkv_win, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv_win.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))

        vid_len_win = cache_win("vid_len", lambda: window_shape.prod(-1))
        txt_len_win = cache_win("txt_len", lambda: txt_len.repeat_interleave(window_count))
        all_len_win = cache_win("all_len", lambda: vid_len_win + txt_len_win)
        concat_win, unconcat_win = cache_win(
            "mm_pnp", lambda: na.repeat_concat_idx(vid_len_win, txt_len, window_count)
        )

        # window rope
        if self.rope:
            if self.rope.mm:
                # repeat text q and k for window mmrope
                _, num_h, _ = txt_q.shape
                txt_q_repeat = rearrange(txt_q, "l h d -> l (h d)")
                txt_q_repeat = na.unflatten(txt_q_repeat, txt_shape)
                txt_q_repeat = [[x] * n for x, n in zip(txt_q_repeat, window_count)]
                txt_q_repeat = list(chain(*txt_q_repeat))
                txt_q_repeat, txt_shape_repeat = na.flatten(txt_q_repeat)
                txt_q_repeat = rearrange(txt_q_repeat, "l (h d) -> l h d", h=num_h)

                txt_k_repeat = rearrange(txt_k, "l h d -> l (h d)")
                txt_k_repeat = na.unflatten(txt_k_repeat, txt_shape)
                txt_k_repeat = [[x] * n for x, n in zip(txt_k_repeat, window_count)]
                txt_k_repeat = list(chain(*txt_k_repeat))
                txt_k_repeat, _ = na.flatten(txt_k_repeat)
                txt_k_repeat = rearrange(txt_k_repeat, "l (h d) -> l h d", h=num_h)

                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q, vid_k, window_shape, txt_q_repeat, txt_k_repeat, txt_shape_repeat, cache_win
                )
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, window_shape, cache_win)
            
        # Attention handles dtype conversion internally using pipeline compute_dtype
        out = self.attn(
            q=concat_win(vid_q, txt_q),
            k=concat_win(vid_k, txt_k),
            v=concat_win(vid_v, txt_v),
            cu_seqlens_q=cache_win(
                "vid_seqlens_q", lambda: safe_pad_operation(all_len_win.cumsum(0), (1, 0)).int()
            ),
            cu_seqlens_k=cache_win(
                "vid_seqlens_k", lambda: safe_pad_operation(all_len_win.cumsum(0), (1, 0)).int()
            ),
            max_seqlen_q=cache_win("vid_max_seqlen_q", lambda: all_len_win.max()),
            max_seqlen_k=cache_win("vid_max_seqlen_k", lambda: all_len_win.max()),
        ).type_as(vid_q)

        # text pooling
        vid_out, txt_out = unconcat_win(out)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")
        vid_out = window_reverse(vid_out)

        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)

        return vid_out, txt_out
```

### `src/models/dit_7b/nadit.py`

```python
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Callable
import torch
from torch import nn

from ...common.cache import Cache
from ...common.distributed.ops import slice_inputs

from . import na
from .embedding import TimeEmbedding
from .modulation import get_ada_layer
from .nablocks import get_nablock
from .normalization import get_norm_layer
from .patch import NaPatchIn, NaPatchOut

# Fake func, no checkpointing is required for inference
def gradient_checkpointing(module: Union[Callable, nn.Module], *args, enabled: bool, **kwargs):
    return module(*args, **kwargs)

@dataclass
class NaDiTOutput:
    vid_sample: torch.Tensor


class NaDiT(nn.Module):
    """
    Native Resolution Diffusion Transformer (NaDiT)
    """

    gradient_checkpointing = False

    def __init__(
        self,
        vid_in_channels: int,
        vid_out_channels: int,
        vid_dim: int,
        txt_in_dim: Optional[int],
        txt_dim: Optional[int],
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: Optional[str],
        norm_eps: float,
        ada: str,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: Optional[str],
        patch_size: Union[int, Tuple[int, int, int]],
        num_layers: int,
        block_type: Union[str, Tuple[str]],
        shared_qkv: bool = False,
        shared_mlp: bool = False,
        mlp_type: str = "normal",
        window: Optional[Tuple] = None,
        window_method: Optional[Tuple[str]] = None,
        temporal_window_size: int = None,
        temporal_shifted: bool = False,
        attention_mode: str = 'sdpa',
        operations=None,
        **kwargs,
    ):
        ada = get_ada_layer(ada)
        norm = get_norm_layer(norm)
        qk_norm = get_norm_layer(qk_norm)
        if isinstance(block_type, str):
            block_type = [block_type] * num_layers
        elif len(block_type) != num_layers:
            raise ValueError("The ``block_type`` list should equal to ``num_layers``.")
        super().__init__()
        ops = operations if operations is not None else nn
        self.vid_in = NaPatchIn(
            in_channels=vid_in_channels,
            patch_size=patch_size,
            dim=vid_dim,
            operations=operations,
        )
        self.txt_in = (
            ops.Linear(txt_in_dim, txt_dim)
            if txt_in_dim and txt_in_dim != txt_dim
            else nn.Identity()
        )
        self.emb_in = TimeEmbedding(
            sinusoidal_dim=256,
            hidden_dim=max(vid_dim, txt_dim),
            output_dim=emb_dim,
            operations=operations,
        )

        if window is None or isinstance(window[0], int):
            window = [window] * num_layers
        if window_method is None or isinstance(window_method, str):
            window_method = [window_method] * num_layers
        if temporal_window_size is None or isinstance(temporal_window_size, int):
            temporal_window_size = [temporal_window_size] * num_layers
        if temporal_shifted is None or isinstance(temporal_shifted, bool):
            temporal_shifted = [temporal_shifted] * num_layers

        self.blocks = nn.ModuleList(
            [
                get_nablock(block_type[i])(
                    vid_dim=vid_dim,
                    txt_dim=txt_dim,
                    emb_dim=emb_dim,
                    heads=heads,
                    head_dim=head_dim,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    norm_eps=norm_eps,
                    ada=ada,
                    qk_bias=qk_bias,
                    qk_rope=qk_rope,
                    qk_norm=qk_norm,
                    shared_qkv=shared_qkv,
                    shared_mlp=shared_mlp,
                    mlp_type=mlp_type,
                    window=window[i],
                    window_method=window_method[i],
                    temporal_window_size=temporal_window_size[i],
                    temporal_shifted=temporal_shifted[i],
                    attention_mode=attention_mode,
                    operations=operations,
                    **kwargs,
                )
                for i in range(num_layers)
            ]
        )
        self.vid_out = NaPatchOut(
            out_channels=vid_out_channels,
            patch_size=patch_size,
            dim=vid_dim,
            operations=operations,
        )

        self.need_txt_repeat = block_type[0] in [
            "mmdit_stwin",
            "mmdit_stwin_spatial",
            "mmdit_stwin_3d_spatial",
        ]

    def set_gradient_checkpointing(self, enable: bool):
        self.gradient_checkpointing = enable

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        timestep: Union[int, float, torch.IntTensor, torch.FloatTensor],  # b
        disable_cache: bool = True,  # for test
    ):
        # Text input.
        if txt_shape.size(-1) == 1 and self.need_txt_repeat:
            txt, txt_shape = na.repeat(txt, txt_shape, "l c -> t l c", t=vid_shape[:, 0])
        # slice vid after patching in when using sequence parallelism
        txt = slice_inputs(txt, dim=0)
        txt = self.txt_in(txt)

        # Video input.
        # Sequence parallel slicing is done inside patching class.
        vid, vid_shape = self.vid_in(vid, vid_shape)

        # Embedding input.
        emb = self.emb_in(timestep, device=vid.device, dtype=vid.dtype)

        # Body
        cache = Cache(disable=disable_cache)
        for i, block in enumerate(self.blocks):
            vid, txt, vid_shape, txt_shape = gradient_checkpointing(
                enabled=(self.gradient_checkpointing and self.training),
                module=block,
                vid=vid,
                txt=txt,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                emb=emb,
                cache=cache,
            )

        vid, vid_shape = self.vid_out(vid, vid_shape, cache)
        return NaDiTOutput(vid_sample=vid)


class NaDiTUpscaler(nn.Module):
    """
    Native Resolution Diffusion Transformer (NaDiT)
    """

    gradient_checkpointing = False

    def __init__(
        self,
        vid_in_channels: int,
        vid_out_channels: int,
        vid_dim: int,
        txt_in_dim: Optional[int],
        txt_dim: Optional[int],
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: Optional[str],
        norm_eps: float,
        ada: str,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: Optional[str],
        patch_size: Union[int, Tuple[int, int, int]],
        num_layers: int,
        block_type: Union[str, Tuple[str]],
        shared_qkv: bool = False,
        shared_mlp: bool = False,
        mlp_type: str = "normal",
        window: Optional[Tuple] = None,
        window_method: Optional[Tuple[str]] = None,
        temporal_window_size: int = None,
        temporal_shifted: bool = False,
        attention_mode: str = 'sdpa',
        operations=None,
        **kwargs,
    ):
        ada = get_ada_layer(ada)
        norm = get_norm_layer(norm)
        qk_norm = get_norm_layer(qk_norm)
        if isinstance(block_type, str):
            block_type = [block_type] * num_layers
        elif len(block_type) != num_layers:
            raise ValueError("The ``block_type`` list should equal to ``num_layers``.")
        super().__init__()
        ops = operations if operations is not None else nn
        self.vid_in = NaPatchIn(
            in_channels=vid_in_channels,
            patch_size=patch_size,
            dim=vid_dim,
            operations=operations,
        )
        self.txt_in = (
            ops.Linear(txt_in_dim, txt_dim)
            if txt_in_dim and txt_in_dim != txt_dim
            else nn.Identity()
        )
        self.emb_in = TimeEmbedding(
            sinusoidal_dim=256,
            hidden_dim=max(vid_dim, txt_dim),
            output_dim=emb_dim,
            operations=operations,
        )

        self.emb_scale = TimeEmbedding(
            sinusoidal_dim=256,
            hidden_dim=max(vid_dim, txt_dim),
            output_dim=emb_dim,
            operations=operations,
        )

        if window is None or isinstance(window[0], int):
            window = [window] * num_layers
        if window_method is None or isinstance(window_method, str):
            window_method = [window_method] * num_layers
        if temporal_window_size is None or isinstance(temporal_window_size, int):
            temporal_window_size = [temporal_window_size] * num_layers
        if temporal_shifted is None or isinstance(temporal_shifted, bool):
            temporal_shifted = [temporal_shifted] * num_layers

        self.blocks = nn.ModuleList(
            [
                get_nablock(block_type[i])(
                    vid_dim=vid_dim,
                    txt_dim=txt_dim,
                    emb_dim=emb_dim,
                    heads=heads,
                    head_dim=head_dim,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    norm_eps=norm_eps,
                    ada=ada,
                    qk_bias=qk_bias,
                    qk_rope=qk_rope,
                    qk_norm=qk_norm,
                    shared_qkv=shared_qkv,
                    shared_mlp=shared_mlp,
                    mlp_type=mlp_type,
                    window=window[i],
                    window_method=window_method[i],
                    temporal_window_size=temporal_window_size[i],
                    temporal_shifted=temporal_shifted[i],
                    attention_mode=attention_mode,
                    operations=operations,
                    **kwargs,
                )
                for i in range(num_layers)
            ]
        )
        self.vid_out = NaPatchOut(
            out_channels=vid_out_channels,
            patch_size=patch_size,
            dim=vid_dim,
            operations=operations,
        )

        self.need_txt_repeat = block_type[0] in [
            "mmdit_stwin",
            "mmdit_stwin_spatial",
            "mmdit_stwin_3d_spatial",
        ]

    def set_gradient_checkpointing(self, enable: bool):
        self.gradient_checkpointing = enable

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        timestep: Union[int, float, torch.IntTensor, torch.FloatTensor],  # b
        downscale: Union[int, float, torch.IntTensor, torch.FloatTensor],  # b
        disable_cache: bool = False,  # for test
    ):

        # Text input.
        if txt_shape.size(-1) == 1 and self.need_txt_repeat:
            txt, txt_shape = na.repeat(txt, txt_shape, "l c -> t l c", t=vid_shape[:, 0])
        # slice vid after patching in when using sequence parallelism
        txt = slice_inputs(txt, dim=0)
        txt = self.txt_in(txt)

        # Video input.
        # Sequence parallel slicing is done inside patching class.
        vid, vid_shape = self.vid_in(vid, vid_shape)

        # Embedding input.
        emb = self.emb_in(timestep, device=vid.device, dtype=vid.dtype)
        emb_scale = self.emb_scale(downscale, device=vid.device, dtype=vid.dtype)
        emb = emb + emb_scale

        # Body
        cache = Cache(disable=disable_cache)
        for i, block in enumerate(self.blocks):
            vid, txt, vid_shape, txt_shape = gradient_checkpointing(
                enabled=(self.gradient_checkpointing and self.training),
                module=block,
                vid=vid,
                txt=txt,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                emb=emb,
                cache=cache,
            )

        vid, vid_shape = self.vid_out(vid, vid_shape, cache)
        return NaDiTOutput(vid_sample=vid)
```

### `src/models/dit_7b/mlp.py`

```python
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn


def get_mlp(mlp_type: Optional[str] = "normal"):
    if mlp_type == "normal":
        return MLP
    elif mlp_type == "swiglu":
        return SwiGLUMLP


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int,
        operations=None,
    ):
        super().__init__()
        ops = operations if operations is not None else nn
        self.proj_in = ops.Linear(dim, dim * expand_ratio)
        self.act = nn.GELU("tanh")
        self.proj_out = ops.Linear(dim * expand_ratio, dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.proj_in(x)
        x = self.act(x)
        x = self.proj_out(x)
        return x


class SwiGLUMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int,
        multiple_of: int = 256,
        operations=None,
    ):
        super().__init__()
        ops = operations if operations is not None else nn
        hidden_dim = int(2 * dim * expand_ratio / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.proj_in_gate = ops.Linear(dim, hidden_dim, bias=False)
        self.proj_out = ops.Linear(hidden_dim, dim, bias=False)
        self.proj_in = ops.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.proj_out(F.silu(self.proj_in_gate(x)) * self.proj_in(x))
        return x
```

### `src/models/dit_7b/embedding.py`

```python
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Optional, Union
import torch
from diffusers.models.embeddings import get_timestep_embedding
from torch import nn


def emb_add(emb1: torch.Tensor, emb2: Optional[torch.Tensor]):
    return emb1 if emb2 is None else emb1 + emb2


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        sinusoidal_dim: int,
        hidden_dim: int,
        output_dim: int,
        operations=None,
    ):
        super().__init__()
        ops = operations if operations is not None else nn
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = ops.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = ops.Linear(hidden_dim, hidden_dim)
        self.proj_out = ops.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()

    def forward(
        self,
        timestep: Union[int, float, torch.IntTensor, torch.FloatTensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.FloatTensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=dtype)
        if timestep.ndim == 0:
            timestep = timestep[None]

        emb = get_timestep_embedding(
            timesteps=timestep,
            embedding_dim=self.sinusoidal_dim,
            flip_sin_to_cos=False,
            downscale_freq_shift=0,
        )
        emb = emb.to(dtype)
        emb = self.proj_in(emb)
        emb = self.act(emb)
        emb = self.proj_hid(emb)
        emb = self.act(emb)
        emb = self.proj_out(emb)
        return emb
```

### `src/models/dit_7b/patch.py`

```python
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Tuple, Union
import torch
from einops import rearrange
from torch import nn
from torch.nn.modules.utils import _triple

from ...common.cache import Cache
from ...common.distributed.ops import gather_outputs, slice_inputs

from . import na


class PatchIn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: Union[int, Tuple[int, int, int]],
        dim: int,
        operations=None,
    ):
        super().__init__()
        ops = operations if operations is not None else nn
        t, h, w = _triple(patch_size)
        self.patch_size = t, h, w
        self.proj = ops.Linear(in_channels * t * h * w, dim)

    def forward(
        self,
        vid: torch.Tensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        vid = rearrange(vid, "b c (T t) (H h) (W w) -> b T H W (t h w c)", t=t, h=h, w=w)
        vid = self.proj(vid)
        return vid


class PatchOut(nn.Module):
    def __init__(
        self,
        out_channels: int,
        patch_size: Union[int, Tuple[int, int, int]],
        dim: int,
        operations=None,
    ):
        super().__init__()
        ops = operations if operations is not None else nn
        t, h, w = _triple(patch_size)
        self.patch_size = t, h, w
        self.proj = ops.Linear(dim, out_channels * t * h * w)

    def forward(
        self,
        vid: torch.Tensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        vid = self.proj(vid)
        vid = rearrange(vid, "b T H W (t h w c) -> b c (T t) (H h) (W w)", t=t, h=h, w=w)
        return vid


class NaPatchIn(PatchIn):
    def forward(
        self,
        vid: torch.Tensor,  # l c
        vid_shape: torch.LongTensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        if not (t == h == w == 1):
            vid, vid_shape = na.rearrange(
                vid, vid_shape, "(T t) (H h) (W w) c -> T H W (t h w c)", t=t, h=h, w=w
            )
        # slice vid after patching in when using sequence parallelism
        vid = slice_inputs(vid, dim=0)
        vid = self.proj(vid)
        return vid, vid_shape


class NaPatchOut(PatchOut):
    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,
        cache: Cache = Cache(disable=True),
    ) -> Tuple[
        torch.FloatTensor,
        torch.LongTensor,
    ]:
        t, h, w = self.patch_size
        vid = self.proj(vid)
        # gather vid before patching out when enabling sequence parallelism
        vid = gather_outputs(
            vid,
            gather_dim=0,
            padding_dim=0,
            unpad_shape=vid_shape,
            cache=cache.namespace("vid"),
        )
        if not (t == h == w == 1):
            vid, vid_shape = na.rearrange(
                vid, vid_shape, "T H W (t h w c) -> (T t) (H h) (W w) c", t=t, h=h, w=w
            )
        return vid, vid_shape
```

### `src/models/dit_7b/nablocks/mmsr_block.py`

```python
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Tuple, Union
import torch
from einops import rearrange
from torch.nn import functional as F

# from ..cache import Cache
from ....common.cache import Cache
from ....common.distributed.ops import gather_heads_scatter_seq, gather_seq_scatter_heads_qkv

from .. import na
from ..attention import FlashAttentionVarlen
from ..blocks.mmdit_window_block import MMWindowAttention, MMWindowTransformerBlock
from ..mm import MMArg
from ..modulation import ada_layer_type
from ..normalization import norm_layer_type
from ..rope import NaRotaryEmbedding3d
from ..window import get_window_op
from ....common.half_precision_fixes import safe_pad_operation

class NaSwinAttention(MMWindowAttention):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        shared_qkv: bool,
        attention_mode: str = 'sdpa',
        operations=None,
        **kwargs,
    ):
        super().__init__(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            window=window,
            window_method=window_method,
            shared_qkv=shared_qkv,
            operations=operations,
        )
        self.rope = NaRotaryEmbedding3d(dim=head_dim // 2) if qk_rope else None
        self.attn = FlashAttentionVarlen(attention_mode=attention_mode)
        self.window_op = get_window_op(window_method)

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:

        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_qkv = gather_seq_scatter_heads_qkv(
            vid_qkv,
            seq_dim=0,
            qkv_shape=vid_shape,
            cache=cache.namespace("vid"),
        )
        txt_qkv = gather_seq_scatter_heads_qkv(
            txt_qkv,
            seq_dim=0,
            qkv_shape=txt_shape,
            cache=cache.namespace("txt"),
        )

        # re-org the input seq for window attn
        cache_win = cache.namespace(f"{self.window_method}_{self.window}_sd3")

        def make_window(x: torch.Tensor):
            t, h, w, _ = x.shape
            window_slices = self.window_op((t, h, w), self.window)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        window_partition, window_reverse, window_shape, window_count = cache_win(
            "win_transform",
            lambda: na.window_idx(vid_shape, make_window),
        )
        vid_qkv_win = window_partition(vid_qkv)

        vid_qkv_win = rearrange(vid_qkv_win, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv_win.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))

        vid_len_win = cache_win("vid_len", lambda: window_shape.prod(-1))
        txt_len_win = cache_win("txt_len", lambda: txt_len.repeat_interleave(window_count))
        all_len_win = cache_win("all_len", lambda: vid_len_win + txt_len_win)
        concat_win, unconcat_win = cache_win(
            "mm_pnp", lambda: na.repeat_concat_idx(vid_len_win, txt_len, window_count)
        )

        # window rope
        if self.rope:
            vid_q, vid_k = self.rope(vid_q, vid_k, window_shape, cache_win)

        # Attention handles dtype conversion internally using pipeline compute_dtype
        out = self.attn(
            q=concat_win(vid_q, txt_q),
            k=concat_win(vid_k, txt_k),
            v=concat_win(vid_v, txt_v),
            cu_seqlens_q=cache_win(
                "vid_seqlens_q", lambda: safe_pad_operation(all_len_win.cumsum(0), (1, 0)).int()
            ),
            cu_seqlens_k=cache_win(
                "vid_seqlens_k", lambda: safe_pad_operation(all_len_win.cumsum(0), (1, 0)).int()
            ),
            max_seqlen_q=cache_win("vid_max_seqlen_q", lambda: all_len_win.max()),
            max_seqlen_k=cache_win("vid_max_seqlen_k", lambda: all_len_win.max()),
        ).type_as(vid_q)

        # text pooling
        vid_out, txt_out = unconcat_win(out)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")
        vid_out = window_reverse(vid_out)

        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)

        return vid_out, txt_out


class NaMMSRTransformerBlock(MMWindowTransformerBlock):
    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: norm_layer_type,
        norm_eps: float,
        ada: ada_layer_type,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: norm_layer_type,
        shared_qkv: bool,
        shared_mlp: bool,
        mlp_type: str,
        **kwargs,
    ):
        super().__init__(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            emb_dim=emb_dim,
            heads=heads,
            head_dim=head_dim,
            expand_ratio=expand_ratio,
            norm=norm,
            norm_eps=norm_eps,
            ada=ada,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            shared_qkv=shared_qkv,
            shared_mlp=shared_mlp,
            mlp_type=mlp_type,
            **kwargs,
        )

        self.attn = NaSwinAttention(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            qk_norm_eps=norm_eps,
            shared_qkv=shared_qkv,
            **kwargs,
        )

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        emb: torch.FloatTensor,
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
    ]:
        hid_len = MMArg(
            cache("vid_len", lambda: vid_shape.prod(-1)),
            cache("txt_len", lambda: txt_shape.prod(-1)),
        )
        ada_kwargs = {
            "emb": emb,
            "hid_len": hid_len,
            "cache": cache,
            "branch_tag": MMArg("vid", "txt"),
        }

        vid_attn, txt_attn = self.attn_norm(vid, txt)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="in", **ada_kwargs)
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, vid_shape, txt_shape, cache)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="out", **ada_kwargs)
        vid_attn, txt_attn = (vid_attn + vid), (txt_attn + txt)

        vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="in", **ada_kwargs)
        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="out", **ada_kwargs)
        vid_mlp, txt_mlp = (vid_mlp + vid_attn), (txt_mlp + txt_attn)

        return vid_mlp, txt_mlp, vid_shape, txt_shape
```

### `src/models/dit_7b/blocks/mmdit_window_block.py`

```python
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Tuple, Union
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _triple
from ....common.half_precision_fixes import safe_pad_operation
from ....common.distributed.ops import (
    gather_heads,
    gather_heads_scatter_seq,
    gather_seq_scatter_heads_qkv,
    scatter_heads,
)

from ..attention import TorchAttention
from ..mlp import get_mlp
from ..mm import MMArg, MMModule
from ..modulation import ada_layer_type
from ..normalization import norm_layer_type
from ..rope import RotaryEmbedding3d


class MMWindowAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        shared_qkv: bool,
        operations=None,
    ):
        super().__init__()
        ops = operations if operations is not None else nn
        dim = MMArg(vid_dim, txt_dim)
        inner_dim = heads * head_dim
        qkv_dim = inner_dim * 3

        self.window = _triple(window)
        self.window_method = window_method
        assert all(map(lambda v: isinstance(v, int) and v >= 0, self.window))

        self.head_dim = head_dim
        self.proj_qkv = MMModule(ops.Linear, dim, qkv_dim, bias=qk_bias, shared_weights=shared_qkv)
        self.proj_out = MMModule(ops.Linear, inner_dim, dim, shared_weights=shared_qkv)
        self.norm_q = MMModule(qk_norm, dim=head_dim, eps=qk_norm_eps, elementwise_affine=True)
        self.norm_k = MMModule(qk_norm, dim=head_dim, eps=qk_norm_eps, elementwise_affine=True)
        self.rope = RotaryEmbedding3d(dim=head_dim // 2) if qk_rope else None
        self.attn = TorchAttention()

    def forward(
        self,
        vid: torch.FloatTensor,  # b T H W c
        txt: torch.FloatTensor,  # b L c
        txt_mask: torch.BoolTensor,  # b L
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        # Project q, k, v.
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_qkv = gather_seq_scatter_heads_qkv(vid_qkv, seq_dim=2)
        _, T, H, W, _ = vid_qkv.shape
        _, L, _ = txt.shape

        if self.window_method == "win":
            nt, nh, nw = self.window
            tt, hh, ww = T // nt, H // nh, W // nw
        elif self.window_method == "win_by_size":
            tt, hh, ww = self.window
            tt, hh, ww = (
                tt if tt > 0 else T,
                hh if hh > 0 else H,
                ww if ww > 0 else W,
            )
            nt, nh, nw = T // tt, H // hh, W // ww
        else:
            raise NotImplementedError

        vid_qkv = rearrange(vid_qkv, "b T H W (o h d) -> o b h (T H W) d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "b L (o h d) -> o b h L d", o=3, d=self.head_dim)
        txt_qkv = scatter_heads(txt_qkv, dim=2)

        vid_q, vid_k, vid_v = vid_qkv.unbind()
        txt_q, txt_k, txt_v = txt_qkv.unbind()

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        if self.rope:
            vid_q, vid_k = self.rope(vid_q, vid_k, (T, H, W))

        def vid_window(v):
            return rearrange(
                v,
                "b h (nt tt nh hh nw ww) d -> b h (nt nh nw) (tt hh ww) d",
                hh=hh,
                ww=ww,
                tt=tt,
                nh=nh,
                nw=nw,
                nt=nt,
            )

        def txt_window(t):
            return rearrange(t, "b h L d -> b h 1 L d").expand(-1, -1, nt * nh * nw, -1, -1)

        # Process video attention.
        vid_msk = safe_pad_operation(txt_mask, (tt * hh * ww, 0), value=True)
        vid_msk = rearrange(vid_msk, "b l -> b 1 1 1 l").expand(-1, 1, 1, tt * hh * ww, -1)
        vid_out = self.attn(
            vid_window(vid_q),
            torch.cat([vid_window(vid_k), txt_window(txt_k)], dim=-2),
            torch.cat([vid_window(vid_v), txt_window(txt_v)], dim=-2),
            vid_msk,
        )
        vid_out = rearrange(
            vid_out,
            "b h (nt nh nw) (tt hh ww) d -> b (nt tt) (nh hh) (nw ww) (h d)",
            hh=hh,
            ww=ww,
            tt=tt,
            nh=nh,
            nw=nw,
        )
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=4, seq_dim=2)

        # Process text attention.
        txt_msk = safe_pad_operation(txt_mask, (T * H * W, 0), value=True)
        txt_msk = rearrange(txt_msk, "b l -> b 1 1 l").expand(-1, 1, L, -1)
        txt_out = self.attn(
            txt_q,
            torch.cat([vid_k, txt_k], dim=-2),
            torch.cat([vid_v, txt_v], dim=-2),
            txt_msk,
        )
        txt_out = rearrange(txt_out, "b h L d -> b L (h d)")
        txt_out = gather_heads(txt_out, dim=2)

        # Project output.
        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out


class MMWindowTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: norm_layer_type,
        norm_eps: float,
        ada: ada_layer_type,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: norm_layer_type,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        shared_qkv: bool,
        shared_mlp: bool,
        mlp_type: str,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        self.attn_norm = MMModule(norm, dim=dim, eps=norm_eps, elementwise_affine=False)
        self.attn = MMWindowAttention(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            qk_norm_eps=norm_eps,
            window=window,
            window_method=window_method,
            shared_qkv=shared_qkv,
            operations=operations,
        )
        self.mlp_norm = MMModule(norm, dim=dim, eps=norm_eps, elementwise_affine=False)
        self.mlp = MMModule(
            get_mlp(mlp_type),
            dim=dim,
            expand_ratio=expand_ratio,
            shared_weights=shared_mlp,
            operations=operations,
        )
        self.ada = MMModule(ada, dim=dim, emb_dim=emb_dim, layers=["attn", "mlp"])

    def forward(
        self,
        vid: torch.FloatTensor,
        txt: torch.FloatTensor,
        txt_mask: torch.BoolTensor,
        emb: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_attn, txt_attn = self.attn_norm(vid, txt)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, emb=emb, layer="attn", mode="in")
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, txt_mask=txt_mask)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, emb=emb, layer="attn", mode="out")
        vid_attn, txt_attn = (vid_attn + vid), (txt_attn + txt)

        vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, emb=emb, layer="mlp", mode="in")
        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, emb=emb, layer="mlp", mode="out")
        vid_mlp, txt_mlp = (vid_mlp + vid_attn), (txt_mlp + txt_attn)

        return vid_mlp, txt_mlp
```

---

## ④ その意味

### `int8_native_ops.py` — HSWQ と comfy.ops の橋渡し

| 関数 | 意味 |
|------|------|
| `checkpoint_is_hswq_int8` | safetensors を開いて `*.comfy_quant` を読み、JSON の `format` が `int8_tensorwise` なら HSWQ INT8 と判定する。拡張子やファイル名推測ではない。 |
| `get_hswq_mixed_precision_ops` | `comfy.ops.mixed_precision_ops(quant_config={}, compute_dtype=fp16)` を返す。空の `quant_config` により、マーカー付き層だけ QuantizedTensor、それ以外は通常 Parameter になる。 |
| `prepare_hswq_state_dict_for_comfy_ops` | `comfy_quant` テンソルを CPU に移す。`layer_conf.numpy()` は CUDA テンソルでは失敗するため必須。 |
| `patch_ops_factory_device` | meta 構築時に `factory_kwargs["device"]` が空/meta のままだと QuantizedTensor が meta に残る。実デバイスを書き込む。 |
| `resolve_linear_ops` | DiT サブモジュール用ヘルパ（`operations` または `torch.nn`）。 |

**なぜ post-load Linear replace では足りないか**  
GGUF 経路は「量子化バッファを独自に持つ Linear」へ後から差し替える。HSWQ / `comfy_quant` は **state_dict ロード時** に `comfy.ops` 側の `_load_from_state_dict` → `_load_quantized_module` が必要。後差し替えではこのフックに乗らない。

### `config.create_object(..., **extra_kwargs)` — YAML を汚さず ops を注入

YAML の `__object__` はそのままに、構築時だけ `operations=` を渡す。FP16 / FP8 / GGUF の既存設定ファイルを変更せず、HSWQ INT8 のときだけ注入できる。

### `model_loader.py` — 検出 → 構築注入 → ロード前 prep

1. `prepare_model_structure`: DiT かつ HSWQ INT8 なら `create_kwargs["operations"] = get_hswq_mixed_precision_ops(fp16)`。`torch.device("meta")` 下で `create_object`。  
2. `_load_and_assign_weights`（相当）: ロードした state に対し `prepare_hswq_state_dict_for_comfy_ops` と `patch_ops_factory_device`。その後通常の `load_state_dict`。  
3. GGUF 分岐とは独立。HSWQ は safetensors + comfy.ops 経路。

### DiT 各モジュールの `operations=None` / `ops.Linear`

`operations` が無いときは従来どおり `torch.nn`（後方互換）。注入時は `comfy.ops` の Linear が建ち、`comfy_quant` / `weight_scale` を正しく解釈する。3B / 7B の Linear を使う経路（MLP、Embedding、Patch、Attention、Window block）へ同じ伝播を入れてある。

### `model_registry.py`

- `seedvr2_7b_int8_convrot.safetensors` / `seedvr2_7b_sharp_int8_convrot.safetensors` を登録。  
- `resolve_dit_config_folder` が registry の `size=7B` を見て `configs_7b` を選ぶ（ファイル名に `7b` が無くてもよい）。

### `seedvr2_int8_bench.py`

カスタムノード配置（Layout A）または隣接 HSWQ ツリー（Layout B）から ComfyUI 本体と SeedVR2 を解決し、FP16 DiT と HSWQ native INT8 DiT を同一条件で比較するベンチ。本番ノードと同じ `model_loader` / `int8_native_ops` 経路を通す。

### 運用上の注意

- ComfyUI 本体に `comfy.ops.mixed_precision_ops` が存在する版が必要。  
- 重みは `models/SEEDVR2/` 等に置き、ノード UI またはベンチ引数でファイル名／パスを指定。  
- INT8 は **DiT のみ**。VAE は従来の FP16（例: `ema_vae_fp16.safetensors`）。  
- 「INT8 ファイルだから」という名前判定ではなく、`comfy_quant` の `int8_tensorwise` が真のゲート。

---

## 参考：ベンチ実行例（custom_nodes 直下）

```powershell
cd D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\seedvr2_videoupscaler
python.exe seedvr2_int8_bench.py --fp16 seedvr2_ema_7b_fp16.safetensors --int8 seedvr2_7b_int8_convrot.safetensors --vae ema_vae_fp16.safetensors
```

（パスは環境に合わせて調整。PowerShell では引数末尾の余分な `"` を付けないこと。）
