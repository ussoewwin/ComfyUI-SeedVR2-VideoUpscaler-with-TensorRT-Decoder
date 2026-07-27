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
