#!/usr/bin/env python3
"""
SeedVR2 NVFP4 deterministic trajectory-divergence comparator (FP16 vs ConvRot NVFP4)
=====================================================================================
SeedVR2 版の決定論的・多シード軌跡比較ベンチマーク(HSWQ NVFP4 専用)。

hswq の ``benchmark/zi_convrot_nvfp4_traj_compare.py``(Z Image 向け)を
SeedVR2 videoupscaler 向けに移植したもの。既存の画像比較ベンチ
(``seedvr2_nvfp4_bench_tc.py``)の「画像を入力 → 加工画像(超解像結果)を
MSE / SSIM / diff PNG で比較する」ロジックはそのまま維持し、
それ以外の機能(下記)を取り入れている。

※ ConvRot INT8 と ConvRot NVFP4 は完全に別物のため、スクリプトも分離している。
   本スクリプトは HSWQ NVFP4(comfy_quant nvfp4, optional ConvRot)専用。
   INT8 用は ``seedvr2_int8_traj_compare.py``(混ぜるな)。

取り入れた機能(参照: zi_convrot_nvfp4_traj_compare.py)
--------------------------------------------------------
1. per-step latent 軌跡比較(EulerSampler にフックし、毎ステップの
   noisy latent x とモデル予測 x0 の cosine / MSE を FP16 vs NVFP4 で比較)
2. 多シード対応(--seeds "42,1337,7")。同一シード = 同一ノイズ。
3. 分岐(bifurcation)検出:単一ステップでの cosine 急落 = 別画像への
   軌道ジャンプ(単なる劣化ではない)を判定。
4. 最終判定 same-image / drifted / bifurcated。
5. GEMM モード要約(HSWQ TC W4A4 scaled_mm が実際に走ったか、
   dequant fallback に落ちたか)。
6. --tc / --parity による NVFP4 実行パスの明示的な強制。
7. 決定論的比較のための cuDNN deterministic 固定。

量化チェックポイントは HSWQ NVFP4(comfy_quant nvfp4, optional ConvRot)専用。
comfy_quant マーカーをスキャンして NVFP4 レイヤー数と convrot レイヤー数を
報告する。

--tc を使う場合のみ ``nvfp4_tc_patch.py``(HSWQ TC forward ブリッジ)が
必要。探索順: --hswq_path → ComfyUI-HSWQ-Loader-and-Tools/nodes →
本スクリプトと同じディレクトリ → $env:HSWQ_TC_PATCH_DIR。

Usage:
    python benchmark/seedvr2_nvfp4_traj_compare.py ^
        --fp16  seedvr2_ema_7b_fp16.safetensors ^
        --nvfp4 seedvr2_7b_hswq_nvfp4_convrot.safetensors ^
        --vae   ema_vae_fp16.safetensors ^
        [--image input.png] [--seeds "42,1337,7"] [--steps 50] [--cfg 7.5] [--tc]
"""
from __future__ import annotations

import argparse
import gc
import json
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
    """PowerShell trailing \\ leaves a final backslash; strip it."""
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
        comfy = _find_comfy_root(SCRIPT_DIR)
        layout = "comfyui_custom_node"
        if comfy is None:
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
            host = _find_comfy_root(Path.cwd())
            if host is not None:
                host_models = host / "models" / "SEEDVR2"
                if host_models.is_dir():
                    model_dir = host_models
        return seed, comfy, (model_dir if model_dir.is_dir() else None), "hswq_repo"

    # Layout C: script lives inside a seedvr2_videoupscaler/benchmark/ subdirectory.
    seed_c = SCRIPT_DIR.parent
    if (seed_c / "inference_cli.py").is_file() and (seed_c / "src").is_dir():
        comfy_c = _find_comfy_root(SCRIPT_DIR)
        layout_c = "comfyui_custom_node_benchmark_subdir"
        if comfy_c is None:
            sibling = seed_c.parent / "ComfyUI-master"
            if (sibling / "comfy" / "ops.py").is_file():
                comfy_c = sibling
                layout_c = "hswq_seedvr2_benchmark_subdir"
        if comfy_c is None:
            raise RuntimeError(
                "Could not find ComfyUI root (comfy/ops.py) above "
                f"{SCRIPT_DIR}, and sibling ComfyUI-master is missing. "
                "Pass --comfy_path."
            )
        model_dir_c = comfy_c / "models" / "SEEDVR2"
        return seed_c, comfy_c, (model_dir_c if model_dir_c.is_dir() else None), layout_c

    raise RuntimeError(
        "Cannot discover SeedVR2 / ComfyUI layout from "
        f"{SCRIPT_DIR}. Place this script in "
        "custom_nodes/seedvr2_videoupscaler/ or pass --seedvr2_path / --comfy_path."
    )


_LAYOUT = "auto"
DEFAULT_SEEDVR2_PATH = None
DEFAULT_COMFY_PATH = None
DEFAULT_MODEL_DIR = None
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "seedvr2_out"


def _resolve_layout(seedvr2_path: str | None, comfy_path: str | None) -> tuple[str, str, str | None, str]:
    """
    Resolve (seedvr2_root, comfy_root, model_dir, layout) at runtime.

    - seedvr2 root: explicit ``--seedvr2_path`` wins; otherwise inferred from
      the script location (script dir or its parent must contain
      ``inference_cli.py`` + ``src/``).
    - comfy root: explicit ``--comfy_path`` wins; otherwise the original
      auto-discovery (ancestor with comfy/ops.py, then sibling ComfyUI-master)
      is tried as a fallback.

    This is more forgiving than the original seedvr2 benches, which resolved
    at import time and could not start outside a ComfyUI tree.
    """
    layout = "explicit"

    if seedvr2_path is None:
        for cand in (SCRIPT_DIR, SCRIPT_DIR.parent):
            if (cand / "inference_cli.py").is_file() and (cand / "src").is_dir():
                seedvr2_path = str(cand)
                layout = "script-relative"
                break

    if comfy_path is None:
        try:
            _d_seed, d_comfy, _d_model, d_layout = _discover_defaults()
            comfy_path = str(d_comfy)
            layout = d_layout
        except RuntimeError:
            comfy_path = None

    if not seedvr2_path or not os.path.isdir(seedvr2_path):
        raise RuntimeError(
            "Could not resolve seedvr2_videoupscaler root. "
            "Pass --seedvr2_path explicitly."
        )
    if not comfy_path or not os.path.isdir(comfy_path):
        raise RuntimeError(
            "Could not resolve ComfyUI root. Pass --comfy_path explicitly "
            "(directory containing comfy/ops.py)."
        )
    return seedvr2_path, comfy_path, None, layout


def _dit_size_tag(*names: str) -> str:
    """SeedVR2 configure_runner selects configs_7b iff '7b' in dit_model filename."""
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
    """Accept either a plain filename (resolved under model_dir) or any path."""
    raw = Path(_clean_path(path_or_name))
    if raw.is_file():
        return raw.resolve()
    if model_dir is not None:
        candidate = (model_dir / raw.name).resolve()
        if candidate.is_file():
            return candidate
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
    """Put package roots on sys.path: seedvr2_videoupscaler + ComfyUI root."""
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


# =============================================================================
# Trajectory hook — EulerSampler.sample monkey-patch
# =============================================================================
# SeedVR2 の DiT は src/common/diffusion/samplers/euler.py の EulerSampler で
# サンプリングされる。ここにフックを入れ、毎ステップの
#   x  : 現在の noisy latent
#   x0 : モデル予測から schedule.convert_from_pred() で復元した clean 推定
# を記録する。参照(zi_convrot_nvfp4_traj_compare.py)の callback
# (step, x0, x, total_steps) と同じ規約にする。

_TRAJ_CB = None
_ORIG_EULER_SAMPLE = None
_TRAJ_PATCHED = False


def _patch_euler_sampler() -> None:
    """Wrap EulerSampler.sample to emit per-step (x0, x) via the module-level hook."""
    global _ORIG_EULER_SAMPLE, _TRAJ_PATCHED
    if _TRAJ_PATCHED:
        return
    from src.common.diffusion.samplers.base import SamplerModelArgs
    from src.common.diffusion.samplers.euler import EulerSampler

    _ORIG_EULER_SAMPLE = EulerSampler.sample

    def sample_with_traj(self, x, f):
        cb = _TRAJ_CB
        timesteps = self.timesteps.timesteps
        progress = self.get_progress_bar()
        i = 0
        for t, s in zip(timesteps[:-1], timesteps[1:]):
            pred = f(SamplerModelArgs(x, t, i))
            if cb is not None:
                pred_x_0, _ = self.schedule.convert_from_pred(
                    pred, self.prediction_type, x, t
                )
                cb(i, pred_x_0, x, len(timesteps))
            x = self.step_to(pred, x, t, s)
            del pred
            i += 1
            progress.update()

        if self.return_endpoint:
            t = timesteps[-1]
            pred = f(SamplerModelArgs(x, t, i))
            if cb is not None:
                pred_x_0, _ = self.schedule.convert_from_pred(
                    pred, self.prediction_type, x, t
                )
                cb(i, pred_x_0, x, len(timesteps))
            x = self.get_endpoint(pred, x, t)
            del pred
            progress.update()

        return x

    EulerSampler.sample = sample_with_traj
    _TRAJ_PATCHED = True


def _unpatch_euler_sampler() -> None:
    global _ORIG_EULER_SAMPLE, _TRAJ_PATCHED
    if not _TRAJ_PATCHED:
        return
    from src.common.diffusion.samplers.euler import EulerSampler

    EulerSampler.sample = _ORIG_EULER_SAMPLE
    _ORIG_EULER_SAMPLE = None
    _TRAJ_PATCHED = False


# =============================================================================
# Config overrides (steps / cfg) — patch the factories used by infer.py
# =============================================================================
# generation_phases.py が runner.config.diffusion.timesteps.sampling.steps = 1
# と直接書き換えるのと同じ流儀で、config オブジェクトの属性を上書きしてから
# ファクトリを呼ぶ。cfg は VideoDiffusionInfer.inference の cfg_scale 引数
# (None なら config 値)をパッチで差し替える。

_ORIG_TIMESTEPS_FACTORY = None
_ORIG_INFERENCE = None
_CFG_OVERRIDE = None
_STEPS_OVERRIDE = None


def _patch_config_overrides(steps: int | None, cfg: float | None) -> None:
    global _ORIG_TIMESTEPS_FACTORY, _ORIG_INFERENCE, _CFG_OVERRIDE, _STEPS_OVERRIDE
    _STEPS_OVERRIDE = steps
    _CFG_OVERRIDE = cfg

    import src.core.infer as infer_mod

    if _ORIG_TIMESTEPS_FACTORY is None:
        _ORIG_TIMESTEPS_FACTORY = infer_mod.create_sampling_timesteps_from_config

        def _timesteps_with_override(config, *a, **k):
            if _STEPS_OVERRIDE is not None:
                config.steps = _STEPS_OVERRIDE
            return _ORIG_TIMESTEPS_FACTORY(config, *a, **k)

        infer_mod.create_sampling_timesteps_from_config = _timesteps_with_override

    if _ORIG_INFERENCE is None:
        _ORIG_INFERENCE = infer_mod.VideoDiffusionInfer.inference

        def _inference_with_cfg(self, noises, conditions, texts_pos, texts_neg,
                                cfg_scale=None, **kw):
            if _CFG_OVERRIDE is not None and cfg_scale is None:
                cfg_scale = _CFG_OVERRIDE
            return _ORIG_INFERENCE(
                self, noises, conditions, texts_pos, texts_neg,
                cfg_scale=cfg_scale, **kw,
            )

        infer_mod.VideoDiffusionInfer.inference = _inference_with_cfg


def _unpatch_config_overrides() -> None:
    global _ORIG_TIMESTEPS_FACTORY, _ORIG_INFERENCE, _CFG_OVERRIDE, _STEPS_OVERRIDE
    if _ORIG_TIMESTEPS_FACTORY is not None:
        import src.core.infer as infer_mod

        infer_mod.create_sampling_timesteps_from_config = _ORIG_TIMESTEPS_FACTORY
        _ORIG_TIMESTEPS_FACTORY = None
    if _ORIG_INFERENCE is not None:
        import src.core.infer as infer_mod

        infer_mod.VideoDiffusionInfer.inference = _ORIG_INFERENCE
        _ORIG_INFERENCE = None
    _CFG_OVERRIDE = None
    _STEPS_OVERRIDE = None


# =============================================================================
# Quant checkpoint inspection (convrot markers)
# =============================================================================

def _resolve_tc_patch_dir(hswq_path: str | None, comfy_root: str) -> str:
    """
    Locate the directory that contains ``nvfp4_tc_patch.py``.

    The TC bridge module is not shipped inside ComfyUI-HSWQ-Loader-and-Tools;
    it lives next to the bench (``benchmark/nvfp4_tc_patch.py``) or wherever
    ``--hswq_path`` / ``HSWQ_TC_PATCH_DIR`` points. Search order:
      1. --hswq_path
      2. <comfy_root>/custom_nodes/ComfyUI-HSWQ-Loader-and-Tools/nodes
      3. this script's directory (benchmark/)
      4. $env:HSWQ_TC_PATCH_DIR
    """
    candidates = []
    if hswq_path:
        candidates.append(hswq_path)
    candidates.append(
        os.path.join(comfy_root, "custom_nodes",
                     "ComfyUI-HSWQ-Loader-and-Tools", "nodes")
    )
    candidates.append(str(SCRIPT_DIR))
    env_dir = os.environ.get("HSWQ_TC_PATCH_DIR", "")
    if env_dir:
        candidates.append(env_dir)
    for c in candidates:
        if os.path.isfile(os.path.join(c, "nvfp4_tc_patch.py")):
            return c
    raise RuntimeError(
        "--tc requires nvfp4_tc_patch.py. Place it next to this script "
        f"({SCRIPT_DIR}) or pass --hswq_path / set HSWQ_TC_PATCH_DIR. "
        "Checked: " + ", ".join(repr(c) for c in candidates if c)
    )


def count_nvfp4_convrot_markers(path: str) -> tuple[int, int, str]:
    """
    Scan safetensors .comfy_quant markers for NVFP4 (comfy_quant nvfp4) and
    convrot flags. Returns (n_convrot, n_nvfp4_layers, hswq_nvfp4_convrot header flag).
    """
    from safetensors import safe_open

    n_quant = 0
    n_cr = 0
    flag = ""
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            flag = str(meta.get("hswq_nvfp4_convrot", "") or "")
            for key in f.keys():
                if not key.endswith(".comfy_quant"):
                    continue
                raw = f.get_tensor(key)
                if hasattr(raw, "detach"):
                    blob = bytes(raw.detach().cpu().tolist())
                else:
                    blob = bytes(raw)
                try:
                    conf = json.loads(blob.decode("utf-8"))
                except Exception:
                    continue
                fmt = str(conf.get("format", "")).lower()
                if fmt != "nvfp4":
                    continue
                n_quant += 1
                if conf.get("convrot") is True or str(
                    conf.get("convrot", "")
                ).lower() in ("1", "true"):
                    n_cr += 1
    except Exception as ex:
        print(f"  [Note] nvfp4 marker scan failed for {path}: {ex}")
        return 0, 0, ""
    return n_cr, n_quant, flag


# =============================================================================
# Metrics
# =============================================================================

def _cos(a, b):
    a = a.reshape(1, -1).float()
    b = b.reshape(1, -1).float()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).item())


def _mse(a, b):
    return float((a.float() - b.float()).pow(2).mean().item())


def _make_traj_cb(store: list):
    """Callback receiving (step, x0, x, total_steps); detaches to float32 CPU."""
    def cb(step, x0, x, total_steps):
        store[0].append(x.detach().float().cpu())
        store[1].append(x0.detach().float().cpu())
    return cb


def _hard_free() -> None:
    """Free VRAM between branches / seeds."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Branch runner (videoupscaler pipeline) with optional trajectory capture
# =============================================================================

def run_branch(
    *,
    label: str,
    dit_model: str,
    model_dir: str,
    frames: torch.Tensor,
    args_ns: argparse.Namespace,
    traj_store=None,
) -> tuple[Image.Image, float, float, list, list]:
    """
    Run one videoupscaler branch (FP16 or quant).

    When ``traj_store`` (a ``[[], []]`` pair) is given, the EulerSampler hook
    appends per-step (x, x0) into it and the same lists are returned as
    (xs, x0s) so the caller can compare trajectories between branches.
    """
    from src.utils.debug import Debug
    from inference_cli import _process_frames_core

    global _TRAJ_CB
    if traj_store is None:
        traj_store = [[], []]
    _TRAJ_CB = _make_traj_cb(traj_store)

    try:
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
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        print(f"  wall: {elapsed:.2f}s  peak_vram={peak_gb:.2f} GiB  out={tuple(result.shape)}")

        img = thwc_to_pil(result)
        del result
        _hard_free()
        return img, elapsed, peak_gb, traj_store[0], traj_store[1]
    finally:
        _TRAJ_CB = None


# =============================================================================
# GEMM mode summary (SeedVR2 / HSWQ nodes flavour)
# =============================================================================

def print_nvfp4_gemm_summary(tc_applied: bool) -> None:
    """Print which NVFP4 GEMM path actually ran (HSWQ TC W4A4 vs stock dequant)."""
    tc_hits = deq_fb = 0
    if tc_applied:
        try:
            from nvfp4.nvfp4_forward import nvfp4_forward_stats

            s = nvfp4_forward_stats()
            tc_hits = int(s.get("scaled_mm_hits", 0))
            deq_fb = int(s.get("dequant_fallbacks", 0))
        except Exception as e:
            print(f"  [Note] HSWQ nvfp4 stats unavailable: {e}")

    print("\n" + "-" * 72)
    if tc_applied:
        mode = "TC (W4A4 scaled_mm)" if tc_hits > 0 else (
            "TC patch active but 0 scaled_mm hits -> dequant fallback"
        )
        print(f"[HSWQ QUANT] GEMM MODE: {mode}")
        print(f"  TC forward: scaled_mm hits={tc_hits}  dequant_fallbacks={deq_fb}")
    else:
        print("[HSWQ QUANT] GEMM MODE: stock ComfyUI mixed_precision (dequant GEMM)")
        print("  (no HSWQ TC patch applied; QuantizedTensor stays packed, matmul dequantizes)")
    print("-" * 72)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "SeedVR2 NVFP4 deterministic per-step trajectory divergence "
            "(FP16 vs ConvRot NVFP4), image-in / processed-image-out"
        )
    )
    ap.add_argument("--fp16", required=True, help="BF16/FP16 baseline model path")
    ap.add_argument(
        "--nvfp4", dest="nvfp4_path", required=True,
        help="HSWQ NVFP4 (comfy_quant nvfp4) quantized model path (convrot optional)",
    )
    ap.add_argument("--vae", required=True, help="SeedVR2 VAE safetensors")
    ap.add_argument(
        "--seedvr2_path",
        default=DEFAULT_SEEDVR2_PATH,
        help="seedvr2_videoupscaler root (auto-discovered from script location if omitted)",
    )
    ap.add_argument(
        "--comfy_path",
        default=DEFAULT_COMFY_PATH,
        help="ComfyUI root for comfy.ops (auto-discovered from script location if omitted)",
    )
    ap.add_argument(
        "--model_dir",
        default=str(DEFAULT_MODEL_DIR) if DEFAULT_MODEL_DIR is not None else None,
        help=(
            "Directory containing DiT/VAE filenames "
            f"(default: {DEFAULT_MODEL_DIR or 'directory of --fp16'})"
        ),
    )
    ap.add_argument(
        "--image",
        default=None,
        help="Optional input image. When omitted, a synthetic RGB pattern is used.",
    )
    ap.add_argument(
        "--resolution",
        type=int,
        default=1080,
        help="Target short-side resolution (videoupscaler default: 1080)",
    )
    ap.add_argument(
        "--seeds",
        default="42",
        help=(
            "comma-separated seeds; same seed = identical noise for both models. "
            "Note: each seed runs the FULL encode->DiT->decode pipeline twice, "
            "so prefer fewer seeds (default: 42; hswq reference defaults to 5)."
        ),
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override sampling steps (default: config, 50 for 3b/7b)",
    )
    ap.add_argument(
        "--cfg",
        type=float,
        default=None,
        help="Override CFG scale (default: config, 7.5)",
    )
    ap.add_argument("--batch_size", type=int, default=1, help="Frames per batch (image=1)")
    ap.add_argument(
        "--color",
        default="lab",
        choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
        help="color_correction (default: lab)",
    )
    ap.add_argument("--attention_mode", default="sdpa")
    ap.add_argument(
        "--tc", "--tc_forward", dest="tc", action="store_true",
        help=(
            "Force HSWQ Tensor Core forward path "
            "(pooled CUDA quantize + cuBLAS scaled_mm_nvfp4). "
            "Requires ComfyUI-HSWQ-Loader-and-Tools."
        ),
    )
    ap.add_argument(
        "--parity",
        action="store_true",
        help=(
            "Force stock ComfyUI mixed_precision path (dequant GEMM, no HSWQ TC patch). "
            "This is the default; provided for interface parity with the hswq reference."
        ),
    )
    ap.add_argument(
        "--hswq_path",
        default=None,
        help="Path to ComfyUI-HSWQ-Loader-and-Tools/nodes directory (auto-detected if omitted)",
    )
    ap.add_argument("--blocks_to_swap", type=int, default=0)
    ap.add_argument("--dit_offload_device", default="none")
    ap.add_argument("--vae_offload_device", default="none")
    ap.add_argument("--tensor_offload_device", default="cpu")
    ap.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument(
        "--show-steps",
        action="store_true",
        help="print the per-step divergence curve (default: only final per seed)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # Deterministic comparison: same seed = same noise. Pin cuDNN to avoid
    # autotuning / algorithm-selection noise between the FP16 and quant runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.tc and args.parity:
        raise SystemExit("--tc and --parity are mutually exclusive")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")

    # Resolve package roots after parsing so explicit --comfy_path works even
    # when the script is not inside a ComfyUI tree (auto-discovery fallback).
    args.seedvr2_path, args.comfy_path, resolved_model_dir, layout = _resolve_layout(
        args.seedvr2_path, args.comfy_path
    )
    if args.model_dir is None and resolved_model_dir is not None:
        args.model_dir = resolved_model_dir

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
    nvfp4_path = _resolve_weight(args.nvfp4_path, model_dir_path, "--nvfp4")
    vae_path = _resolve_weight(args.vae, model_dir_path, "--vae")
    if args.image is not None and not Path(args.image).is_file():
        raise FileNotFoundError(f"--image not found: {args.image}")

    tag = _dit_size_tag(str(fp16_path), str(nvfp4_path))
    print(f"[BENCH] DiT size tag: {tag}")

    model_dir = str(model_dir_path) if model_dir_path is not None else str(fp16_path.parent)
    model_dir_p = Path(model_dir)
    vae_name = vae_path.name
    nvfp4_name = nvfp4_path.name
    fp16_name = fp16_path.name

    for src, name in ((vae_path, vae_name), (nvfp4_path, nvfp4_name), (fp16_path, fp16_name)):
        target = model_dir_p / name
        if src.resolve() != target.resolve():
            if not target.is_file():
                raise FileNotFoundError(
                    f"{name} must live under --model_dir: expected {target}"
                )

    os.makedirs(args.output_dir, exist_ok=True)

    # --- quant checkpoint validation (NVFP4) ---
    seed_root, comfy_root = _install_package_paths(
        seedvr2_path=args.seedvr2_path,
        comfy_path=args.comfy_path,
    )

    from src.optimization.nvfp4_native_ops import checkpoint_is_nvfp4
    from src.utils.model_registry import DEFAULT_VAE as _DEFAULT_VAE

    if not checkpoint_is_nvfp4(str(nvfp4_path)):
        raise RuntimeError(
            f"--nvfp4 does not look like HSWQ NVFP4 comfy_quant: {nvfp4_path}"
        )
    n_cr, n_quant, hflag = count_nvfp4_convrot_markers(str(nvfp4_path))
    print("[BENCH] quant kind: nvfp4 (dedicated NVFP4 trajectory bench)")
    print(
        f"[CONVROT markers] nvfp4_layers={n_quant} convrot_stamps={n_cr} "
        f"hswq_nvfp4_convrot={hflag!r}"
    )
    print(f"[BENCH] python: {sys.executable}")
    print(f"[BENCH] layout: {layout}")
    print(f"[BENCH] script_dir: {SCRIPT_DIR}")
    print(f"[BENCH] seedvr2_path: {args.seedvr2_path}")
    print(f"[BENCH] comfy_path: {args.comfy_path}")
    print(f"[BENCH] model_dir: {model_dir}")
    print("[BENCH] mode: NVFP4 (construction-time mixed_precision_ops)")

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

    # --- HSWQ TC forward patch ---
    tc_applied = False
    if args.tc:
        tc_patch_dir = _resolve_tc_patch_dir(
            hswq_path=args.hswq_path, comfy_root=comfy_root
        )
        print(f"[BENCH] Applying HSWQ TC forward patch...")
        print(f"[BENCH] nvfp4_tc_patch.py dir: {tc_patch_dir}")
        if tc_patch_dir not in sys.path:
            sys.path.insert(0, tc_patch_dir)
        from nvfp4_tc_patch import apply_tc_forward_patch

        hswq_nodes = args.hswq_path
        if hswq_nodes is None:
            cand = os.path.join(
                comfy_root, "custom_nodes",
                "ComfyUI-HSWQ-Loader-and-Tools", "nodes",
            )
            if os.path.isdir(cand):
                hswq_nodes = cand
        if hswq_nodes is None:
            # Fallback: nvfp4/ package co-located with the tc patch bridge.
            for d in (tc_patch_dir, str(SCRIPT_DIR)):
                if os.path.isdir(os.path.join(d, "nvfp4")):
                    hswq_nodes = d
                    break
        if hswq_nodes is None:
            raise RuntimeError(
                "--tc requires the HSWQ nvfp4 package. Pass "
                "--hswq_path /path/to/ComfyUI-HSWQ-Loader-and-Tools/nodes"
            )
        print(f"[BENCH] HSWQ nvfp4 package dir: {hswq_nodes}")
        apply_tc_forward_patch(hswq_path=hswq_nodes, comfy_root=comfy_root)
        tc_applied = True
        print(
            "[BENCH] TC forward patch applied — NVFP4 branch uses "
            "pooled CUDA quantize + cuBLAS scaled_mm_nvfp4"
        )

    # --- trajectory + config hooks (before any model load) ---
    _patch_euler_sampler()
    _patch_config_overrides(steps=args.steps, cfg=args.cfg)

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
        seed=seeds[0],
        color_correction=args.color,
        batch_size=args.batch_size,
        attention_mode=args.attention_mode,
        blocks_to_swap=args.blocks_to_swap,
        dit_offload_device=args.dit_offload_device,
        vae_offload_device=args.vae_offload_device,
        tensor_offload_device=args.tensor_offload_device,
    )

    quant_tag = "nvfp4_tc" if tc_applied else "nvfp4"
    quant_label = (
        "NVFP4 + TC forward" if tc_applied else "NVFP4 (native QuantizedTensor)"
    )

    # --- per-seed runs ---
    print("\n" + "=" * 72)
    print("SeedVR2 deterministic trajectory + image comparison")
    print("=" * 72)
    BIFURC_DROP = 0.05   # single-step cosine drop threshold = sudden jump (different image)
    SAME_IMG_COS = 0.98  # final cosine above this = same picture (not merely different)

    final_rows = []
    multi = len(seeds) > 1
    for s in seeds:
        ns.seed = s
        print(f"\n########## SEED {s} ##########")

        # --- FP16 branch ---
        img_fp16, t_fp16, v_fp16, fxs, fx0s = run_branch(
            label="FP16",
            dit_model=fp16_name,
            model_dir=model_dir,
            frames=frames,
            args_ns=ns,
        )
        fp16_name_out = f"seedvr2_fp16{'_{}'.format('s' + str(s)) if multi else ''}.png"
        out_fp16 = Path(args.output_dir) / fp16_name_out
        img_fp16.save(out_fp16)
        print(f"  saved: {out_fp16}")

        # --- quant branch ---
        img_quant, t_quant, v_quant, nxs, nx0s = run_branch(
            label=quant_label,
            dit_model=nvfp4_name,
            model_dir=model_dir,
            frames=frames,
            args_ns=ns,
        )
        quant_name_out = f"seedvr2_{quant_tag}{'_{}'.format('s' + str(s)) if multi else ''}.png"
        out_quant = Path(args.output_dir) / quant_name_out
        img_quant.save(out_quant)
        print(f"  saved: {out_quant}")

        # --- image-level comparison (kept from the original seedvr2 benches) ---
        if img_fp16.size != img_quant.size:
            print(
                f"  [BENCH] size mismatch FP16={img_fp16.size} QUANT={img_quant.size}; "
                "resizing quant to FP16 for metrics"
            )
            img_quant = img_quant.resize(img_fp16.size, Image.Resampling.LANCZOS)
        mse_img, ssim_img = calculate_metrics(img_fp16, img_quant)
        diff = Image.fromarray(
            np.abs(np.asarray(img_fp16).astype(np.int16) - np.asarray(img_quant).astype(np.int16))
            .clip(0, 255)
            .astype(np.uint8)
        )
        diff_name = f"seedvr2_diff_{quant_tag}{'_{}'.format('s' + str(s)) if multi else ''}.png"
        out_diff = Path(args.output_dir) / diff_name
        diff.save(out_diff)
        print(f"  saved: {out_diff}")
        print(f"  [image] MSE: {mse_img:.6f}  SSIM: {ssim_img:.6f}")

        # --- trajectory comparison ---
        n_steps = min(len(fxs), len(nxs))
        step_cos = [_cos(fxs[i], nxs[i]) for i in range(n_steps)]
        max_drop = 0.0
        drop_at = 0
        for i in range(1, n_steps):
            d = step_cos[i - 1] - step_cos[i]
            if d > max_drop:
                max_drop, drop_at = d, i

        if args.show_steps:
            print(f"\n--- Seed {s}: per-step (x = noisy latent, x0 = model prediction) ---")
            print(f"{'step':>4} {'x-cos':>8} {'x-MSE':>10} {'x0-cos':>8} {'x0-MSE':>10}")
            for i in range(n_steps):
                print(
                    f"{i+1:>4} {step_cos[i]:>8.5f} {_mse(fxs[i], nxs[i]):>10.3e} "
                    f"{_cos(fx0s[i], nx0s[i]):>8.5f} {_mse(fx0s[i], nx0s[i]):>10.3e}"
                )

        # Final sample = last-step clean estimate (x0). EulerSampler's
        # return_endpoint path emits it as the final trajectory x0 entry,
        # matching the reference script's ffinal/nfinal comparison.
        fin_cos = _cos(fx0s[-1], nx0s[-1]) if fx0s and nx0s else float("nan")
        fin_mse = _mse(fx0s[-1], nx0s[-1]) if fx0s and nx0s else float("nan")
        x0_cos = fin_cos
        if max_drop > BIFURC_DROP:
            verdict = f"bifurcated @step {drop_at}"
        elif fin_cos >= SAME_IMG_COS:
            verdict = "same-image"
        else:
            verdict = "drifted (different image)"
        final_rows.append(
            (s, fin_cos, fin_mse, x0_cos, verdict, max_drop, drop_at, mse_img, ssim_img)
        )
        print(
            f"[seed {s}] final-latent-cos={fin_cos:.5f}  max_step_drop={max_drop:.4f}"
            f"{' @step ' + str(drop_at) if max_drop > BIFURC_DROP else ''}  -> {verdict}"
        )
        print(
            f"[seed {s}] FP16 wall: {t_fp16:.2f}s / {v_fp16:.2f} GiB | "
            f"{quant_tag} wall: {t_quant:.2f}s / {v_quant:.2f} GiB"
        )
        _hard_free()

    # --- multi-seed summary ---
    print("\n--- Multi-seed summary ---")
    print(
        f"{'seed':>8} {'lat-cos':>9} {'lat-mse':>11} {'max-drop':>9} {'img-MSE':>10} "
        f"{'img-SSIM':>9} {'verdict':>22}"
    )
    for s, fc, fm, xc, v, md, da, mi, si in final_rows:
        print(
            f"{s:>8} {fc:>9.5f} {fm:>11.3e} {md:>9.4f} {mi:>10.6f} {si:>9.6f} {v:>22}"
        )
    cos_vals = [r[1] for r in final_rows]
    n_bif = sum(1 for r in final_rows if "bifurcated" in r[4])
    n_diff = sum(1 for r in final_rows if r[4] != "same-image")
    print(
        f"\nfinal-latent-cosine: min={min(cos_vals):.5f}  "
        f"mean={sum(cos_vals) / len(cos_vals):.5f}  max={max(cos_vals):.5f}"
    )
    print(f"same-image seeds : {len(seeds) - n_diff}/{len(seeds)}")
    print(
        f"bifurcated seeds : {n_bif}/{len(seeds)}   "
        "(sudden trajectory jump = different picture, not degradation)"
    )

    print_nvfp4_gemm_summary(tc_applied)

    _unpatch_config_overrides()
    _unpatch_euler_sampler()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        # Restore hooks on abnormal exit so subsequent runs in the same
        # interpreter are not affected.
        try:
            _unpatch_config_overrides()
            _unpatch_euler_sampler()
        except Exception:
            pass
        raise
