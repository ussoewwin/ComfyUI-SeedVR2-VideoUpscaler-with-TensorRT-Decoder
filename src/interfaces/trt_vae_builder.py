"""ComfyUI Node: SeedVR2 Build TensorRT VAE Engines.

Runs the explicit build scripts (tools/cloud_export_gpu.py + tools/cloud_build_engine.py)
as subprocesses so the user chooses the frame count and tile size (256/512) by hand.
There is NO auto-build at inference time: if the engine is missing the pipeline falls
back to the standard PyTorch VAE.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from comfy_api.latest import io

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

from ..utils.model_registry import DEFAULT_VAE, get_available_vae_models
from ..utils.constants import get_base_cache_dir

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "tensorrt_backend" / "artifacts"


class SeedVR2BuildTensorRTVAE(io.ComfyNode):
    """Build dedicated TensorRT VAE engines for a chosen frame count and tile size."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        vae_models = get_available_vae_models()
        return io.Schema(
            node_id="SeedVR2BuildTensorRTVAE",
            display_name="SeedVR2 Build TensorRT VAE Engines",
            category="SEEDVR2",
            description=(
                "Builds dedicated TensorRT VAE engines (encoder + decoder) for a chosen "
                "frame count and tile size by running tools/cloud_export_gpu.py and "
                "tools/cloud_build_engine.py. Frame count is normalized to 4n+1. "
                "Engine files land in tensorrt_backend/artifacts/ and are picked up "
                "automatically by the TensorRT VAE loader after a restart."
            ),
            inputs=[
                io.Combo.Input("model",
                    options=vae_models,
                    default=DEFAULT_VAE,
                    tooltip="VAE model file (fp16 safetensors)."
                ),
                io.Int.Input("frames",
                    default=89,
                    min=5,
                    max=4096,
                    step=4,
                    tooltip="Batch/frame size for the engine. Auto-normalized to 4n+1 "
                            "(e.g. 89, 93, 97, 101, 185, 205)."
                ),
                io.Combo.Input("tile_size",
                    options=["256", "512"],
                    default="256",
                    tooltip="Spatial tile size. 256px fits larger frame counts in 16GB; "
                            "512px is higher quality but limited to ~21-29 frames on 16GB."
                ),
                io.Float.Input("workspace_gb",
                    default=8.0,
                    min=1.0,
                    max=32.0,
                    step=0.5,
                    optional=True,
                    tooltip="TensorRT workspace in GB during engine build."
                ),
                io.Boolean.Input("force_rebuild",
                    default=False,
                    optional=True,
                    tooltip="Rebuild even if the engine already exists."
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip="Build status summary."
                )
            ]
        )

    @classmethod
    def execute(cls, model: str, frames: int = 89, tile_size: str = "256",
                workspace_gb: float = 8.0, force_rebuild: bool = False) -> io.NodeOutput:
        frames = ((frames - 1) // 4) * 4 + 1  # normalize to 4n+1
        tile = int(tile_size)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        python = sys.executable
        worker = ROOT / "tools" / "cloud_export_gpu.py"
        builder = ROOT / "tools" / "cloud_build_engine.py"
        model_dir = Path(get_base_cache_dir())

        enc_stem = f"vae_encoder_{frames}f_tile{tile}"
        dec_stem = f"vae_decoder_tile_{tile}_{frames}f"

        jobs = [
            ("encoder", enc_stem, f"{enc_stem}.rtxplan"),
            ("decoder", dec_stem, f"{dec_stem}.rtxplan"),
        ]

        status_lines = []
        needed = []
        for kind, stem, eng_name in jobs:
            eng_path = ARTIFACTS_DIR / eng_name
            if force_rebuild or not eng_path.exists() or eng_path.stat().st_size < 1_000_000:
                needed.append((kind, stem, eng_name))

        if not needed:
            msg = f"Engines for {frames}f tile{tile} already exist."
            print(f"[SeedVR2] {msg}", flush=True)
            status_lines.append(msg)
            for kind, stem, eng_name in jobs:
                eng_path = ARTIFACTS_DIR / eng_name
                status_lines.append(f" - {eng_name} ({eng_path.stat().st_size / 2**20:.1f} MB)")
        else:
            pbar = ProgressBar(len(needed)) if ProgressBar is not None else None
            for kind, stem, eng_name in needed:
                onnx_path = ARTIFACTS_DIR / f"{stem}.onnx"
                eng_path = ARTIFACTS_DIR / eng_name
                t0 = time.perf_counter()
                print(f"[SeedVR2] Building {frames}f {kind} (tile {tile})...", flush=True)

                # 1) ONNX export (GPU trace)
                r1 = subprocess.run(
                    [python, str(worker), "--repo", str(ROOT), "--kind", kind,
                     "--frames", str(frames), "--tile", str(tile),
                     "--output", str(onnx_path), "--model", model,
                     "--model-dir", str(model_dir)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                )
                tail1 = "\n".join((r1.stdout or "").splitlines()[-3:])
                if r1.returncode != 0:
                    err1 = "\n".join((r1.stderr or "").splitlines()[-10:])
                    raise RuntimeError(f"ONNX export failed for {kind}:\n{tail1}\n{err1}")
                print(f"  [worker] {tail1}", flush=True)

                # 2) TRT build
                r2 = subprocess.run(
                    [python, str(builder), str(onnx_path), "--output", str(eng_path),
                     "--workspace-gb", str(workspace_gb)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                )
                tail2 = "\n".join((r2.stdout or "").splitlines()[-3:])
                if r2.returncode != 0:
                    err2 = "\n".join((r2.stderr or "").splitlines()[-10:])
                    raise RuntimeError(f"Engine build failed for {kind}:\n{tail2}\n{err2}")
                print(f"  [builder] {tail2}", flush=True)

                size_mb = eng_path.stat().st_size / 2**20
                status_lines.append(f" - Built {eng_name} ({size_mb:.1f} MB in {time.perf_counter() - t0:.1f}s)")
                if pbar:
                    pbar.update(1)

        return io.NodeOutput("\n".join(status_lines))
