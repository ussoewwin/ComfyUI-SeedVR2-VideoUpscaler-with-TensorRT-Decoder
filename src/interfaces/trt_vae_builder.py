"""Dedicated ComfyUI Node to Build TensorRT RTX VAE Engines directly inside ComfyUI.

Builds a dedicated fixed-shape TensorRT engine pair (encoder + decoder) for ANY
user-specified batch size (e.g. 100, 185, 205 frames). Large batches (>30f) are
traced on CPU float16 so the active CUDA VAE is never disturbed.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from comfy_api.latest import io

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

from ..optimization.memory_manager import get_device_list
from ..core.generation_utils import prepare_runner, setup_generation_context
from ..core.model_loader import materialize_model
from ..utils.debug import Debug
from ..utils.constants import get_base_cache_dir
from ..utils.downloads import download_weight
from ..utils.model_registry import (
    DEFAULT_DIT,
    DEFAULT_VAE,
    get_available_vae_models,
)
from ..models.video_vae_v3.modules.types import MemoryState
from ..models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "tensorrt_backend" / "artifacts"


class _EncoderModule(torch.nn.Module):
    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(video, memory_state=MemoryState.DISABLED)
        if self.quant_conv is not None:
            hidden = self.quant_conv(hidden, memory_state=MemoryState.DISABLED)
        return hidden


class _DecoderModule(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent, memory_state=MemoryState.DISABLED)


def _configure_fixed_vae(vae: torch.nn.Module) -> None:
    if hasattr(vae, "disable_slicing"):
        vae.disable_slicing()
    if hasattr(vae, "set_memory_limit"):
        vae.set_memory_limit(None, None)
    for module in vae.modules():
        if isinstance(module, InflatedCausalConv3d):
            module.set_memory_limit(float("inf"))
            if hasattr(module, "set_memory_device"):
                module.set_memory_device(None)
        if hasattr(module, "slicing"):
            module.slicing = False


def _build_trt_engine(onnx_path: Path, engine_path: Path, workspace_gb: float = 8.0) -> None:
    import tensorrt_rtx as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT-RTX could not parse {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"TensorRT-RTX failed to build engine: {engine_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(blob)


class SeedVR2BuildTensorRTVAE(io.ComfyNode):
    """
    SeedVR2 Build TensorRT VAE Engines Node

    Builds a dedicated fixed-shape TensorRT RTX VAE engine pair (encoder + decoder)
    for any user-specified batch size. The frame count is auto-normalized to 4n+1.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        vae_models = get_available_vae_models()

        return io.Schema(
            node_id="SeedVR2BuildTensorRTVAE",
            display_name="SeedVR2 Build TensorRT VAE Engines",
            category="SEEDVR2",
            description=(
                "Builds a dedicated TensorRT RTX VAE engine pair (encoder + decoder) for any batch size. "
                "The frame count is auto-normalized to 4n+1 (e.g. 100 -> 101, 185 -> 185, 205 -> 205). "
                "Large batches (>30f) are exported via CPU fp16 so the active VAE is protected. "
                "Connect the SEEDVR2_VAE output directly to SeedVR2 Video Upscaler."
            ),
            inputs=[
                io.Combo.Input("model",
                    options=vae_models,
                    default=DEFAULT_VAE,
                    tooltip="VAE model file to convert into TensorRT RTX engines."
                ),
                io.Int.Input("batch_size",
                    default=205,
                    min=5,
                    max=4096,
                    step=4,
                    tooltip="Batch size (frames) to build a dedicated engine for. Auto-normalized to 4n+1 (e.g. 100, 185, 205)."
                ),
                io.Float.Input("workspace_gb",
                    default=8.0,
                    min=1.0,
                    max=32.0,
                    step=0.5,
                    optional=True,
                    tooltip="TensorRT workspace memory allocation in GB during engine compilation (default: 8.0 GB)."
                ),
                io.Boolean.Input("force_rebuild",
                    default=False,
                    optional=True,
                    tooltip="Force rebuilding engines even if they already exist on disk."
                ),
            ],
            outputs=[
                io.Custom("SEEDVR2_VAE").Output(
                    tooltip="VAE configuration ready to connect to SeedVR2 Video Upscaler node."
                ),
                io.String.Output(
                    tooltip="Status and summary of built TensorRT VAE engines."
                )
            ]
        )

    @classmethod
    def execute(cls, model: str, batch_size: int = 205, workspace_gb: float = 8.0, force_rebuild: bool = False) -> io.NodeOutput:
        import importlib.util
        tools_file = ROOT / "tools" / "onnx_export_utils.py"
        if tools_file.exists():
            spec = importlib.util.spec_from_file_location("onnx_export_utils", str(tools_file))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            export_portable_onnx = mod.export_portable_onnx
        else:
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from tools.onnx_export_utils import export_portable_onnx

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        model_dir = Path(get_base_cache_dir())
        model_dir.mkdir(parents=True, exist_ok=True)

        if not (model_dir / model).exists():
            print(f"[SeedVR2] Downloading {model} to {model_dir}...")
            download_weight(DEFAULT_DIT, model, str(model_dir))

        frames = ((batch_size - 1) // 4) * 4 + 1
        lat_frames = (frames - 1) // 4 + 1
        dec_tile_px = 256 if frames >= 21 else 512
        dec_lat_tile = dec_tile_px // 8

        enc_stem = f"vae_encoder_{frames}f_tile512"
        dec_stem = f"vae_decoder_tile_{dec_tile_px}_{frames}f"

        profiles = [
            ("encoder", enc_stem, (1, 3, frames, 512, 512)),
            ("decoder", dec_stem, (1, 16, lat_frames, dec_lat_tile, dec_lat_tile)),
        ]

        needed = []
        for mode, stem, shape in profiles:
            eng_path = ARTIFACTS_DIR / f"{stem}.rtxplan"
            if force_rebuild or not eng_path.exists() or eng_path.stat().st_size < 1_000_000:
                needed.append((mode, stem, shape))

        status_lines = []
        if not needed:
            msg = f"TensorRT VAE engines for {frames}f already built and ready."
            print(f"[SeedVR2] {msg}")
            status_lines.append(msg)
            for mode, stem, shape in profiles:
                eng_path = ARTIFACTS_DIR / f"{stem}.rtxplan"
                size_mb = eng_path.stat().st_size / (1024 * 1024)
                status_lines.append(f" - {stem}.rtxplan ({size_mb:.1f} MB)")
        else:
            debug = Debug(enabled=True)
            ctx = setup_generation_context(dit_device="cuda", vae_device="cuda", debug=debug)
            print(f"[SeedVR2] Initializing VAE structure for TensorRT export...")
            runner, _ = prepare_runner(
                dit_model=DEFAULT_DIT,
                vae_model=model,
                model_dir=str(model_dir),
                debug=debug,
                ctx=ctx,
            )
            cfg = getattr(runner, "config", None) or ctx.get("config")
            materialize_model(runner, "vae", torch.device("cuda"), cfg, debug)
            vae = runner.vae
            vae.eval().to(device="cuda", dtype=torch.float16)
            _configure_fixed_vae(vae)

            pbar = ProgressBar(len(needed)) if ProgressBar is not None else None

            for mode, stem, shape in needed:
                onnx_path = ARTIFACTS_DIR / f"{stem}.onnx"
                eng_path = ARTIFACTS_DIR / f"{stem}.rtxplan"

                print(f"[SeedVR2] Building TensorRT {frames}f {mode} profile...", flush=True)
                t0 = time.perf_counter()

                if frames > 30:
                    import copy
                    print(f"[SeedVR2] Exporting {frames}f {mode} ONNX via CPU fp16 (active VAE protected)...", flush=True)
                    vae_copy = copy.deepcopy(vae).to(device="cpu", dtype=torch.float16)
                    if mode == "encoder":
                        mod_export = _EncoderModule(vae_copy).eval()
                    else:
                        mod_export = _DecoderModule(vae_copy.decoder).eval()
                    dummy = torch.zeros(shape, dtype=torch.float16, device="cpu")
                    export_portable_onnx(mod_export, (dummy,), onnx_path, legacy=True)
                    del mod_export, vae_copy
                else:
                    print(f"[SeedVR2] Exporting {frames}f {mode} ONNX on GPU...", flush=True)
                    if mode == "encoder":
                        mod_export = _EncoderModule(vae).eval().to(device="cuda", dtype=torch.float16)
                    else:
                        mod_export = _DecoderModule(vae.decoder).eval().to(device="cuda", dtype=torch.float16)
                    dummy = torch.zeros(shape, dtype=torch.float16, device="cuda")
                    export_portable_onnx(mod_export, (dummy,), onnx_path, legacy=True)
                    del mod_export, dummy

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                _build_trt_engine(onnx_path, eng_path, workspace_gb=workspace_gb)
                elapsed = time.perf_counter() - t0
                size_mb = eng_path.stat().st_size / (1024 * 1024)
                print(f"[SeedVR2] Successfully built {stem}.rtxplan ({size_mb:.1f} MB in {elapsed:.1f}s)")
                status_lines.append(f" - Built {stem}.rtxplan ({size_mb:.1f} MB, {elapsed:.1f}s)")

                if pbar:
                    pbar.update(1)

            del vae, runner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            from comfy_execution.utils import get_executing_context
            node_id = get_executing_context().node_id
        except Exception:
            node_id = "seedvr2_trt_vae_builder"

        vae_config: Dict[str, Any] = {
            "model": model,
            "device": "cuda:0",
            "offload_device": "none",
            "cache_model": False,
            "encode_tiled": False,
            "encode_tile_size": 512,
            "encode_tile_overlap": 64,
            "decode_tiled": False,
            "decode_tile_size": 512,
            "decode_tile_overlap": 64,
            "tile_debug": "false",
            "use_tensorrt_vae": True,
            "vae_backend": "tensorrt",
            "batch_size": batch_size,
            "node_id": node_id,
        }

        status_text = "\n".join(status_lines)
        return io.NodeOutput(vae_config, status_text)
