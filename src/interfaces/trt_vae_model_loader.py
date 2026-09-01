"""Dedicated Full-Batch TensorRT RTX Engine Builder for ComfyUI SeedVR2.
Builds dedicated 1-shot TensorRT engines for ANY batch size with minimal VRAM (~400MB) using Studio's compact tracer.
"""

from __future__ import annotations

import copy
import gc
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from comfy_api.latest import io

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


def _available_engine_frames(kind: str = "encoder") -> list[str]:
    """Scan the artifacts dir and auto-populate the engine_frames dropdown.

    kind="encoder" scans vae_encoder_<N>f_tile*.rtxplan,
    kind="decoder" scans vae_decoder_tile_*_<N>f.rtxplan.
    Dropping an engine into tensorrt_backend/artifacts/ is enough to make it
    selectable after a ComfyUI restart.
    """
    import re
    frames = set()
    pattern = "vae_encoder_*f_tile*.rtxplan" if kind == "encoder" else "vae_decoder_tile_*_*f.rtxplan"
    try:
        if ARTIFACTS_DIR.is_dir():
            for pth in ARTIFACTS_DIR.glob(pattern):
                if kind == "encoder":
                    m = re.search(r"_(\d+)f_tile", pth.name)
                else:
                    m = re.search(r"_(\d+)f\.rtxplan", pth.name)
                if m:
                    n = int(m.group(1))
                    # Only 4n+1 frame counts are valid (the exporter normalizes to 4n+1,
                    # so e.g. a file named 195f actually contains a 193f graph).
                    if (n - 1) % 4 == 0:
                        frames.add(str(n))
    except Exception:
        pass
    return ["auto"] + sorted(frames, key=int, reverse=True)


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


def configure_fixed_vae(vae: torch.nn.Module) -> None:
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


def build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_gb: float = 8.0,
    spatial_tile: int = 512,
    frames: int = 21,
    is_decoder: bool = False,
) -> None:
    """Build dedicated TensorRT engine with target spatial tile size."""
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

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        name = inp.name
        if is_decoder:
            lat_f = (frames - 1) // 4 + 1
            lat_tile = spatial_tile // 8
            target_shape = (1, 16, lat_f, lat_tile, lat_tile)
        else:
            target_shape = (1, 3, frames, spatial_tile, spatial_tile)
        profile.set_shape(name, target_shape, target_shape, target_shape)
    config.add_optimization_profile(profile)

    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"TensorRT-RTX failed to build engine: {engine_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(blob)


def get_export_portable_onnx():
    import importlib.util
    tools_file = ROOT / "tools" / "onnx_export_utils.py"
    if tools_file.exists():
        spec = importlib.util.spec_from_file_location("onnx_export_utils", str(tools_file))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.export_portable_onnx
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.onnx_export_utils import export_portable_onnx
    return export_portable_onnx



def _find_existing_dit(model_dir: Path) -> str:
    """Pick an existing DiT checkpoint for structure creation; prefers 7b -> 3b -> any.

    The DiT name is only used to build the (meta) structure; its weights are never
    loaded by the TensorRT engine builder.
    """
    try:
        if Path(model_dir).is_dir():
            candidates = sorted(
                p.name for p in Path(model_dir).glob("seedvr2_*.safetensors")
                if not p.name.endswith(".download")
            )
            if candidates:
                for pref in ("7b", "3b", ""):
                    for name in candidates:
                        if pref in name:
                            return name
    except Exception:
        pass
    return DEFAULT_DIT


def ensure_trt_engine_for_frames(frames: int, vae: torch.nn.Module | None = None, model: str = DEFAULT_VAE, workspace_gb: float = 8.0, dit_model: str | None = None) -> None:
    """Build dedicated static TensorRT RTX engines for the exact batch size (205f, 185f, 100f, etc.).

    - Encoder: (1, 3, frames, 512, 512) fully static
    - Decoder: (1, 16, lat_frames, tile, tile) fully static, tile = 256px (lat 32) for frames>=21 else 512px (lat 64)
    - Large batches (>30f) are traced on CPU float16 so the active CUDA VAE is never disturbed.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = ((frames - 1) // 4) * 4 + 1
    lat_frames = (frames - 1) // 4 + 1

    enc_stem = f"vae_encoder_{frames}f_tile512"
    dec_stem = f"vae_decoder_tile_256_{frames}f" if frames >= 21 else f"vae_decoder_tile_512_{frames}f"
    dec_tile_px = 256 if frames >= 21 else 512
    dec_lat_tile = dec_tile_px // 8

    enc_eng = ARTIFACTS_DIR / f"{enc_stem}.rtxplan"
    dec_eng = ARTIFACTS_DIR / f"{dec_stem}.rtxplan"

    needs_enc = not enc_eng.exists() or enc_eng.stat().st_size < 1_000_000
    needs_dec = not dec_eng.exists() or dec_eng.stat().st_size < 1_000_000

    if not needs_enc and not needs_dec:
        return

    export_portable_onnx = get_export_portable_onnx()
    created_vae = False

    if vae is None:
        model_dir = Path(get_base_cache_dir())
        model_dir.mkdir(parents=True, exist_ok=True)
        if not (model_dir / model).exists():
            print(f"[SeedVR2 TensorRT] Downloading {model} to {model_dir}...")
            download_weight(DEFAULT_DIT, model, str(model_dir))

        debug = Debug(enabled=True)
        ctx = setup_generation_context(dit_device="cuda", vae_device="cuda", debug=debug)
        dit_model = dit_model or _find_existing_dit(model_dir)
        print(f"[SeedVR2 TensorRT] Compiling dedicated {frames}-frame TensorRT RTX engines (DiT structure: {dit_model})...")
        runner, _ = prepare_runner(
            dit_model=dit_model,
            vae_model=model,
            model_dir=str(model_dir),
            debug=debug,
            ctx=ctx,
        )
        cfg = getattr(runner, "config", None) or ctx.get("config")
        materialize_model(runner, "vae", torch.device("cuda"), cfg, debug)
        vae = runner.vae
        vae.eval().to(device="cuda", dtype=torch.float16)
        created_vae = True
    else:
        print(f"[SeedVR2 TensorRT] Compiling dedicated {frames}-frame TensorRT RTX engines from active VAE...")

    configure_fixed_vae(vae)

    if needs_enc:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        onnx_path = ARTIFACTS_DIR / f"{enc_stem}.onnx"
        t0 = time.perf_counter()
        if frames > 30:
            # Large batch: deepcopy to CPU and trace in-process with CUDA hidden.
            # Using 160x160 spatial dummy (~33GB RAM) to guarantee 100% zero OOM within 64GB RAM.
            import copy
            print(f"[SeedVR2 TensorRT] Exporting {frames}f encoder ONNX via CPU fp16 (160x160 trace, 64GB RAM safe)...")
            vae_copy = copy.deepcopy(vae).to(device="cpu", dtype=torch.float16)
            encoder_mod = _EncoderModule(vae_copy).eval()
            dummy = torch.zeros((1, 3, frames, 160, 160), dtype=torch.float16, device="cpu")
            dynamic_axes = {
                "video": {3: "height", 4: "width"},
                "latent_raw": {3: "latent_height", 4: "latent_width"},
            }
            export_portable_onnx(encoder_mod, (dummy,), onnx_path, legacy=True, dynamic_axes=dynamic_axes)
            del encoder_mod, vae_copy
        else:
            print(f"[SeedVR2 TensorRT] Exporting {frames}f encoder ONNX on GPU (512x512)...")
            encoder_mod = _EncoderModule(vae).eval().to(device="cuda", dtype=torch.float16)
            dummy = torch.zeros((1, 3, frames, 512, 512), dtype=torch.float16, device="cuda")
            export_portable_onnx(encoder_mod, (dummy,), onnx_path, legacy=True)
            del encoder_mod, dummy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[SeedVR2 TensorRT] Building {frames}-frame encoder engine: {enc_eng.name} (512x512 tile)...")
        build_trt_engine(onnx_path, enc_eng, workspace_gb=workspace_gb, spatial_tile=512, frames=frames, is_decoder=False)
        print(f"[SeedVR2 TensorRT] Built {enc_eng.name} in {time.perf_counter() - t0:.1f}s")

    if needs_dec:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        onnx_path = ARTIFACTS_DIR / f"{dec_stem}.onnx"
        t0 = time.perf_counter()
        if frames > 30:
            import copy
            print(f"[SeedVR2 TensorRT] Exporting {frames}f decoder ONNX via CPU fp16 (20x20 trace, 64GB RAM safe)...")
            vae_copy = copy.deepcopy(vae).to(device="cpu", dtype=torch.float16)
            decoder_mod = _DecoderModule(vae_copy.decoder).eval()
            dummy = torch.zeros((1, 16, lat_frames, 20, 20), dtype=torch.float16, device="cpu")
            dynamic_axes = {
                "latent": {3: "latent_height", 4: "latent_width"},
                "sample": {3: "height", 4: "width"},
            }
            export_portable_onnx(decoder_mod, (dummy,), onnx_path, legacy=True, dynamic_axes=dynamic_axes)
            del decoder_mod, vae_copy
        else:
            print(f"[SeedVR2 TensorRT] Exporting {frames}f decoder ONNX on GPU ({dec_tile_px}x{dec_tile_px})...")
            decoder_mod = _DecoderModule(vae.decoder).eval().to(device="cuda", dtype=torch.float16)
            dummy = torch.zeros((1, 16, lat_frames, dec_lat_tile, dec_lat_tile), dtype=torch.float16, device="cuda")
            export_portable_onnx(decoder_mod, (dummy,), onnx_path, legacy=True)
            del decoder_mod, dummy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[SeedVR2 TensorRT] Building {frames}-frame decoder engine: {dec_eng.name} ({dec_tile_px}x{dec_tile_px} tile)...")
        build_trt_engine(onnx_path, dec_eng, workspace_gb=workspace_gb, spatial_tile=dec_tile_px, frames=frames, is_decoder=True)
        print(f"[SeedVR2 TensorRT] Built {dec_eng.name} in {time.perf_counter() - t0:.1f}s")

    if created_vae:
        del vae, runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SeedVR2LoadTensorRTVAEModel(io.ComfyNode):
    """
    SeedVR2 Load TensorRT VAE Model Node
    
    Direct drop-in replacement for standard VAE Loader.
    Enables dedicated TensorRT RTX VAE acceleration (2x-5x faster) on NVIDIA GPUs.
    Automatically builds dedicated engines matching your workflow batch size for 1-shot direct execution.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        devices = get_device_list()
        vae_models = get_available_vae_models()

        return io.Schema(
            node_id="SeedVR2LoadTensorRTVAEModel",
            display_name="SeedVR2 Load TensorRT VAE Model",
            category="SEEDVR2",
            description=(
                "Load and configure SeedVR2 VAE with TensorRT RTX acceleration. "
                "Drop-in replacement for standard VAE Loader that provides 2x-5x faster encoding and decoding. "
                "Automatically builds dedicated engines for your batch size for 1-shot maximum speed.\n\n"
                "Connect directly to SeedVR2 Video Upscaler node."
            ),
            inputs=[
                io.Combo.Input("model",
                    options=vae_models,
                    default=DEFAULT_VAE,
                    tooltip="VAE model file for TensorRT acceleration."
                ),
                io.Combo.Input("device",
                    options=devices,
                    default=devices[0],
                    tooltip="GPU device for VAE inference"
                ),
                io.Boolean.Input("encode_tiled",
                    default=False,
                    optional=True,
                    tooltip="TRT パスでは無効（エンジンが自動タイル処理）。フォールバック（通常 VAE）時のみ有効。"
                ),
                io.Int.Input("encode_tile_size",
                    default=512,
                    min=64,
                    step=32,
                    optional=True,
                    tooltip="TRT パスでは無効（エンジンのタイルサイズ 256px/512px を使用）。フォールバック時のみ有効。"
                ),
                io.Int.Input("encode_tile_overlap",
                    default=64,
                    min=0,
                    step=32,
                    optional=True,
                    tooltip="TRT パスでは無効。フォールバック時のみ有効。"
                ),

                io.Combo.Input("tile_debug",
                    options=["false", "encode", "decode"],
                    default="false",
                    optional=True,
                    tooltip="Tile debug visualization mode"
                ),
                io.Combo.Input("engine_frames",
                    options=_available_engine_frames(),
                    default="auto",
                    optional=True,
                    tooltip="TensorRT engine frame size. Auto-populated from engines in tensorrt_backend/artifacts/. auto = pick the largest available engine."
                ),
            ],
            outputs=[
                io.Custom("SEEDVR2_VAE").Output(
                    tooltip="VAE configuration ready to connect to SeedVR2 Video Upscaler node (with TensorRT enabled)."
                )
            ]
        )

    @classmethod
    def execute(
        cls,
        model: str,
        device: str,
        encode_tiled: bool = False,
        encode_tile_size: int = 512,
        encode_tile_overlap: int = 64,
        tile_debug: str = "false",
        engine_frames: str = "auto",
    ) -> io.NodeOutput:
        try:
            from comfy_execution.utils import get_executing_context
            node_id = get_executing_context().node_id
        except Exception:
            node_id = "seedvr2_trt_vae"

        vae_config: Dict[str, Any] = {
            "model": model,
            "device": device,
            "offload_device": "none",
            "cache_model": False,
            "encode_tiled": encode_tiled,
            "encode_tile_size": encode_tile_size,
            "encode_tile_overlap": encode_tile_overlap,
            "tile_debug": tile_debug,
            "engine_frames": engine_frames,
            "use_tensorrt_vae": True,
            "vae_backend": "tensorrt",
            "node_id": node_id,
        }

        return io.NodeOutput(vae_config)


class SeedVR2LoadTensorRTVAEDecoder(io.ComfyNode):
    """Decoder-only TensorRT VAE config (separate engine frame size from the encoder)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        devices = get_device_list()
        vae_models = get_available_vae_models()
        return io.Schema(
            node_id="SeedVR2LoadTensorRTVAEDecoder",
            display_name="SeedVR2 Load TensorRT VAE Decoder",
            category="SEEDVR2",
            description=(
                "Decoder-only TensorRT VAE configuration. Lets you choose a different "
                "engine frame size for decoding than for encoding (e.g. encode 89f / decode 65f). "
                "Connect to the vae_decode input of SeedVR2 Video Upscaler."
            ),
            inputs=[
                io.Combo.Input("model",
                    options=vae_models,
                    default=DEFAULT_VAE,
                    tooltip="VAE model file."
                ),
                io.Combo.Input("device",
                    options=devices,
                    default=devices[0],
                    tooltip="GPU device for VAE inference"
                ),
                io.Boolean.Input("decode_tiled",
                    default=False,
                    optional=True,
                    tooltip="TRT パスでは無効（エンジンが自動タイル処理）。フォールバック時のみ有効。"
                ),
                io.Int.Input("decode_tile_size",
                    default=512,
                    min=64,
                    step=32,
                    optional=True,
                    tooltip="TRT パスでは無効（エンジンのタイルサイズ 256px/512px を使用）。フォールバック時のみ有効。"
                ),
                io.Int.Input("decode_tile_overlap",
                    default=64,
                    min=0,
                    step=32,
                    optional=True,
                    tooltip="TRT パスでは無効。フォールバック時のみ有効。"
                ),
                io.Combo.Input("engine_frames",
                    options=_available_engine_frames("decoder"),
                    default="auto",
                    optional=True,
                    tooltip="TensorRT decoder engine frame size. Auto-populated from artifacts. "
                            "auto = pick the largest available engine."
                ),
            ],
            outputs=[
                io.Custom("SEEDVR2_VAE").Output(
                    tooltip="VAE configuration for the decoder path."
                )
            ]
        )

    @classmethod
    def execute(cls, model: str, device: str, decode_tiled: bool = False,
                decode_tile_size: int = 512, decode_tile_overlap: int = 64,
                engine_frames: str = "auto") -> io.NodeOutput:
        try:
            from comfy_execution.utils import get_executing_context
            node_id = get_executing_context().node_id
        except Exception:
            node_id = "seedvr2_trt_vae_decoder"

        vae_config: Dict[str, Any] = {
            "model": model,
            "device": device,
            "offload_device": "none",
            "cache_model": False,
            "decode_tiled": decode_tiled,
            "decode_tile_size": decode_tile_size,
            "decode_tile_overlap": decode_tile_overlap,
            "tile_debug": "false",
            "use_tensorrt_vae": True,
            "vae_backend": "tensorrt",
            "engine_frames": engine_frames,
            "node_id": node_id,
        }
        return io.NodeOutput(vae_config)
