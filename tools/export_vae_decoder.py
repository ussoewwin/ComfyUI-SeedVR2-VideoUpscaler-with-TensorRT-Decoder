"""Export a fixed-shape SeedVR2 VAE decoder graph for TensorRT."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure torch/lib is at the very front of PATH and DLL search to prevent version mismatch
torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if torch_lib.exists():
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
tools_dir = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

from src.core.generation_utils import prepare_runner, setup_generation_context  # noqa: E402
from src.core.model_loader import materialize_model  # noqa: E402
from src.utils.debug import Debug  # noqa: E402
from src.utils.model_registry import DEFAULT_DIT, DEFAULT_VAE  # noqa: E402
from src.models.video_vae_v3.modules.types import MemoryState  # noqa: E402
from src.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d  # noqa: E402
from onnx_export_utils import export_portable_onnx  # noqa: E402


class Decoder(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent, memory_state=MemoryState.DISABLED)


def configure_fixed_profile(vae: torch.nn.Module) -> None:
    """Disable adaptive VAE slicing for a fixed-shape export profile."""
    if hasattr(vae, "disable_slicing"):
        vae.disable_slicing()
    if hasattr(vae, "set_memory_limit"):
        vae.set_memory_limit(None, None)
    for module in vae.modules():
        if isinstance(module, InflatedCausalConv3d):
            module.set_memory_limit(float("inf"))
            module.set_memory_device(None)
        if hasattr(module, "slicing"):
            module.slicing = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--latent-frames", type=int, default=2)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--dit-model", type=str, default=DEFAULT_DIT)
    parser.add_argument("--vae-model", type=str, default=DEFAULT_VAE)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "SEEDVR2")
    parser.add_argument("--output", type=Path, default=ROOT / "tensorrt_backend" / "artifacts" / "vae_decoder.onnx")
    parser.add_argument("--legacy-export", action="store_true", default=True)
    args = parser.parse_args()

    debug = Debug(enabled=True)
    setup_generation_context(dit_device="cuda", vae_device="cuda", offload_device=None, debug=debug)
    runner = prepare_runner(
        dit_model=args.dit_model,
        vae_model=args.vae_model,
        model_dir=str(args.model_dir),
        debug=debug,
    )
    vae = runner.vae
    materialize_model(vae, target_device=torch.device("cuda"), debug=debug)
    vae.eval()
    vae.to(device="cuda", dtype=torch.float16)
    configure_fixed_profile(vae)

    decoder = Decoder(vae.decoder).eval().to(device="cuda", dtype=torch.float16)
    latent_h = args.height // 8
    latent_w = args.width // 8
    dummy_latent = torch.zeros(
        (1, args.channels, args.latent_frames, latent_h, latent_w),
        dtype=torch.float16,
        device="cuda",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_portable_onnx(decoder, (dummy_latent,), args.output, legacy=args.legacy_export)
    print(f"Exported VAE decoder ONNX to {args.output}")


if __name__ == "__main__":
    main()
