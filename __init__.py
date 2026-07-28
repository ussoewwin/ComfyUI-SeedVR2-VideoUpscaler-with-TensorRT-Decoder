"""
ComfyUI-SeedVR2_VideoUpscaler
Official SeedVR2 integration for ComfyUI
"""

import sys
import subprocess

# Check critical dependencies early to provide better error messages
# and auto-install if possible, especially useful for Vast.ai / RunPod
def ensure_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name.split(">")[0].split("=")[0].split("<")[0]
    
    try:
        __import__(import_name)
        return  # Already available
    except (ImportError, ModuleNotFoundError):
        pass
    if True:  # Package is missing - install it
        print("\n" + "="*80)
        print(f"SeedVR2: '{import_name}' module not found.")
        print(f"SeedVR2: Current Python executable: {sys.executable}")
        print(f"SeedVR2: Attempting automatic installation of {package_name}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
            print(f"SeedVR2: Successfully installed {package_name}")
        except Exception as e:
            print(f"SeedVR2: Auto-installation failed: {e}")
            print("This often happens on Vast.ai / RunPod when pip installs to a different Python environment.")
            print(f"Please run the following command manually in your terminal:")
            print(f"    {sys.executable} -m pip install \"{package_name}\"")
        print("="*80 + "\n")

# All critical dependencies from requirements.txt
# (torch/torchvision/numpy are assumed present via ComfyUI)
_REQUIRED_PACKAGES = [
    ("safetensors", None),
    ("tqdm", None),
    ("psutil", None),
    ("einops", None),
    ("omegaconf>=2.3.0", "omegaconf"),
    ("diffusers>=0.33.1", "diffusers"),
    ("transformers", None),
    ("accelerate", None),
    ("peft>=0.17.0", "peft"),
    ("rotary_embedding_torch>=0.5.3", "rotary_embedding_torch"),
    ("opencv-python", "cv2"),
    ("gguf", None),
    ("matplotlib", None),
]

for pkg, imp in _REQUIRED_PACKAGES:
    ensure_package(pkg, imp)

# Windows cp932: patch inductor jinja open(encoding=utf-8) before any torch.compile
try:
    from .src.core.fix_inductor import _fix_inductor_windows_encoding

    _fix_inductor_windows_encoding()
except Exception as _seedvr2_inductor_fix_err:  # noqa: BLE001
    print(f"[SeedVR2] Warning: inductor Windows encoding fix skipped: {_seedvr2_inductor_fix_err}")

from .src.optimization.compatibility import ensure_triton_compat  # noqa: F401
from .src.interfaces import comfy_entrypoint, SeedVR2Extension

__all__ = ["comfy_entrypoint", "SeedVR2Extension"]