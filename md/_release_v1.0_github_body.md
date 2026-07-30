This document provides a complete explanation of the `ModuleNotFoundError` issues often encountered when running SeedVR2 (and other custom nodes with heavy external dependencies) on cloud environments like Vast.ai or RunPod. It details the root causes and explains the automated code-level fix implemented in this repository.

---

## 1. The Error and How It Occurs

**[The Error]**
```python
ModuleNotFoundError: No module named 'diffusers'
ModuleNotFoundError: No module named 'rotary_embedding_torch'
```

Users often encounter this error when ComfyUI attempts to load SeedVR2, even if they have just run `pip install diffusers` in their Vast.ai terminal and saw a **"Successfully installed"** message.

### Core Causes (A Clash of Two Factors)

This error is the result of a "perfect storm" between ComfyUI's core design philosophy and the way cloud environment templates are structured.

#### Factor A: ComfyUI's "Anti-Bloat" Philosophy
In the AI image generation ecosystem, libraries like `diffusers`, `transformers`, and `accelerate` are **absolute industry standards** (Automatic1111 runs on them). 
However, ComfyUI was designed with exactly the opposite philosophy: to run lean, fast, and purely on raw PyTorch, avoiding heavy abstractions and bloat. Therefore, **ComfyUI intentionally excludes `diffusers` and similar libraries from its official core requirements.**
As a result, a "clean" ComfyUI environment provided by cloud platforms will not have these standard libraries installed by default.

#### Factor B: "Invisible" Multiple Python Environments
ComfyUI templates on Vast.ai typically separate the computing environment into multiple layers:
1. **Terminal (User-facing)**: The system Python (e.g., `/usr/bin/python`).
2. **ComfyUI Execution Process**: A hidden Python virtual environment (e.g., `/workspace/ComfyUI/venv/bin/python`).

When a user opens a terminal and types `pip install`, the command always targets the "System" environment. However, ComfyUI actually loads nodes using the "Execution Process" environment. Because users do not have an easy way to target the hidden virtual environment from the UI, manual installation attempts fall into the wrong destination, causing the `ModuleNotFoundError`.

---

## 2. The Solution (Code Modification Details)

To break through this design trap, we heavily modified the startup entry point of SeedVR2 (`__init__.py`).

**Modified File:** `/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/__init__.py`

### Implementation Details

```python
import sys
import subprocess

def ensure_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name.split(">")[0].split("=")[0].split("<")[0]
    
    try:
        # First, attempt to actually import the module
        __import__(import_name)
        return  # Already available
    except (ImportError, ModuleNotFoundError):
        pass
    
    # Package is missing - install it
    print("\n" + "="*80)
    print(f"SeedVR2: '{import_name}' module not found.")
    print(f"SeedVR2: Current Python executable: {sys.executable}")
    print(f"SeedVR2: Attempting automatic installation of {package_name}...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        print(f"SeedVR2: Successfully installed {package_name}")
    except Exception as e:
        print(f"SeedVR2: Auto-installation failed: {e}")
    print("="*80 + "\n")

# Register all dependencies from requirements.txt for auto-installation
_REQUIRED_PACKAGES = [
    ("safetensors", None),
    ("tqdm", None),
    ("omegaconf>=2.3.0", "omegaconf"),
    ("diffusers>=0.33.1", "diffusers"),
    ("transformers", None),
    ("accelerate", None),
    ("peft>=0.17.0", "peft"),
    ("rotary_embedding_torch>=0.5.3", "rotary_embedding_torch"),
    ("opencv-python", "cv2"), # Mapping pip name to import name
    ("gguf", None),
]

for pkg, imp in _REQUIRED_PACKAGES:
    ensure_package(pkg, imp)
```

---

## 3. How and Why This Works

Here is a breakdown of why this specific code resolves the environment mismatch:

### 1. Identifying the "True Path" via `sys.executable`
The most critical part of this fix is the use of `sys.executable`. 
`sys.executable` points to **the exact path of the Python interpreter that is currently executing the script** (which is the hidden `venv` path running ComfyUI).

By running `subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])`, we completely bypass the terminal environment and **force pip to install the package directly into the currently active ComfyUI virtual environment.** This solves the "invisible destination" problem entirely.

### 2. Reliable Detection via `try / __import__()`
The standard `importlib.find_spec()` can sometimes return false positives if a package is broken, corrupted, or partially installed. 
By wrapping an actual `__import__()` call in a `try/except` block, we guarantee that the auto-installer only triggers if the module is genuinely unable to load (`ModuleNotFoundError`).

### 3. Handling Name Discrepancies
Some Python packages have different names for installation via pip and for importing in code (e.g., `pip install opencv-python` vs `import cv2`). 
The `_REQUIRED_PACKAGES` array explicitly separates the "pip package name" from the "import check name." This precise mapping prevents the script from entering an infinite installation loop caused by false negatives.

### Summary
With this update, even if SeedVR2 is manually installed via `git clone` on Vast.ai or RunPod, it will self-heal during ComfyUI startup. It scans its inner environment, detects missing dependencies, and forcefully injects them precisely where they belong, eliminating the need for users to troubleshoot hidden Python paths.

