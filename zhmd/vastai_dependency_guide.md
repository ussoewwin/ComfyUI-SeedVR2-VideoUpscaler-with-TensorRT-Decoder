# ComfyUI 云环境依赖错误完整说明

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../md/vastai_dependency_guide.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

本文完整说明在 Vast.ai、RunPod 等云环境中运行 SeedVR2（以及其他依赖较重的自定义节点）时，经常遇到的 `ModuleNotFoundError`：根因是什么，以及本仓库如何用代码级自动修复解决。

---

## 1. 错误现象与发生方式

**[错误]**
```python
ModuleNotFoundError: No module named 'diffusers'
ModuleNotFoundError: No module named 'rotary_embedding_torch'
```

即使用户刚在 Vast.ai 终端执行了 `pip install diffusers`，并且看到 **"Successfully installed"**，ComfyUI 加载 SeedVR2 时仍可能报上述错误。

### 核心原因（两个因素的叠加）

这是 ComfyUI 的设计理念与云模板环境结构「完美碰撞」的结果。

#### 因素 A：ComfyUI 的「反膨胀」哲学
在 AI 图像生成生态中，`diffusers`、`transformers`、`accelerate` 等是**行业标准**（Automatic1111 就建立在其上）。
但 ComfyUI 的设计理念恰好相反：保持精简、快速，尽量只依赖原生 PyTorch，避免厚重抽象与膨胀。因此 **ComfyUI 官方核心依赖故意不包含 `diffusers` 等库**。
结果是：云平台提供的「干净」ComfyUI 环境默认不会安装这些标准库。

#### 因素 B：「看不见」的多重 Python 环境
Vast.ai 上的 ComfyUI 模板通常把计算环境分成多层：
1. **终端（用户可见）**：系统 Python（例如 `/usr/bin/python`）。
2. **ComfyUI 执行进程**：隐藏的虚拟环境（例如 `/workspace/ComfyUI/venv/bin/python`）。

用户在终端输入 `pip install` 时，命令总是打到「系统」环境；而 ComfyUI 实际用「执行进程」环境加载节点。用户很难从 UI 精确指定隐藏 venv，因此手动安装经常装错位置，最终仍出现 `ModuleNotFoundError`。

---

## 2. 解决方案（代码改动说明）

为打破该陷阱，我们大幅修改了 SeedVR2 的启动入口（`__init__.py`）。

**修改文件：** `/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/__init__.py`

### 实现细节

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

## 3. 为何有效

下面说明这段代码如何解决环境错位：

### 1. 用 `sys.executable` 定位「真正路径」
修复的关键是使用 `sys.executable`。
`sys.executable` 指向**当前正在执行该脚本的 Python 解释器**（即运行 ComfyUI 的隐藏 `venv` 路径）。

通过 `subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])`，完全绕过终端环境，**强制把包装进当前活跃的 ComfyUI 虚拟环境**，从而解决「看不见的安装目标」问题。

### 2. 用 `try / __import__()` 做可靠检测
标准的 `importlib.find_spec()` 在包损坏、不完整安装时可能误报「已存在」。
把真正的 `__import__()` 包在 `try/except` 中，可以保证只有模块确实无法加载（`ModuleNotFoundError`）时才触发自动安装。

### 3. 处理安装名与导入名不一致
部分包 pip 名与 import 名不同（例如 `pip install opencv-python` 与 `import cv2`）。
`_REQUIRED_PACKAGES` 显式区分「pip 包名」与「导入检查名」，避免因假阴性陷入无限安装循环。

### 总结
有了本更新，即使在 Vast.ai / RunPod 上用 `git clone` 手动安装 SeedVR2，也会在 ComfyUI 启动时自愈：扫描内部环境、发现缺失依赖，并精确安装到正确位置，用户无需再排查隐藏 Python 路径。
