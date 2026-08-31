"""Portable, observable ONNX export helpers for TensorRT engine preparation."""

from __future__ import annotations

import contextlib
import gc
import time
from contextlib import nullcontext
from pathlib import Path

import torch


def _register_custom_symbolics():
    try:
        from torch.onnx import register_custom_op_symbolic

        def symbolic_cudnn_convolution(g, input, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32):
            return g.op(
                "Conv",
                input,
                weight,
                pads_i=[p for p in padding for _ in (0, 1)] if isinstance(padding, (list, tuple)) else None,
                strides_i=stride if isinstance(stride, (list, tuple)) else None,
                dilations_i=dilation if isinstance(dilation, (list, tuple)) else None,
                group_i=groups if isinstance(groups, int) else 1,
            )

        register_custom_op_symbolic("::cudnn_convolution", symbolic_cudnn_convolution, 20)
        register_custom_op_symbolic("aten::cudnn_convolution", symbolic_cudnn_convolution, 20)
    except Exception:
        pass


_register_custom_symbolics()


def _math_attention_context():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except (ImportError, AttributeError):
        return nullcontext()


def _portable_export(module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: Path, *, legacy: bool, dynamic_axes: dict | None = None) -> None:
    is_encoder = args[0].ndim == 5 and args[0].shape[1] == 3
    with torch.inference_mode(), torch.no_grad(), _math_attention_context():
        # Free traced intermediates before graph serialization to cap commit growth.
        gc.collect()
        torch.onnx.export(
            module,
            args,
            str(output),
            input_names=["video"] if is_encoder else ["latent"],
            output_names=["latent_raw"] if is_encoder else ["sample"],
            opset_version=20,
            dynamo=not legacy,
            optimize=False,
            do_constant_folding=False,
            dynamic_axes=dynamic_axes,
        )


@contextlib.contextmanager
def _cuda_disabled():
    """Force CPU-only tracing by hiding CUDA at every level.

    v2: besides stubbing torch.cuda.is_available/device_count, we also patch
    Tensor.cuda(), Tensor.to(device=...cuda...) and the common tensor factory
    functions so that ANY code path that tries to allocate on the GPU (with or
    without an is_available() guard) silently falls back to CPU for the duration
    of the trace. This closes the case where a module holds a plain-attribute
    CUDA tensor or a forward() references cuda:0 directly.
    """
    import torch as _torch

    orig_avail = _torch.cuda.is_available
    orig_count = _torch.cuda.device_count
    orig_init = getattr(_torch.cuda, "init", None)
    orig_cuda_method = _torch.Tensor.cuda
    orig_to_method = _torch.Tensor.to
    orig_new_method = _torch.Tensor.new
    factories = {
        "zeros": _torch.zeros, "ones": _torch.ones, "empty": _torch.empty,
        "randn": _torch.randn, "rand": _torch.rand, "tensor": _torch.tensor,
        "full": _torch.full, "arange": _torch.arange, "eye": _torch.eye,
        "randint": _torch.randint, "linspace": _torch.linspace,
        "zeros_like": _torch.zeros_like, "ones_like": _torch.ones_like,
        "empty_like": _torch.empty_like, "randn_like": _torch.randn_like,
    }
    orig_factories = dict(factories)

    def _force_cpu_device(device):
        if device is None:
            return None
        if isinstance(device, str):
            return "cpu" if "cuda" in device else device
        if isinstance(device, _torch.device):
            return _torch.device("cpu") if device.type == "cuda" else device
        return device

    def _patched_cuda(self, *a, **k):
        return self

    def _patched_to(self, *a, **k):
        new_args = tuple(_force_cpu_device(x) if isinstance(x, (str, _torch.device)) else x for x in a)
        new_k = dict(k)
        if "device" in new_k:
            new_k["device"] = _force_cpu_device(new_k["device"])
        return orig_to_method(self, *new_args, **new_k)

    def _patched_new(self, *a, **k):
        if "device" in k:
            k["device"] = _force_cpu_device(k["device"])
        return orig_new_method(self, *a, **k)

    def _make_factory_patch(orig):
        def wrapper(*a, **k):
            if "device" in k:
                k["device"] = _force_cpu_device(k["device"])
            return orig(*a, **k)
        return wrapper

    try:
        _torch.cuda.is_available = lambda: False
        _torch.cuda.device_count = lambda: 0
        if orig_init is not None:
            _torch.cuda.init = lambda: None
        try:
            _torch.cuda._lazy_init = lambda *a, **k: None
        except Exception:
            pass
        _torch.Tensor.cuda = _patched_cuda
        _torch.Tensor.to = _patched_to
        _torch.Tensor.new = _patched_new
        for name, orig in orig_factories.items():
            setattr(_torch, name, _make_factory_patch(orig))
        yield
    finally:
        _torch.cuda.is_available = orig_avail
        _torch.cuda.device_count = orig_count
        if orig_init is not None:
            _torch.cuda.init = orig_init
        _torch.Tensor.cuda = orig_cuda_method
        _torch.Tensor.to = orig_to_method
        _torch.Tensor.new = orig_new_method
        for name, orig in orig_factories.items():
            setattr(_torch, name, orig)




def _force_cpu_fp16(module: torch.nn.Module, args: tuple[torch.Tensor, ...]):
    """Force module and args to CPU float16.

    .to("cpu") only moves parameters and buffers; some modules keep plain-attribute
    tensors (caches, scratch, etc.) on CUDA, which would drag the trace onto the GPU.
    We scan every submodule attribute and move any stray CUDA tensor as well.
    """
    module = module.to(device="cpu", dtype=torch.float16)
    for m in module.modules():
        for key, value in list(vars(m).items()):
            if isinstance(value, torch.Tensor) and value.is_cuda:
                setattr(m, key, value.detach().to(device="cpu", dtype=torch.float16))
            elif isinstance(value, (list, tuple)):
                moved = []
                changed = False
                for item in value:
                    if isinstance(item, torch.Tensor) and item.is_cuda:
                        moved.append(item.detach().to(device="cpu", dtype=torch.float16))
                        changed = True
                    else:
                        moved.append(item)
                if changed:
                    setattr(m, key, type(value)(moved))
    cpu_args = tuple(a.detach().to(device="cpu", dtype=torch.float16) for a in args)
    return module, cpu_args


def export_portable_onnx(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output: Path,
    *,
    legacy: bool = True,
    dynamic_axes: dict | None = None,
) -> None:
    """Export ONNX.

    - Large batches (>30 frames): always traced on CPU float16 (system RAM) so a
      16GB GPU can never OOM while building e.g. a 205f engine.
    - Standard batches (5f/21f): traced on GPU via CUDA/cuDNN, with automatic CPU
      fp16 fallback if the GPU runs out of memory.
    """
    is_large_batch = args[0].shape[2] > 30
    started = time.perf_counter()

    if is_large_batch:
        print(f"[SeedVR2 TensorRT] Large batch detected ({args[0].shape[2]} frames). Exporting ONNX via CPU fp16 (system RAM, CUDA-hide v2)...", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cpu_module, cpu_args = _force_cpu_fp16(module, args)
        with _cuda_disabled():
            _portable_export(cpu_module, cpu_args, output, legacy=legacy, dynamic_axes=dynamic_axes)
        del cpu_module, cpu_args
        gc.collect()
        print(f"[SeedVR2 TensorRT] ONNX export finished in {time.perf_counter() - started:.1f}s", flush=True)
        return

    print(f"[SeedVR2 TensorRT] Exporting ONNX graph (CUDA/cuDNN) -> {output.name}...", flush=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        _portable_export(module, args, output, legacy=legacy, dynamic_axes=dynamic_axes)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print("[SeedVR2 TensorRT] CUDA VRAM limit reached during ONNX trace; falling back to CPU fp16 export...", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cpu_module, cpu_args = _force_cpu_fp16(module, args)
        _portable_export(cpu_module, cpu_args, output, legacy=legacy, dynamic_axes=dynamic_axes)
        del cpu_module, cpu_args
        gc.collect()
    print(f"[SeedVR2 TensorRT] ONNX export finished in {time.perf_counter() - started:.1f}s", flush=True)
