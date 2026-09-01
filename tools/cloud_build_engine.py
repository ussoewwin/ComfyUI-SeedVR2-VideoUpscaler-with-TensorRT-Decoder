"""Cloud TensorRT engine builder for large-batch VAE engines.

Builds a .rtxplan from a static ONNX graph on a high-VRAM cloud GPU
(e.g. 80GB A100/H100) to avoid local 16GB VRAM workspace limits.

!!! CRITICAL !!!
TensorRT engines are GPU-architecture specific. An engine built on a cloud GPU
will NOT run on a local RTX 5060 Ti. If your cloud GPU is a different
architecture, use this script ONLY to build the ONNX (see export_onnx_worker.py)
and build the engine locally instead.

Usage:
    python cloud_build_engine.py <onnx_path> --output <engine_path> [--workspace-gb 24] [--min-ws]

Setup (first time):
    pip install -r cloud_requirements.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path, help="input static ONNX graph")
    parser.add_argument("--output", type=Path, required=True, help="output .rtxplan path")
    parser.add_argument("--workspace-gb", type=float, default=24.0,
                        help="TensorRT workspace limit in GB (default 24)")
    parser.add_argument("--min-ws", action="store_true",
                        help="binary-search the smallest workspace that still builds (minimizes runtime VRAM)")
    args = parser.parse_args()

    import torch
    import tensorrt_rtx as trt

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available on this machine", flush=True)
        return 2

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"VRAM: {props.total_memory / 2**30:.1f} GiB, arch: {props.major}.{props.minor}", flush=True)
    print(f"TensorRT-RTX: {trt.__version__}", flush=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse_from_file(str(args.onnx)):
        errors = "\n".join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors))
        print(f"ERROR: TensorRT could not parse {args.onnx}:\n{errors}", flush=True)
        return 1

    config = builder.create_builder_config()

    def _build(ws_gb: float):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(ws_gb * (1 << 30)))
        return builder.build_serialized_network(network, config)

    t0 = time.perf_counter()
    if args.min_ws:
        # Binary search for the smallest workspace that still builds.
        lo, hi = 2.0, args.workspace_gb
        blob = _build(hi)
        if blob is None:
            while blob is None and hi <= 48:
                hi += 2.0
                blob = _build(hi)
        if blob is None:
            print("ERROR: engine build failed even at high workspace", flush=True)
            return 1
        while hi - lo > 0.5:
            mid = (lo + hi) / 2
            b = _build(mid)
            if b is not None:
                blob = b
                hi = mid
            else:
                lo = mid
        ws_used = hi
        print(f"Min workspace: {ws_used:.1f} GB (smallest that builds)", flush=True)
    else:
        blob = _build(args.workspace_gb)
        ws_used = args.workspace_gb
    dt = time.perf_counter() - t0

    if blob is None:
        print("ERROR: engine build failed (workspace too small or graph unsupported)", flush=True)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(blob)
    print(f"OK: {args.output} ({blob.nbytes / 2**20:.1f} MiB in {dt:.1f}s)", flush=True)
    print("WARNING: this engine is GPU-architecture specific; it may not run on other GPUs.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
