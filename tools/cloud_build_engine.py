"""Cloud TensorRT engine builder for large-batch VAE engines.

Builds a .rtxplan from a static ONNX graph on a high-VRAM cloud GPU
(e.g. 80GB A100/H100) to avoid local 16GB VRAM workspace limits.

!!! CRITICAL !!!
TensorRT engines are GPU-architecture specific. An engine built on a cloud GPU
will NOT run on a local RTX 5060 Ti. If your cloud GPU is a different
architecture, use this script ONLY to build the ONNX (see export_onnx_worker.py)
and build the engine locally instead.

Usage:
    python cloud_build_engine.py <onnx_path> --output <engine_path> [--workspace-gb 24]

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
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))

    t0 = time.perf_counter()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        print("ERROR: engine build failed (workspace too small or graph unsupported)", flush=True)
        return 1
    dt = time.perf_counter() - t0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(blob)
    print(f"OK: {args.output} ({blob.nbytes / 2**20:.1f} MiB in {dt:.1f}s)", flush=True)
    print("WARNING: this engine is GPU-architecture specific; it may not run on other GPUs.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
