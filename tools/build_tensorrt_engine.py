"""Parse an ONNX graph with TensorRT-RTX and build a serialized engine."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt_rtx as trt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    args = parser.parse_args()
    output = args.output or args.onnx.with_suffix(".rtxplan")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse_from_file(str(args.onnx)):
        errors = "\n".join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors))
        raise RuntimeError(f"TensorRT-RTX could not parse {args.onnx}:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30))
    )
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT-RTX failed to build the engine")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(blob)
    print(f"TensorRT-RTX: {trt.__version__}")
    print(f"Parsed layers: {network.num_layers}")
    print(f"Inputs/outputs: {network.num_inputs}/{network.num_outputs}")
    print(f"Engine: {output} ({blob.nbytes / (1024 * 1024):.1f} MiB)")


if __name__ == "__main__":
    main()
