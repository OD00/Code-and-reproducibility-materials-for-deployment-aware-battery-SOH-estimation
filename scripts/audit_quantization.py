"""Audit INT8 tensor quantization parameters in the four deployment models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb


def tensor_summary(detail: dict) -> dict:
    q = detail["quantization_parameters"]
    scales = np.asarray(q["scales"])
    zero_points = np.asarray(q["zero_points"])
    return {
        "name": detail["name"],
        "shape": [int(value) for value in detail["shape"]],
        "dtype": np.dtype(detail["dtype"]).name,
        "scales": [float(value) for value in scales],
        "zero_points": [int(value) for value in zero_points],
        "quantized_dimension": int(q["quantized_dimension"]),
        "strategy": (
            "per_channel" if len(scales) > 1 else "per_tensor" if len(scales) == 1 else "none"
        ),
    }


def audit(path: Path) -> dict:
    model_bytes = path.read_bytes()
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    detail_by_name = {item["name"]: item for item in interpreter.get_tensor_details()}

    model = schema_fb.Model.GetRootAsModel(model_bytes, 0)
    subgraph = model.Subgraphs(0)
    constant_names: set[str] = set()
    for index in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(index)
        buffer = model.Buffers(tensor.Buffer())
        if buffer is not None and buffer.DataLength() > 0:
            constant_names.add(tensor.Name().decode("utf-8"))

    counts = {
        "constant_int8_per_channel": 0,
        "constant_int8_per_tensor": 0,
        "runtime_int8_per_channel": 0,
        "runtime_int8_per_tensor": 0,
    }
    for name, detail in detail_by_name.items():
        if np.dtype(detail["dtype"]) != np.dtype(np.int8):
            continue
        scales = np.asarray(detail["quantization_parameters"]["scales"])
        if len(scales) == 0:
            continue
        prefix = "constant" if name in constant_names else "runtime"
        suffix = "per_channel" if len(scales) > 1 else "per_tensor"
        counts[f"{prefix}_int8_{suffix}"] += 1

    return {
        "path": str(path.resolve()),
        "size_bytes": len(model_bytes),
        "inputs": [tensor_summary(item) for item in input_details],
        "outputs": [tensor_summary(item) for item in output_details],
        "quantized_tensor_counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for item in args.model:
        name, raw_path = item.split("=", 1)
        result[name] = audit(Path(raw_path))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
