"""Generate TFLite C arrays and fixed Pico 2 firmware test fixtures."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

from paper_experiments.common.models import ALL_MODEL_NAMES
from paper_experiments.experiment_3_mcu_deployment.convert_tensorflow_int8 import interpreter_class, quantize


def c_array(name: str, data: bytes, width: int = 12) -> str:
    lines = []
    for index in range(0, len(data), width):
        lines.append("  " + ", ".join(f"0x{value:02x}" for value in data[index : index + width]) + ",")
    return f"alignas(16) const unsigned char {name}[] = {{\n" + "\n".join(lines) + f"\n}};\nconst unsigned int {name}_len = {len(data)};\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--model", choices=ALL_MODEL_NAMES, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Optional optimized-model directory containing model_int8.tflite",
    )
    parser.add_argument(
        "--fixed-test-set",
        type=Path,
        help="Optional fixed_test_set.npz override for an optimized-model directory",
    )
    parser.add_argument("--fixture-samples", type=int, default=64)
    parser.add_argument("--firmware-dir-name", default="firmware_assets")
    args = parser.parse_args()
    prepared = args.prepared_dir.resolve()
    model_dir = (
        args.model_dir.resolve()
        if args.model_dir is not None
        else prepared / "models" / args.model
    )
    tflite_path = model_dir / "model_int8.tflite"
    if not tflite_path.exists():
        raise FileNotFoundError(tflite_path)
    symbol = "g_model"
    firmware_dir = model_dir / args.firmware_dir_name
    firmware_dir.mkdir(parents=True, exist_ok=False)
    (firmware_dir / "model_data.h").write_text(
        f"#pragma once\nextern const unsigned char {symbol}[];\nextern const unsigned int {symbol}_len;\n",
        encoding="utf-8",
    )
    (firmware_dir / "model_data.cc").write_text(
        '#include "model_data.h"\n' + c_array(symbol, tflite_path.read_bytes()), encoding="utf-8"
    )

    Interpreter = interpreter_class()
    interpreter = Interpreter(model_content=tflite_path.read_bytes())
    interpreter.allocate_tensors()
    op_names = []
    for item in interpreter._get_ops_details():
        name = item["op_name"]
        if name == "DELEGATE":
            continue
        if name not in op_names:
            op_names.append(name)
    resolver_methods = {
        "ADD": "AddAdd",
        "BATCH_MATMUL": "AddBatchMatMul",
        "CONCATENATION": "AddConcatenation",
        "CONV_2D": "AddConv2D",
        "DEPTHWISE_CONV_2D": "AddDepthwiseConv2D",
        "DEQUANTIZE": "AddDequantize",
        "EXP": "AddExp",
        "FULLY_CONNECTED": "AddFullyConnected",
        "LOG": "AddLog",
        "LOGISTIC": "AddLogistic",
        "MAX_POOL_2D": "AddMaxPool2D",
        "MEAN": "AddMean",
        "MUL": "AddMul",
        "PACK": "AddPack",
        "PAD": "AddPad",
        "QUANTIZE": "AddQuantize",
        "RELU": "AddRelu",
        "RESHAPE": "AddReshape",
        "RSQRT": "AddRsqrt",
        "SOFTMAX": "AddSoftmax",
        "SPLIT": "AddSplit",
        "SPLIT_V": "AddSplitV",
        "SQUARED_DIFFERENCE": "AddSquaredDifference",
        "STRIDED_SLICE": "AddStridedSlice",
        "SUB": "AddSub",
        "SUM": "AddSum",
        "TANH": "AddTanh",
        "TRANSPOSE": "AddTranspose",
        "UNIDIRECTIONAL_SEQUENCE_LSTM": "AddUnidirectionalSequenceLSTM",
        "UNPACK": "AddUnpack",
    }
    unsupported = [name for name in op_names if name not in resolver_methods]
    if unsupported:
        raise ValueError(f"No firmware resolver mapping for TFLite ops: {unsupported}")
    registrations = "\n".join(
        f"  if (resolver.{resolver_methods[name]}() != kTfLiteOk) return kTfLiteError;" for name in op_names
    )
    (firmware_dir / "model_ops_resolver.h").write_text(
        "#pragma once\n"
        '#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"\n'
        f"constexpr int kModelOperatorCapacity = {len(op_names)};\n"
        "inline TfLiteStatus RegisterModelOps(\n"
        "    tflite::MicroMutableOpResolver<kModelOperatorCapacity>& resolver) {\n"
        f"{registrations}\n  return kTfLiteOk;\n}}\n",
        encoding="utf-8",
    )
    (firmware_dir / "tflite_operators.json").write_text(json.dumps(op_names, indent=2), encoding="utf-8")
    inputs = interpreter.get_input_details()
    details = {}
    for detail in inputs:
        match = next((name for name in ("voltage", "current", "duration") if name in detail["name"]), None)
        if match:
            details[match] = detail
    fixture = {}
    fixed_test_set = (
        args.fixed_test_set.resolve()
        if args.fixed_test_set is not None
        else prepared / "fixed_test_set.npz"
    )
    with np.load(fixed_test_set) as test:
        fixture_count = min(args.fixture_samples, len(test["target_soh"]))
        fixture_indices = np.linspace(
            0, len(test["target_soh"]) - 1, fixture_count, dtype=int
        )
        for name, detail in details.items():
            fixture[name] = quantize(test[name][fixture_indices], detail).reshape(-1).tolist()
        fixture["target_soh"] = test["target_soh"][fixture_indices].reshape(-1).astype(float).tolist()
        fixture["source_test_indices"] = fixture_indices.astype(int).tolist()
        fixture["sample_count"] = int(fixture_count)
    (firmware_dir / "test_fixture.json").write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
    arrays = []
    for name in ("voltage", "current", "duration"):
        values = fixture[name]
        chunks = [", ".join(str(item) for item in values[index : index + 20]) for index in range(0, len(values), 20)]
        symbol_name = {"voltage": "kVoltageInput", "current": "kCurrentInput", "duration": "kDurationInput"}[name]
        arrays.append(f"const int8_t {symbol_name}[] = {{\n  " + ",\n  ".join(chunks) + "\n};")
    target_chunks = [
        ", ".join(f"{value:.9g}f" for value in fixture["target_soh"][index : index + 16])
        for index in range(0, len(fixture["target_soh"]), 16)
    ]
    arrays.append("const float kTargetSoh[] = {\n  " + ",\n  ".join(target_chunks) + "\n};")
    sample_count = fixture["sample_count"]
    voltage_per_sample = len(fixture["voltage"]) // sample_count
    current_per_sample = len(fixture["current"]) // sample_count
    duration_per_sample = len(fixture["duration"]) // sample_count
    (firmware_dir / "test_fixture.h").write_text(
        "#pragma once\n#include <cstdint>\nextern const int8_t kVoltageInput[];\nextern const int8_t kCurrentInput[];\nextern const int8_t kDurationInput[];\nextern const float kTargetSoh[];\nextern const int kTestSampleCount;\nextern const int kVoltageElementsPerSample;\nextern const int kCurrentElementsPerSample;\nextern const int kDurationElementsPerSample;\n",
        encoding="utf-8",
    )
    (firmware_dir / "test_fixture.cc").write_text(
        '#include "test_fixture.h"\n' + "\n".join(arrays) +
        f"\nconst int kTestSampleCount = {sample_count};\n"
        f"const int kVoltageElementsPerSample = {voltage_per_sample};\n"
        f"const int kCurrentElementsPerSample = {current_per_sample};\n"
        f"const int kDurationElementsPerSample = {duration_per_sample};\n",
        encoding="utf-8",
    )
    template = Path(__file__).resolve().parent / "firmware_template"
    for name in ("main.cpp", "main_functions.cpp", "main_functions.h", "CMakeLists.txt"):
        shutil.copy2(template / name, firmware_dir / name)
    target_name = "paper_exp3_" + re.sub(r"[^a-z0-9]+", "_", args.model.lower()).strip("_")
    if args.firmware_dir_name != "firmware_assets":
        target_name += "_" + re.sub(r"[^a-z0-9]+", "_", args.firmware_dir_name.lower()).strip("_")
    cmake_path = firmware_dir / "CMakeLists.txt"
    cmake_path.write_text(
        cmake_path.read_text(encoding="utf-8").replace("paper_experiment_3", target_name),
        encoding="utf-8",
    )
    (firmware_dir / "README.md").write_text(
        "# Pico 2 firmware assets\n\n"
        "Copy `model_data.cc/.h` into a new Pico TFLM target. The JSON fixture contains the fixed, already-quantized "
        f"three-input test samples and FP32 SOH labels ({sample_count} evenly spaced held-out windows). Register only "
        "the operators reported by the TFLite analyzer, "
        "then record AllocateTensors status, arena usage, repeated latency, and serial output. Do not report MCU "
        "deployment success until the generated UF2 runs on the board.\n",
        encoding="utf-8",
    )
    print(firmware_dir)


if __name__ == "__main__":
    main()
