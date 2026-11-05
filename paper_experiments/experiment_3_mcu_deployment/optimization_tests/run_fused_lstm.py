"""Validate, full-INT8 convert, and test a fused CNN-LSTM mapping."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

from paper_experiments.common.models import create_model, unwrap_prediction
from paper_experiments.experiment_3_mcu_deployment.convert_tensorflow_int8 import (
    dequantize,
    interpreter_class,
    quantize,
)
from paper_experiments.experiment_3_mcu_deployment.optimization_tests.fused_lstm_tensorflow import (
    FusedLSTMModel,
)
from src.common.experiment import regression_metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("CnnLSTMControl", "CnnLSTMSimpleCNN"),
        default="CnnLSTMControl",
    )
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--max-abs-tolerance", type=float, default=2e-4)
    args = parser.parse_args()

    prepared = args.prepared_dir.resolve()
    source_model_dir = prepared / "models" / args.model
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    state = torch.load(
        source_model_dir / "fp32_state_dict.pt",
        map_location="cpu",
        weights_only=True,
    )
    pytorch_model = create_model(args.model)
    pytorch_model.load_state_dict(state, strict=True)
    pytorch_model.eval()
    mapped = FusedLSTMModel(state, model_name=args.model)
    concrete = mapped.__call__.get_concrete_function()

    comparisons = []
    maximum = 0.0
    with np.load(prepared / "fixed_test_set.npz") as test, torch.no_grad():
        indices = np.linspace(
            0,
            len(test["target_soh"]) - 1,
            min(args.validation_samples, len(test["target_soh"])),
            dtype=int,
        )
        for index in indices:
            arrays = [
                test[name][index : index + 1].astype(np.float32)
                for name in ("voltage", "current", "duration")
            ]
            reference = float(
                unwrap_prediction(
                    pytorch_model(*(torch.from_numpy(value) for value in arrays))
                )
                .detach()
                .reshape(-1)[0]
            )
            prediction = float(
                mapped(*(tf.convert_to_tensor(value) for value in arrays))["soh"]
                .numpy()
                .reshape(-1)[0]
            )
            difference = prediction - reference
            maximum = max(maximum, abs(difference))
            comparisons.append(
                {
                    "sample_index": int(index),
                    "pytorch": reference,
                    "tensorflow_fused": prediction,
                    "difference": difference,
                }
            )
    if maximum > args.max_abs_tolerance:
        raise ValueError(
            f"Fused LSTM FP32 mapping error {maximum:.9g} exceeds "
            f"{args.max_abs_tolerance:.9g}"
        )

    saved_model_dir = output / "tensorflow_saved_model"
    with tempfile.TemporaryDirectory(prefix="battery_fused_lstm_") as temp_name:
        staged = Path(temp_name) / "saved_model"
        tf.saved_model.save(
            mapped,
            staged,
            signatures={"serving_default": concrete},
        )
        converter = tf.lite.TFLiteConverter.from_saved_model(str(staged))
        with np.load(prepared / "representative_source_only.npz") as representative:
            calibration = {
                name: representative[name].astype(np.float32)
                for name in ("voltage", "current", "duration")
            }

        def representative_dataset():
            for index in range(len(calibration["voltage"])):
                yield {
                    name: value[index : index + 1]
                    for name, value in calibration.items()
                }

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        shutil.copytree(staged, saved_model_dir)

    model_path = output / "model_int8.tflite"
    model_path.write_bytes(tflite_model)
    Interpreter = interpreter_class()
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    op_counts = Counter(item["op_name"] for item in interpreter._get_ops_details())
    if op_counts["UNIDIRECTIONAL_SEQUENCE_LSTM"] != 2:
        raise ValueError(
            "Conversion did not preserve two fused LSTM operators: "
            f"{dict(op_counts)}"
        )

    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    details_by_name = {}
    for detail in inputs:
        name = next(
            item
            for item in ("voltage", "current", "duration")
            if item in detail["name"]
        )
        details_by_name[name] = detail

    rows = []
    with np.load(prepared / "fixed_test_set.npz") as test:
        for index in range(len(test["target_soh"])):
            for name, detail in details_by_name.items():
                interpreter.set_tensor(
                    detail["index"],
                    quantize(test[name][index : index + 1], detail),
                )
            interpreter.invoke()
            prediction = float(
                dequantize(
                    interpreter.get_tensor(outputs[0]["index"]),
                    outputs[0],
                ).reshape(-1)[0]
            )
            rows.append(
                {
                    "sample_index": index,
                    "target_soh": float(test["target_soh"][index, 0]),
                    "int8_prediction_soh": prediction,
                }
            )
    write_csv(output / "int8_test_predictions.csv", rows)
    targets = np.asarray([row["target_soh"] for row in rows])
    predictions = np.asarray([row["int8_prediction_soh"] for row in rows])
    status = {
        "status": "success",
        "model": args.model,
        "source_fold": str(prepared),
        "mapping_max_abs": maximum,
        "mapping_tolerance": args.max_abs_tolerance,
        "calibration_samples": len(calibration["voltage"]),
        "calibration_source_only": True,
        "test_samples": len(rows),
        "model_bytes": len(tflite_model),
        "node_count": sum(op_counts.values()),
        "tensor_count": len(interpreter.get_tensor_details()),
        "operator_counts": dict(op_counts),
        "int8_metrics": regression_metrics(targets, predictions),
        "mapping_comparisons": comparisons,
        "inputs": [
            {
                "name": item["name"],
                "shape": item["shape"].tolist(),
                "dtype": str(item["dtype"]),
                "quantization": list(item["quantization"]),
            }
            for item in inputs
        ],
        "output": {
            "name": outputs[0]["name"],
            "shape": outputs[0]["shape"].tolist(),
            "dtype": str(outputs[0]["dtype"]),
            "quantization": list(outputs[0]["quantization"]),
        },
    }
    (output / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
