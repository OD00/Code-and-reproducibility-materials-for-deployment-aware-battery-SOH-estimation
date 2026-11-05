"""Stage 2b/3: full-INT8 quantize the manual TensorFlow graph and test locally."""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

from paper_experiments.common.models import ALL_MODEL_NAMES
from paper_experiments.experiment_3_mcu_deployment.manual_tensorflow import ManualTensorFlowModel
from src.common.experiment import regression_metrics


def interpreter_class():
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter
    return Interpreter


def quantize(value: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise ValueError(f"Invalid input quantization for {detail['name']}: {detail['quantization']}")
    info = np.iinfo(detail["dtype"])
    return np.clip(np.rint(value / scale + zero_point), info.min, info.max).astype(detail["dtype"])


def dequantize(value: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    return (value.astype(np.float32) - zero_point) * scale


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--model", choices=ALL_MODEL_NAMES, required=True)
    args = parser.parse_args()
    prepared = args.prepared_dir.resolve()
    model_dir = prepared / "models" / args.model
    status_path = model_dir / "int8_status.json"
    output_path = model_dir / "model_int8.tflite"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    try:
        state = torch.load(model_dir / "fp32_state_dict.pt", map_location="cpu", weights_only=True)
        model = ManualTensorFlowModel(args.model, state)
        concrete = model.__call__.get_concrete_function()
        with np.load(prepared / "representative_source_only.npz") as representative:
            calibration = {
                name: representative[name].astype(np.float32)
                for name in ("voltage", "current", "duration")
            }
        calibration_count = len(calibration["voltage"])

        def representative_dataset():
            for index in range(calibration_count):
                yield {
                    name: value[index : index + 1]
                    for name, value in calibration.items()
                }

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        output_path.write_bytes(tflite_model)

        Interpreter = interpreter_class()
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()
        if len(inputs) != 3 or len(outputs) != 1:
            raise ValueError(f"Unexpected INT8 signature: inputs={len(inputs)}, outputs={len(outputs)}")
        details_by_name = {}
        for detail in inputs:
            match = next((name for name in ("voltage", "current", "duration") if name in detail["name"]), None)
            if match is None:
                raise ValueError(f"Cannot identify TFLite input: {detail['name']}")
            if detail["dtype"] != np.int8:
                raise ValueError(f"Input {detail['name']} is not int8: {detail['dtype']}")
            details_by_name[match] = detail
        if outputs[0]["dtype"] != np.int8:
            raise ValueError(f"Output is not int8: {outputs[0]['dtype']}")

        fp32_predictions = list(csv.DictReader((model_dir / "fp32_test_predictions.csv").open(encoding="utf-8")))
        rows = []
        with np.load(prepared / "fixed_test_set.npz") as test:
            if len(fp32_predictions) != len(test["target_soh"]):
                raise ValueError("FP32 and INT8 test-set lengths differ")
            for index in range(len(test["target_soh"])):
                for name, detail in details_by_name.items():
                    interpreter.set_tensor(detail["index"], quantize(test[name][index : index + 1], detail))
                interpreter.invoke()
                prediction = float(dequantize(interpreter.get_tensor(outputs[0]["index"]), outputs[0]).reshape(-1)[0])
                target = float(test["target_soh"][index, 0])
                fp32_prediction = float(fp32_predictions[index]["fp32_prediction_soh"])
                rows.append(
                    {
                        "sample_index": index,
                        "target_soh": target,
                        "fp32_prediction_soh": fp32_prediction,
                        "int8_prediction_soh": prediction,
                        "int8_error": prediction - target,
                        "quantization_delta": prediction - fp32_prediction,
                    }
                )
        write_csv(model_dir / "int8_test_predictions.csv", rows)
        targets = np.asarray([row["target_soh"] for row in rows])
        fp32_array = np.asarray([row["fp32_prediction_soh"] for row in rows])
        int8_array = np.asarray([row["int8_prediction_soh"] for row in rows])
        delta = int8_array - fp32_array
        status = {
            "status": "success",
            "conversion_path": "PyTorch state_dict -> manually implemented TensorFlow graph -> TFLite full INT8; ONNX not used",
            "model_bytes": output_path.stat().st_size,
            "calibration_samples": calibration_count,
            "calibration_source_only": True,
            "test_samples": len(rows),
            "FP32_metrics": regression_metrics(targets, fp32_array),
            "INT8_metrics": regression_metrics(targets, int8_array),
            "INT8_vs_FP32_RMSE": float(np.sqrt(np.mean(np.square(delta)))),
            "INT8_vs_FP32_MAE": float(np.mean(np.abs(delta))),
            "INT8_vs_FP32_MaxAbs": float(np.max(np.abs(delta))),
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
    except Exception as error:
        status = {
            "status": "failed",
            "stage": "manual_tensorflow_to_full_int8_or_local_test",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
