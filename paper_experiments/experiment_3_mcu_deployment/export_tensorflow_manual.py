"""Stage 2a: manually map one PyTorch checkpoint to an equivalent TensorFlow graph."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

from paper_experiments.common.models import ALL_MODEL_NAMES, create_model, unwrap_prediction
from paper_experiments.experiment_3_mcu_deployment.manual_tensorflow import ManualTensorFlowModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--model", choices=ALL_MODEL_NAMES, required=True)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--max-abs-tolerance", type=float, default=2e-4)
    args = parser.parse_args()
    prepared = args.prepared_dir.resolve()
    model_dir = prepared / "models" / args.model
    status_path = model_dir / "tensorflow_mapping_status.json"
    try:
        state = torch.load(model_dir / "fp32_state_dict.pt", map_location="cpu", weights_only=True)
        pytorch_model = create_model(args.model)
        pytorch_model.load_state_dict(state, strict=True)
        pytorch_model.eval()
        if args.model in ("CnnMambaImproved", "CnnMambaSimpleCNN"):
            if not torch.cuda.is_available():
                raise RuntimeError("CnnMambaImproved reference validation requires CUDA")
            pytorch_model.cuda()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        tensorflow_model = ManualTensorFlowModel(args.model, state)
        tensorflow_model.__call__.get_concrete_function()
        rows = []
        maximum = 0.0
        with np.load(prepared / "fixed_test_set.npz") as test, torch.no_grad():
            count = min(args.validation_samples, len(test["target_soh"]))
            indices = np.linspace(0, len(test["target_soh"]) - 1, count, dtype=int)
            for index in indices:
                arrays = [test[name][index : index + 1].astype(np.float32) for name in ("voltage", "current", "duration")]
                reference = float(
                    unwrap_prediction(
                        pytorch_model(*(torch.from_numpy(value).to(device) for value in arrays))
                    ).detach().cpu().reshape(-1)[0]
                )
                mapped = float(tensorflow_model(*(tf.convert_to_tensor(value) for value in arrays))["soh"].numpy().reshape(-1)[0])
                error = mapped - reference
                maximum = max(maximum, abs(error))
                rows.append({"sample_index": int(index), "pytorch": reference, "tensorflow": mapped, "difference": error})
        if maximum > args.max_abs_tolerance:
            raise ValueError(f"TensorFlow mapping max abs error {maximum:.9g} exceeds {args.max_abs_tolerance:.9g}")
        saved_model_dir = model_dir / "tensorflow_saved_model"
        if saved_model_dir.exists():
            raise FileExistsError(saved_model_dir)
        # TensorFlow's Windows file layer can fail on non-ASCII SavedModel
        # paths. Save to an ASCII temporary directory, then copy the finished
        # artifact back without changing graph construction or weights.
        with tempfile.TemporaryDirectory(prefix="battery_tf_manual_") as temp_name:
            staged_saved_model = Path(temp_name) / "saved_model"
            tf.saved_model.save(
                tensorflow_model,
                staged_saved_model,
                signatures={"serving_default": tensorflow_model.__call__.get_concrete_function()},
            )
            shutil.copytree(staged_saved_model, saved_model_dir)
        status = {
            "status": "success",
            "mapping": "manual direct PyTorch state_dict to TensorFlow variables; ONNX not used",
            "validated_samples": len(rows),
            "max_abs_difference": maximum,
            "tolerance": args.max_abs_tolerance,
            "reference_device": str(device),
            "saved_model": str(saved_model_dir),
            "comparisons": rows,
        }
    except Exception as error:
        status = {
            "status": "failed",
            "stage": "manual_pytorch_to_tensorflow",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(saved_model_dir)


if __name__ == "__main__":
    main()
