"""Prepare identical CALCE test inputs and representative calibration data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from paper_experiments.common.models import ALL_MODEL_NAMES, MODEL_NAMES
from src.A1.model_input_a1_v2 import load_calce_cells, stack_cells

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data/model_input_a1_v2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(fold: Path) -> tuple[Path, dict]:
    configs = list(fold.glob("*_config.json"))
    checkpoints = list(fold.glob("*_best.pt"))
    if len(configs) != 1 or len(checkpoints) != 1:
        raise ValueError(f"Expected one config and checkpoint in {fold}")
    return checkpoints[0], json.loads(configs[0].read_text(encoding="utf-8"))


def scale(value, mean, std):
    return ((value - float(mean)) / float(std)).astype(np.float32)


def stratified_training_indices(record, samples_per_cell: int, seed: int) -> np.ndarray:
    """Select source-train windows across early/middle/late lifecycle stages."""
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    per_stage = [samples_per_cell // 3] * 3
    for index in range(samples_per_cell % 3):
        per_stage[index] += 1
    for stage, requested in enumerate(per_stage):
        candidates = np.flatnonzero(
            (record.source_split == 0) & (record.lifecycle_stage == stage)
        )
        if len(candidates) < requested:
            raise ValueError(
                f"{record.cell_id} lifecycle stage {stage} has only {len(candidates)} "
                f"training windows; {requested} requested"
            )
        selected.extend(rng.choice(candidates, size=requested, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment1-root", type=Path, default=here.parent / "experiment_1_cross_cell" / "results" / "a2_1c_formal")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=42, choices=(42, 123, 2026))
    parser.add_argument("--test-cell", default="CS2_38", choices=("CS2_35", "CS2_36", "CS2_37", "CS2_38"))
    parser.add_argument("--representative-samples-per-cell", type=int, default=128)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODEL_NAMES,
        default=list(MODEL_NAMES),
        help="Models to copy and prepare; defaults to the historical four-model experiment.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=0,
        help="0 uses every outer-test window; positive values select evenly spaced windows.",
    )
    parser.add_argument("--output-dir", type=Path, default=here / "prepared" / "seed_42_test_CS2_38")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cells = load_calce_cells(args.data_root.resolve())
    source_cells = sorted(set(("CS2_35", "CS2_36", "CS2_37", "CS2_38")) - {args.test_cell})
    test_record = cells[args.test_cell]
    test_raw = stack_cells(cells, [args.test_cell], None)
    if args.test_samples < 0:
        raise ValueError("--test-samples must be non-negative")
    test_count = len(test_raw[3]) if args.test_samples == 0 else min(args.test_samples, len(test_raw[3]))
    test_indices = np.linspace(0, len(test_raw[3]) - 1, test_count, dtype=int)

    representative_chunks = []
    representative_rows: list[dict] = []
    for cell_offset, cell_id in enumerate(source_cells):
        record = cells[cell_id]
        indices = stratified_training_indices(
            record, args.representative_samples_per_cell, args.seed + cell_offset
        )
        representative_chunks.append(
            (record.voltage[indices], record.current[indices], record.duration[indices])
        )
        for index in indices:
            item = record.metadata[int(index)]
            representative_rows.append(
                {
                    "cell_id": cell_id,
                    "source_sample_index": int(index),
                    "source_split": item["source_split"],
                    "lifecycle_stage": item["lifecycle_stage"],
                    "block_id": item["block_id"],
                    "target_cycle": item["target_cycle"],
                }
            )
    source_voltage = np.concatenate([item[0] for item in representative_chunks])
    source_current = np.concatenate([item[1] for item in representative_chunks])
    source_duration = np.concatenate([item[2] for item in representative_chunks])

    model_manifests = {}
    reference_scalers = None
    for model in args.models:
        fold = args.experiment1_root.resolve() / f"seed_{args.seed}" / model / f"CALCE_group_1C__test_{args.test_cell}"
        checkpoint, config = load_config(fold)
        scaler_config = config["scalers"]
        scalers = {
            "voltage_mean": scaler_config["x1"]["mean"],
            "voltage_std": scaler_config["x1"]["std"],
            "current_mean": scaler_config["x2"]["mean"],
            "current_std": scaler_config["x2"]["std"],
            "duration_mean": scaler_config["duration"]["mean"],
            "duration_std": scaler_config["duration"]["std"],
        }
        if reference_scalers is None:
            reference_scalers = scalers
        elif any(abs(float(scalers[key]) - float(reference_scalers[key])) > 1e-9 for key in scalers):
            raise ValueError("A2 models in the same fold do not share identical scalers")
        model_dir = output / "models" / model
        model_dir.mkdir(parents=True)
        shutil.copy2(checkpoint, model_dir / "fp32_state_dict.pt")
        shutil.copy2(fold / next(path.name for path in fold.glob("*_config.json")), model_dir / "source_config.json")
        model_manifests[model] = {
            "checkpoint_sha256": sha256(model_dir / "fp32_state_dict.pt"),
            "source_fold": str(fold),
            "parameter_count": config["parameter_count"],
        }

    s = reference_scalers
    rep = {
        "voltage": scale(source_voltage, s["voltage_mean"], s["voltage_std"]),
        "current": scale(source_current, s["current_mean"], s["current_std"]),
        "duration": scale(source_duration, s["duration_mean"], s["duration_std"]),
    }
    test = {
        "voltage": scale(test_raw[0][test_indices], s["voltage_mean"], s["voltage_std"]),
        "current": scale(test_raw[1][test_indices], s["current_mean"], s["current_std"]),
        "duration": scale(test_raw[2][test_indices], s["duration_mean"], s["duration_std"]),
        "target_soh": test_raw[3][test_indices].astype(np.float32),
    }
    np.savez_compressed(output / "representative_source_only.npz", **rep)
    np.savez_compressed(output / "fixed_test_set.npz", **test)
    write_csv(output / "representative_manifest.csv", representative_rows)
    write_csv(
        output / "test_manifest.csv",
        [
            {
                "cell_id": args.test_cell,
                "source_sample_index": int(index),
                "target_cycle": test_record.metadata[int(index)]["target_cycle"],
                "target_cycle_uid": test_record.metadata[int(index)]["target_cycle_uid"],
            }
            for index in test_indices
        ],
    )
    manifest = {
        "experiment": "Experiment 3: FP32 to INT8 edge deployment",
        "seed": args.seed,
        "test_cell": args.test_cell,
        "source_cells": source_cells,
        "representative_samples_per_cell": args.representative_samples_per_cell,
        "representative_samples": len(representative_rows),
        "test_samples": len(test_indices),
        "input_shapes": {"voltage": [1, 12, 256], "current": [1, 12, 256], "duration": [1, 12, 1]},
        "output_shape": [1, 1],
        "scalers": reference_scalers,
        "calibration_is_source_only": True,
        "calibration_source_split": "train only",
        "calibration_lifecycle_stratified": True,
        "test_absent_from_calibration": True,
        "representative_sha256": sha256(output / "representative_source_only.npz"),
        "fixed_test_sha256": sha256(output / "fixed_test_set.npz"),
        "models": model_manifests,
    }
    (output / "deployment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared shared deployment inputs at {output}")


if __name__ == "__main__":
    main()
