from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "data/model_input_v1"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/model_input_a1_v2"
DEFAULT_RAW = Path(__file__).resolve().parents[1] / "data/raw/CALCE"
WINDOW_SIZE = 12
MAX_OBSERVED_GAP = 5
PURGE_WINDOWS = WINDOW_SIZE - 1
ROLE_TRAIN = np.int8(0)
ROLE_VALIDATION = np.int8(1)
ROLE_PURGED = np.int8(2)
ROLE_NAMES = {0: "train", 1: "validation", 2: "purged"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def gap_tolerant_windows(observed: np.ndarray) -> np.ndarray:
    candidates = np.lib.stride_tricks.sliding_window_view(
        np.arange(len(observed), dtype=np.int32), WINDOW_SIZE
    )
    gaps = np.diff(observed[candidates], axis=1)
    keep = np.all((gaps > 0) & (gaps <= MAX_OBSERVED_GAP), axis=1)
    return np.asarray(candidates[keep], dtype=np.int32)


def assign_blocks(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each lifecycle third into three contiguous blocks, 2 train : 1 val."""
    n = len(rows)
    roles = np.full(n, ROLE_PURGED, dtype=np.int8)
    stages = np.full(n, -1, dtype=np.int8)
    blocks = np.full(n, -1, dtype=np.int8)
    # The middle block of every lifecycle stage is validation. This gives the
    # validation set early/middle/late-life coverage while the two training
    # blocks retain both edges of each stage (including the late-life tail).
    validation_position = (1, 1, 1)
    block_id = 0
    for stage, stage_ids in enumerate(np.array_split(np.arange(n), 3)):
        for position, block_ids in enumerate(np.array_split(stage_ids, 3)):
            if len(block_ids) == 0:
                raise RuntimeError("A lifecycle block is empty")
            stages[block_ids] = stage
            blocks[block_ids] = block_id
            roles[block_ids] = (
                ROLE_VALIDATION if position == validation_position[stage] else ROLE_TRAIN
            )
            block_id += 1

    # Twelve-cycle windows overlap. At every role boundary, purge the first
    # eleven windows on the later side so no cycle row can appear in both sets.
    boundaries = np.flatnonzero(roles[1:] != roles[:-1]) + 1
    for boundary in boundaries:
        roles[boundary : min(boundary + PURGE_WINDOWS, n)] = ROLE_PURGED

    train_rows = set(rows[roles == ROLE_TRAIN].ravel().tolist())
    validation_rows = set(rows[roles == ROLE_VALIDATION].ravel().tolist())
    if train_rows.intersection(validation_rows):
        raise RuntimeError("Train and validation windows share cycle rows")
    for stage in range(3):
        stage_roles = roles[stages == stage]
        if not np.any(stage_roles == ROLE_TRAIN) or not np.any(
            stage_roles == ROLE_VALIDATION
        ):
            raise RuntimeError(f"Lifecycle stage {stage} lacks train or validation")
    return roles, stages, blocks


def source_format(raw_root: Path, cell_id: str) -> str:
    suffixes = sorted(
        {path.suffix.lower().lstrip(".") for path in (raw_root / cell_id).rglob("*") if path.is_file()}
    )
    return "+".join(suffixes) or "unknown"


def csv_write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Output exists; refusing to overwrite: {args.output}")
    staging = args.output.with_name(args.output.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "CALCE").mkdir(parents=True)

    old_manifest_path = args.source / "dataset_manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    old_by_cell = {
        item["cell_id"]: item for item in old_manifest["datasets"]["CALCE"]["cells"]
    }
    coverage_rows: list[dict] = []
    split_rows: list[dict] = []
    distribution_rows: list[dict] = []
    cells_manifest: list[dict] = []

    for source_path in sorted((args.source / "CALCE").glob("*.npz")):
        with np.load(source_path, allow_pickle=False) as data:
            cell_id = str(data["cell_id"])
            observed = np.asarray(data["observed_cycle_index"], dtype=np.int32)
            capacity = np.asarray(data["capacity_ah"], dtype=np.float32)
            soh = np.asarray(data["label_soh_nominal"], dtype=np.float32)
            old_rows = np.asarray(data["window_row_indices"], dtype=np.int32)
            rows = gap_tolerant_windows(observed)
            roles, stages, blocks = assign_blocks(rows)
            targets = rows[:, -1]
            observed_windows = observed[rows]
            deltas = np.zeros(rows.shape, dtype=np.int16)
            deltas[:, 1:] = np.diff(observed_windows, axis=1).astype(np.int16)
            spans = (observed_windows[:, -1] - observed_windows[:, 0]).astype(
                np.int16
            )[:, None]
            arrays = {
                key: np.asarray(data[key])
                for key in data.files
                if not key.startswith("window_") and key != "schema_version"
            }
            arrays.update(
                {
                    "schema_version": np.asarray("model_input_a1_v2"),
                    "window_size": np.asarray(WINDOW_SIZE, dtype=np.int32),
                    "max_observed_cycle_gap": np.asarray(MAX_OBSERVED_GAP, dtype=np.int32),
                    "window_row_indices": rows,
                    "window_target_row_index": targets,
                    "window_target_capacity_ah": capacity[targets],
                    "window_target_soh_nominal": soh[targets],
                    "window_target_cycle_uid": np.asarray(data["cycle_uid"])[targets],
                    "window_cycle_delta": deltas,
                    "window_cycle_span": spans,
                    "a1_source_split": roles,
                    "a1_lifecycle_stage": stages,
                    "a1_block_id": blocks,
                    "source_file_format": np.asarray(source_format(args.raw_root, cell_id)),
                }
            )
            output_path = staging / "CALCE" / source_path.name
            atomic_npz(output_path, arrays)

            target_soh = soh[targets, 0]
            old_target_soh = soh[old_rows[:, -1], 0]
            coverage_rows.append(
                {
                    "cell_id": cell_id,
                    "valid_cycles": len(observed),
                    "v1_windows": len(old_rows),
                    "v2_windows": len(rows),
                    "raw_soh_min": float(soh.min()),
                    "v1_target_soh_min": float(old_target_soh.min()),
                    "v2_target_soh_min": float(target_soh.min()),
                    "raw_targets_soh_lt_0.9": int(np.sum(soh[:, 0] < 0.9)),
                    "v1_targets_soh_lt_0.9": int(np.sum(old_target_soh < 0.9)),
                    "v2_targets_soh_lt_0.9": int(np.sum(target_soh < 0.9)),
                    "cycle_span_p95": float(np.percentile(spans, 95)),
                    "cycle_span_max": int(spans.max()),
                }
            )
            for stage in range(3):
                for role in (ROLE_TRAIN, ROLE_VALIDATION, ROLE_PURGED):
                    mask = (stages == stage) & (roles == role)
                    values = target_soh[mask]
                    split_rows.append(
                        {
                            "cell_id": cell_id,
                            "lifecycle_stage": stage,
                            "split": ROLE_NAMES[int(role)],
                            "window_count": len(values),
                            "soh_min": float(values.min()) if len(values) else "",
                            "soh_max": float(values.max()) if len(values) else "",
                            "soh_mean": float(values.mean()) if len(values) else "",
                        }
                    )
            distribution_rows.append(
                {
                    "cell_id": cell_id,
                    "condition_group": str(data["condition_group"]),
                    "source_format": source_format(args.raw_root, cell_id),
                    "charge_duration_mean_s": float(data["charge_duration_s"].mean()),
                    "charge_duration_std_s": float(data["charge_duration_s"].std()),
                    "charge_voltage_mean_v": float(data["charge_voltage_v"].mean()),
                    "charge_current_mean_a": float(data["charge_current_a"].mean()),
                    "charge_segment_count_max": int(data["charge_segment_count"].max()),
                }
            )
            digest = sha256_file(output_path)
            train_n = int(np.sum(roles == ROLE_TRAIN))
            val_n = int(np.sum(roles == ROLE_VALIDATION))
            cells_manifest.append(
                {
                    "cell_id": cell_id,
                    "condition_group": str(data["condition_group"]),
                    "file": source_path.name,
                    "source_file_format": source_format(args.raw_root, cell_id),
                    "valid_cycles": len(observed),
                    "windows": len(rows),
                    "train_windows": train_n,
                    "validation_windows": val_n,
                    "purged_windows": int(np.sum(roles == ROLE_PURGED)),
                    "train_validation_ratio": train_n / val_n,
                    "sha256": digest,
                    "v1_windows": int(old_by_cell[cell_id]["valid_12_cycle_windows"]),
                }
            )

    csv_write(staging / "window_coverage_audit.csv", coverage_rows)
    csv_write(staging / "split_coverage_audit.csv", split_rows)
    csv_write(staging / "input_distribution_audit.csv", distribution_rows)
    manifest = {
        "schema_version": "model_input_a1_v2",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_root": str(args.source),
        "source_manifest_sha256": sha256_file(old_manifest_path),
        "scope": "CALCE A1 only; model_input_v1 and TJU are unchanged",
        "window": {
            "size": WINDOW_SIZE,
            "max_adjacent_observed_cycle_gap": MAX_OBSERVED_GAP,
            "gap_semantics": "observed-cycle order after QC; not an assertion of physical-cycle continuity",
            "cycle_delta_saved": True,
        },
        "source_split": {
            "lifecycle_stages": 3,
            "blocks_per_stage": 3,
            "validation_block_positions_by_stage": [1, 1, 1],
            "nominal_ratio": "2 train : 1 validation",
            "purge_windows_at_role_boundary": PURGE_WINDOWS,
            "role_codes": {"0": "train", "1": "validation", "2": "purged"},
            "outer_test_rule": "held-out cell uses all windows only for test",
        },
        "cells": cells_manifest,
    }
    (staging / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = """# model_input_a1_v2

本目录是 A1 的 CALCE 专用数据版本；原 `model_input_v1` 未覆盖，TJU 未改动。

- 12 个循环组成一个样本；相邻保留循环的 `observed_cycle_index` 间隔最多为 5。
- `window_cycle_delta` 保存每个窗口内的实际观察序号间隔，避免把缺口伪装成完全连续。
- 每个电芯按时间顺序分成早/中/晚三个寿命阶段，每阶段再分三个连续区块；其中两个训练、一个验证。
- 每个阶段的中间块用于验证，前后两个块用于训练，使两侧都覆盖完整寿命过程。
- 训练/验证角色变化边界清除 11 个窗口，保证二者不共享任何原始循环。
- 外层 LOCO 的目标电芯完全不参与训练或验证，其全部窗口仅用于最终测试。

`a1_source_split`: 0=train, 1=validation, 2=purged。区块划分只用于源电芯；当该电芯是外层测试电芯时忽略此字段。
"""
    (staging / "README.md").write_text(readme, encoding="utf-8")
    checksum_lines = []
    for path in sorted((staging / "CALCE").glob("*.npz")):
        checksum_lines.append(f"{sha256_file(path)}  CALCE/{path.name}")
    (staging / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    os.replace(staging, args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
