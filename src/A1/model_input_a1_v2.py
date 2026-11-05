"""A1 CALCE loader for gap-aware windows and block train/validation splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.A1.model_input_v1 import (
    A1Fold,
    CALCE_CONDITION_CELLS,
    apply_a1_scalers,
    build_stratified_fold_plan,
    fit_a1_scalers,
)


DEFAULT_MODEL_INPUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "model_input_a1_v2"
SPLIT_CODES = {"train": 0, "validation": 1, "purged": 2}


@dataclass(frozen=True)
class CellWindows:
    dataset: str
    cell_id: str
    condition_group: str
    source_file: Path
    voltage: np.ndarray
    current: np.ndarray
    duration: np.ndarray
    soh: np.ndarray
    capacity_ah: np.ndarray
    source_split: np.ndarray
    lifecycle_stage: np.ndarray
    block_id: np.ndarray
    row_indices: np.ndarray
    cycle_delta: np.ndarray
    metadata: tuple[dict, ...]

    @property
    def sample_count(self) -> int:
        return int(self.soh.shape[0])


def _text(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(()).item())


def load_cell_windows(path: Path) -> CellWindows:
    with np.load(path, allow_pickle=False) as data:
        if _text(data["schema_version"]) != "model_input_a1_v2":
            raise ValueError(f"Unexpected schema in {path.name}")
        rows = np.asarray(data["window_row_indices"], dtype=np.int64)
        observed = np.asarray(data["observed_cycle_index"], dtype=np.int64)
        observed_windows = observed[rows]
        gaps = np.diff(observed_windows, axis=1)
        maximum_gap = int(np.asarray(data["max_observed_cycle_gap"]).reshape(()))
        if not np.all((gaps > 0) & (gaps <= maximum_gap)):
            raise ValueError(f"Invalid observed-cycle gap in {path.name}")
        split = np.asarray(data["a1_source_split"], dtype=np.int8)
        stages = np.asarray(data["a1_lifecycle_stage"], dtype=np.int8)
        blocks = np.asarray(data["a1_block_id"], dtype=np.int8)
        train_rows = set(rows[split == 0].ravel().tolist())
        validation_rows = set(rows[split == 1].ravel().tolist())
        if train_rows.intersection(validation_rows):
            raise ValueError(f"Source train/validation cycle leakage in {path.name}")
        voltage = np.asarray(data["charge_voltage_v"], dtype=np.float32)[rows]
        current = np.asarray(data["charge_current_a"], dtype=np.float32)[rows]
        duration = np.asarray(data["charge_duration_s"], dtype=np.float32)[rows]
        soh = np.asarray(data["window_target_soh_nominal"], dtype=np.float32)
        capacity = np.asarray(data["window_target_capacity_ah"], dtype=np.float32)
        target_uids = np.asarray(data["window_target_cycle_uid"])
        target_rows = np.asarray(data["window_target_row_index"], dtype=np.int64)
        deltas = np.asarray(data["window_cycle_delta"], dtype=np.int16)
        dataset = _text(data["dataset"])
        cell_id = _text(data["cell_id"])
        group = _text(data["condition_group"])
        n = len(rows)
        if not all(len(value) == n for value in (voltage, current, duration, soh, capacity, split, stages, blocks, deltas)):
            raise ValueError(f"Window arrays have inconsistent lengths in {path.name}")
        metadata = tuple(
            {
                "dataset": dataset,
                "record": cell_id,
                "condition_group": group,
                "start_cycle": int(observed_windows[index, 0]),
                "target_cycle": int(observed_windows[index, -1]),
                "target_cycle_uid": str(target_uids[index]),
                "target_row_index": int(target_rows[index]),
                "capacity_ah": float(capacity[index, 0]),
                "soh_nominal": float(soh[index, 0]),
                "source_split": ("train", "validation", "purged")[int(split[index])],
                "lifecycle_stage": int(stages[index]),
                "block_id": int(blocks[index]),
                "cycle_span": int(observed_windows[index, -1] - observed_windows[index, 0]),
                "source_file": str(path),
                "source_sample_index": index,
            }
            for index in range(n)
        )
    return CellWindows(dataset, cell_id, group, path, voltage, current, duration, soh,
                       capacity, split, stages, blocks, rows, deltas, metadata)


def load_calce_cells(data_root: Path = DEFAULT_MODEL_INPUT_ROOT) -> dict[str, CellWindows]:
    expected = {cell for group in CALCE_CONDITION_CELLS.values() for cell in group}
    cells = {cell: load_cell_windows(Path(data_root) / "CALCE" / f"{cell}.npz") for cell in sorted(expected)}
    for group, names in CALCE_CONDITION_CELLS.items():
        for name in names:
            if cells[name].condition_group != group:
                raise ValueError(f"Condition mismatch for {name}")
    return cells


def stack_cells(
    cells: Mapping[str, CellWindows],
    selected: Iterable[str],
    source_split: str | Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[dict, ...]]:
    names = tuple(selected)
    if not names:
        raise ValueError("At least one cell must be selected")
    if source_split is None:
        allowed = None
    else:
        split_names = (source_split,) if isinstance(source_split, str) else tuple(source_split)
        allowed = np.asarray([SPLIT_CODES[name] for name in split_names], dtype=np.int8)
    chunks = []
    for name in names:
        record = cells[name]
        mask = np.ones(record.sample_count, dtype=bool) if allowed is None else np.isin(record.source_split, allowed)
        if not np.any(mask):
            raise ValueError(f"No {source_split} windows for {name}")
        chunks.append((record, mask))
    return (
        np.concatenate([record.voltage[mask] for record, mask in chunks]),
        np.concatenate([record.current[mask] for record, mask in chunks]),
        np.concatenate([record.duration[mask] for record, mask in chunks]),
        np.concatenate([record.soh[mask] for record, mask in chunks]),
        tuple(item for record, mask in chunks for item, keep in zip(record.metadata, mask) if keep),
    )


def validate_outer_plan(cells: Mapping[str, CellWindows], folds: Mapping[str, A1Fold]) -> list[dict]:
    rows = []
    for cell_id, record in sorted(cells.items()):
        train = int(np.sum(record.source_split == 0))
        validation = int(np.sum(record.source_split == 1))
        for stage in range(3):
            stage_roles = record.source_split[record.lifecycle_stage == stage]
            if not np.any(stage_roles == 0) or not np.any(stage_roles == 1):
                raise ValueError(f"{cell_id} stage {stage} lacks train/validation")
        rows.append({"cell": cell_id, "windows": record.sample_count, "train": train,
                     "validation": validation, "purged": int(np.sum(record.source_split == 2)),
                     "ratio": train / validation})
    for fold in folds.values():
        if fold.test_cell in fold.source_cells or len(fold.source_cells) != 3:
            raise ValueError(f"Invalid outer isolation in {fold.name}")
    return rows


__all__ = [
    "A1Fold", "CALCE_CONDITION_CELLS", "CellWindows", "DEFAULT_MODEL_INPUT_ROOT",
    "apply_a1_scalers", "build_stratified_fold_plan", "fit_a1_scalers",
    "load_calce_cells", "stack_cells", "validate_outer_plan",
]
