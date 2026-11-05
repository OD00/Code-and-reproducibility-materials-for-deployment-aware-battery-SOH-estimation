import argparse
import copy
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..models import (
    CnnLSTMControl,
    CnnLstmAttention,
    CnnMambaImproved,
    CnnMambaSimple,
    CnnSE,
    CnnTransformer,
    ResNet1d,
)
from .paths import DATA_DIR, OUTPUT_DIR


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_EPOCHS = 50
DEFAULT_LR = 0.0005211317875270432
DEFAULT_BATCH_SIZE = 16
DEFAULT_WINDOW = 12
DEFAULT_POINTS = 120
DEFAULT_SEED = 42
DEFAULT_ADAM_BETAS = (0.8, 0.99)
CALCE_SC_FILES = ("sc_35.pt", "sc_36.pt", "sc_37.pt", "sc_38.pt")


@dataclass
class BatteryRecord:
    dataset: str
    name: str
    voltage: List[np.ndarray]
    current: List[np.ndarray]
    capacity: np.ndarray

    @property
    def retention(self) -> np.ndarray:
        return self.capacity / self.capacity[0]


class StandardScaler1D:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std) if float(std) > 1e-12 else 1.0

    @classmethod
    def fit(cls, values: np.ndarray):
        values = np.asarray(values, dtype=np.float32)
        return cls(float(values.mean()), float(values.std()))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean

    def to_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std}


class MinMaxScaler1D:
    def __init__(self, vmin: float, vmax: float):
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.scale = self.vmax - self.vmin
        if abs(self.scale) < 1e-12:
            self.scale = 1.0

    @classmethod
    def fit(cls, values: np.ndarray):
        values = np.asarray(values, dtype=np.float32)
        return cls(float(values.min()), float(values.max()))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.vmin) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.scale + self.vmin

    def to_dict(self) -> Dict[str, float]:
        return {"min": self.vmin, "max": self.vmax, "scale": self.scale}


class IdentityScaler1D:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)

    def to_dict(self) -> Dict[str, str]:
        return {"type": "identity"}


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def resample_curve(curve: np.ndarray, points: int = DEFAULT_POINTS) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    if len(curve) == points:
        return curve
    if len(curve) == 0:
        return np.zeros(points, dtype=np.float32)
    if len(curve) == 1:
        return np.full(points, curve[0], dtype=np.float32)
    old_x = np.linspace(0.0, 1.0, len(curve), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, points, dtype=np.float32)
    return np.interp(new_x, old_x, curve).astype(np.float32)


def to_curve_list(curves) -> List[np.ndarray]:
    arr = np.asarray(curves, dtype=object)
    if arr.dtype != object and arr.ndim == 2:
        return [np.asarray(row, dtype=np.float32) for row in arr]
    return [np.asarray(row, dtype=np.float32) for row in list(curves)]


def load_calce_records(retention_min: Optional[float] = None) -> List[BatteryRecord]:
    data = np.load(DATA_DIR / "CALCE" / "calce_all_ica_dva_data.npy", allow_pickle=True).item()
    records = []
    for name, cell in data.items():
        capacity = np.asarray(cell["capacity"], dtype=np.float32)
        voltage = to_curve_list(cell["voltage"])
        current = to_curve_list(cell["current"])
        if retention_min is not None:
            mask = capacity / capacity[0] >= retention_min
            capacity = capacity[mask]
            voltage = [v for v, keep in zip(voltage, mask) if keep]
            current = [c for c, keep in zip(current, mask) if keep]
        records.append(BatteryRecord("CALCE", name, voltage, current, capacity))
    return records


def load_calce_windowed_data(
    retention_min: Optional[float] = None,
    files: Sequence[str] = CALCE_SC_FILES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """Load the preprocessed CALCE TensorDataset files.

    The sc_*.pt files are already 12-cycle windows with min-max scaled voltage,
    current, and capacity labels. The returned y is therefore normalized
    capacity, not physical Ah.
    """
    x1_all, x2_all, y_all, meta = [], [], [], []
    for file_name in files:
        path = DATA_DIR / "CALCE" / file_name
        dataset = torch.load(path, map_location="cpu", weights_only=False)
        record_name = path.stem
        sample_index = 0
        for i in range(len(dataset)):
            voltage, current, label = dataset[i]
            y_value = float(torch.as_tensor(label).reshape(-1)[-1])
            if retention_min is not None and y_value < retention_min:
                continue
            x1_all.append(torch.as_tensor(voltage).detach().cpu().numpy())
            x2_all.append(torch.as_tensor(current).detach().cpu().numpy())
            y_all.append(y_value)
            meta.append(
                {
                    "dataset": "CALCE",
                    "record": record_name,
                    "start_cycle": sample_index + 1,
                    "target_cycle": sample_index + DEFAULT_WINDOW,
                    "raw_capacity": y_value,
                    "retention": y_value,
                    "source_file": str(path),
                    "source_sample_index": i,
                }
            )
            sample_index += 1
    return (
        np.asarray(x1_all, dtype=np.float32),
        np.asarray(x2_all, dtype=np.float32),
        np.asarray(y_all, dtype=np.float32).reshape(-1, 1),
        meta,
    )


def load_pt_records(dataset_dir: str, dataset_name: str, exclude_names: Optional[Iterable[str]] = None) -> List[BatteryRecord]:
    exclude = set(exclude_names or [])
    records = []
    for path in sorted((DATA_DIR / dataset_dir).glob("*.pt")):
        if path.stem in exclude:
            continue
        voltage, current, capacity = torch.load(path, map_location="cpu", weights_only=False)
        records.append(
            BatteryRecord(
                dataset_name,
                path.stem,
                to_curve_list(np.asarray(voltage, dtype=np.float32)),
                to_curve_list(np.asarray(current, dtype=np.float32)),
                np.asarray(capacity, dtype=np.float32),
            )
        )
    return records


def build_windows(
    records: Sequence[BatteryRecord],
    window: int = DEFAULT_WINDOW,
    points: int = DEFAULT_POINTS,
    label_mode: str = "raw",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    x1_all, x2_all, y_all, meta = [], [], [], []
    for record in records:
        labels = record.capacity if label_mode == "raw" else record.retention
        if len(labels) < window:
            continue
        voltage = [resample_curve(v, points) for v in record.voltage]
        current = [resample_curve(c, points) for c in record.current]
        for end in range(window - 1, len(labels)):
            start = end - window + 1
            x1_all.append(np.stack(voltage[start : end + 1], axis=0))
            x2_all.append(np.stack(current[start : end + 1], axis=0))
            y_all.append(float(labels[end]))
            meta.append(
                {
                    "dataset": record.dataset,
                    "record": record.name,
                    "start_cycle": start + 1,
                    "target_cycle": end + 1,
                    "raw_capacity": float(record.capacity[end]),
                    "retention": float(record.retention[end]),
                }
            )
    return (
        np.asarray(x1_all, dtype=np.float32),
        np.asarray(x2_all, dtype=np.float32),
        np.asarray(y_all, dtype=np.float32).reshape(-1, 1),
        meta,
    )


def chronological_split_by_meta(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    meta: Sequence[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]]:
    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, item in enumerate(meta):
        groups.setdefault((item["dataset"], item["record"]), []).append(i)

    split_indices = {"train": [], "val": [], "test": []}
    for indices in groups.values():
        indices = sorted(indices, key=lambda idx: meta[idx]["target_cycle"])
        n = len(indices)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        split_indices["train"].extend(indices[:n_train])
        split_indices["val"].extend(indices[n_train : n_train + n_val])
        split_indices["test"].extend(indices[n_train + n_val :])

    out = {}
    for split, indices in split_indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        out[split] = (x1[idx], x2[idx], y[idx], [meta[i] for i in idx])
    return out


def subset_by_indices(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    meta: Sequence[Dict],
    indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    idx = np.asarray(indices, dtype=np.int64)
    return x1[idx], x2[idx], y[idx], [meta[i] for i in idx]


def leave_one_record_out_splits(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    meta: Sequence[Dict],
    val_ratio: float = 0.2,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]]]:
    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, item in enumerate(meta):
        groups.setdefault((item["dataset"], item["record"]), []).append(i)

    folds = {}
    for holdout in sorted(groups):
        train_val_indices = []
        for key, indices in groups.items():
            if key != holdout:
                train_val_indices.extend(indices)

        train_indices, val_indices = [], []
        for key, indices in groups.items():
            if key == holdout:
                continue
            ordered = sorted(indices, key=lambda idx: meta[idx]["target_cycle"])
            n_val = max(1, int(len(ordered) * val_ratio))
            train_indices.extend(ordered[:-n_val])
            val_indices.extend(ordered[-n_val:])

        fold_name = f"test_{holdout[1]}"
        folds[fold_name] = {
            "train": subset_by_indices(x1, x2, y, meta, train_indices),
            "val": subset_by_indices(x1, x2, y, meta, val_indices),
            "test": subset_by_indices(x1, x2, y, meta, sorted(groups[holdout], key=lambda idx: meta[idx]["target_cycle"])),
        }
    return folds


def leave_one_record_out_full_train_splits(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    meta: Sequence[Dict],
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]]]:
    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, item in enumerate(meta):
        groups.setdefault((item["dataset"], item["record"]), []).append(i)

    folds = {}
    for holdout in sorted(groups):
        train_indices = []
        for key, indices in groups.items():
            if key != holdout:
                train_indices.extend(sorted(indices, key=lambda idx: meta[idx]["target_cycle"]))

        test_indices = sorted(groups[holdout], key=lambda idx: meta[idx]["target_cycle"])
        fold_name = f"test_{holdout[1]}"
        train_split = subset_by_indices(x1, x2, y, meta, train_indices)
        folds[fold_name] = {
            "train": train_split,
            "val": train_split,
            "test": subset_by_indices(x1, x2, y, meta, test_indices),
        }
    return folds


def _group_indices_by_record(meta: Sequence[Dict]) -> Dict[Tuple[str, str], List[int]]:
    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, item in enumerate(meta):
        groups.setdefault((item["dataset"], item["record"]), []).append(i)
    return {
        key: sorted(indices, key=lambda idx: meta[idx]["target_cycle"])
        for key, indices in groups.items()
    }


def _assert_split_cycle_isolation(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]],
) -> None:
    """Raise if two splits for one cell use any of the same raw cycles."""
    cycles: Dict[Tuple[str, str, str], set[int]] = {}
    for split, (_, _, _, split_meta) in splits.items():
        for item in split_meta:
            key = (item["dataset"], item["record"], split)
            cycles.setdefault(key, set()).update(
                range(int(item["start_cycle"]), int(item["target_cycle"]) + 1)
            )

    records = {(dataset, record) for dataset, record, _ in cycles}
    for dataset, record in records:
        present = [
            split
            for split in ("train", "val", "test")
            if (dataset, record, split) in cycles
        ]
        for left_pos, left in enumerate(present):
            for right in present[left_pos + 1 :]:
                overlap = cycles[(dataset, record, left)] & cycles[(dataset, record, right)]
                if overlap:
                    preview = sorted(overlap)[:10]
                    raise ValueError(
                        f"Raw-cycle overlap for {dataset}/{record}: "
                        f"{left} vs {right}, examples={preview}"
                    )


def purged_chronological_split_by_meta(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    meta: Sequence[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    purge_gap: int = DEFAULT_WINDOW - 1,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]]:
    """Split every cell chronologically and purge adjacent sliding windows.

    ``purge_gap`` counts discarded window positions at each split boundary.
    With stride-one windows of length 12, a gap of 11 makes the raw cycles used
    by train, validation and test mutually exclusive.
    """
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected train_ratio > 0, val_ratio > 0 and sum < 1.")
    if purge_gap < 0:
        raise ValueError("purge_gap must be non-negative.")

    split_indices = {"train": [], "val": [], "test": []}
    for key, ordered in _group_indices_by_record(meta).items():
        usable = len(ordered) - 2 * purge_gap
        if usable < 3:
            raise ValueError(
                f"{key} has {len(ordered)} windows, insufficient for three "
                f"non-empty splits with purge_gap={purge_gap}."
            )

        n_train = max(1, int(usable * train_ratio))
        n_val = max(1, int(usable * val_ratio))
        if n_train + n_val >= usable:
            n_val = max(1, usable - n_train - 1)
        n_test = usable - n_train - n_val
        if n_test < 1:
            raise ValueError(f"{key} produced an empty test split.")

        train_end = n_train
        val_start = train_end + purge_gap
        val_end = val_start + n_val
        test_start = val_end + purge_gap
        split_indices["train"].extend(ordered[:train_end])
        split_indices["val"].extend(ordered[val_start:val_end])
        split_indices["test"].extend(ordered[test_start : test_start + n_test])

    splits = {
        split: subset_by_indices(x1, x2, y, meta, indices)
        for split, indices in split_indices.items()
    }
    _assert_split_cycle_isolation(splits)
    return splits


def lifecycle_block_split_by_meta(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    meta: Sequence[Dict],
    block_cycles: int = 36,
    split_pattern: Sequence[str] = (
        "train",
        "val",
        "train",
        "train",
        "test",
        "train",
        "train",
        "val",
        "train",
        "train",
    ),
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]]:
    """Cover the lifecycle with disjoint raw-cycle blocks.

    Every source cell is partitioned into adjacent raw-cycle blocks. A sliding
    window is retained only when all of its raw cycles fall inside one block;
    windows crossing a block boundary are discarded. Blocks are then assigned
    with a repeating 7:2:1 train/validation/test pattern, so every split samples
    multiple lifecycle regions without sharing any raw cycle with another
    split.
    """
    valid_splits = {"train", "val", "test"}
    pattern = tuple(split_pattern)
    if block_cycles <= 0:
        raise ValueError("block_cycles must be positive.")
    if not pattern or set(pattern) != valid_splits:
        raise ValueError(
            "split_pattern must be non-empty and contain train, val and test."
        )

    split_indices: Dict[str, List[int]] = {name: [] for name in valid_splits}
    split_meta: Dict[str, List[Dict]] = {name: [] for name in valid_splits}

    for (dataset, record), ordered in _group_indices_by_record(meta).items():
        if not ordered:
            continue
        min_cycle = min(int(meta[idx]["start_cycle"]) for idx in ordered)
        max_cycle = max(int(meta[idx]["target_cycle"]) for idx in ordered)
        max_window_span = max(
            int(meta[idx]["target_cycle"]) - int(meta[idx]["start_cycle"]) + 1
            for idx in ordered
        )
        if block_cycles < max_window_span:
            raise ValueError(
                f"block_cycles={block_cycles} is shorter than the largest "
                f"window ({max_window_span}) for {dataset}/{record}."
            )

        record_counts = {name: 0 for name in valid_splits}
        for idx in ordered:
            start_cycle = int(meta[idx]["start_cycle"])
            target_cycle = int(meta[idx]["target_cycle"])
            block_id = (start_cycle - min_cycle) // block_cycles
            block_start = min_cycle + block_id * block_cycles
            block_end = min(block_start + block_cycles - 1, max_cycle)

            # Discard every stride-one window that crosses a block boundary.
            if target_cycle > block_end:
                continue

            split = pattern[block_id % len(pattern)]
            split_indices[split].append(idx)
            split_meta[split].append(
                {
                    **meta[idx],
                    "split_scheme": "lifecycle_disjoint_raw_cycle_blocks",
                    "split_block_id": block_id,
                    "split_block_start_cycle": block_start,
                    "split_block_end_cycle": block_end,
                    "split_block_cycles": block_cycles,
                }
            )
            record_counts[split] += 1

        empty = [name for name, count in record_counts.items() if count == 0]
        if empty:
            raise ValueError(
                f"{dataset}/{record} has no samples for {empty}; decrease "
                "block_cycles or use a shorter split pattern."
            )

    splits = {
        split: (
            x1[np.asarray(split_indices[split], dtype=np.int64)],
            x2[np.asarray(split_indices[split], dtype=np.int64)],
            y[np.asarray(split_indices[split], dtype=np.int64)],
            split_meta[split],
        )
        for split in ("train", "val", "test")
    }
    _assert_split_cycle_isolation(splits)
    return splits


def purged_train_val_split_by_meta(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    meta: Sequence[Dict],
    train_ratio: float = 0.8,
    purge_gap: int = DEFAULT_WINDOW - 1,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]]:
    """Create chronological train/validation blocks for every source cell."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if purge_gap < 0:
        raise ValueError("purge_gap must be non-negative.")

    split_indices = {"train": [], "val": []}
    for key, ordered in _group_indices_by_record(meta).items():
        usable = len(ordered) - purge_gap
        if usable < 2:
            raise ValueError(
                f"{key} has {len(ordered)} windows, insufficient for "
                f"train/validation with purge_gap={purge_gap}."
            )
        n_train = max(1, int(usable * train_ratio))
        if n_train >= usable:
            n_train = usable - 1
        n_val = usable - n_train
        split_indices["train"].extend(ordered[:n_train])
        val_start = n_train + purge_gap
        split_indices["val"].extend(ordered[val_start : val_start + n_val])

    splits = {
        split: subset_by_indices(x1, x2, y, meta, indices)
        for split, indices in split_indices.items()
    }
    _assert_split_cycle_isolation(splits)
    return splits


def leave_one_record_out_purged_splits(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    meta: Sequence[Dict],
    val_ratio: float = 0.2,
    purge_gap: int = DEFAULT_WINDOW - 1,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]]]:
    """Build LOCO folds with a chronological source-cell validation block."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    groups = _group_indices_by_record(meta)
    folds = {}

    for holdout in sorted(groups):
        train_indices: List[int] = []
        val_indices: List[int] = []
        for key, ordered in groups.items():
            if key == holdout:
                continue
            usable = len(ordered) - purge_gap
            if usable < 2:
                raise ValueError(
                    f"{key} has {len(ordered)} windows, insufficient for "
                    f"train/validation with purge_gap={purge_gap}."
                )
            n_val = max(1, int(usable * val_ratio))
            n_train = usable - n_val
            if n_train < 1:
                raise ValueError(f"{key} produced an empty training split.")
            train_indices.extend(ordered[:n_train])
            val_start = n_train + purge_gap
            val_indices.extend(ordered[val_start : val_start + n_val])

        fold_name = f"test_{holdout[1]}"
        fold = {
            "train": subset_by_indices(x1, x2, y, meta, train_indices),
            "val": subset_by_indices(x1, x2, y, meta, val_indices),
            "test": subset_by_indices(x1, x2, y, meta, groups[holdout]),
        }
        _assert_split_cycle_isolation(fold)
        folds[fold_name] = fold
    return folds


def fixed_validation_record_splits(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    meta: Sequence[Dict],
    validation_record: str,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]]]:
    """Use one complete record for validation and rotate the other records as tests.

    For every fold, the validation and test records are completely isolated
    physical cells. All remaining records form the model-selection training
    split. No temporal purge is required because no record occurs in more than
    one split.
    """
    groups = _group_indices_by_record(meta)
    validation_keys = [key for key in groups if key[1] == validation_record]
    if len(validation_keys) != 1:
        raise ValueError(
            f"Expected exactly one record named {validation_record!r}; "
            f"found {len(validation_keys)}."
        )
    if len(groups) < 3:
        raise ValueError("At least three records are required.")

    validation_key = validation_keys[0]
    folds = {}
    for test_key in sorted(key for key in groups if key != validation_key):
        train_indices: List[int] = []
        for key, indices in groups.items():
            if key not in {validation_key, test_key}:
                train_indices.extend(indices)

        fold_name = f"test_{test_key[1]}"
        fold = {
            "train": subset_by_indices(x1, x2, y, meta, train_indices),
            "val": subset_by_indices(
                x1, x2, y, meta, groups[validation_key]
            ),
            "test": subset_by_indices(x1, x2, y, meta, groups[test_key]),
        }
        if any(len(split[2]) == 0 for split in fold.values()):
            raise ValueError(f"{fold_name} produced an empty split.")
        _assert_split_cycle_isolation(fold)
        folds[fold_name] = fold
    return folds


def nested_leave_one_record_out_splits(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    meta: Sequence[Dict],
) -> Dict[str, Dict]:
    """Build outer LOCO tests with source-only inner LOCO validation folds.

    Each outer fold holds out one complete physical record for final testing.
    Every remaining source record then serves once as the inner validation
    record while the other source records form the inner training split. The
    final split trains on every non-test record and keeps the outer test record
    untouched.
    """
    groups = _group_indices_by_record(meta)
    if len(groups) < 4:
        raise ValueError("Nested LOCO requires at least four records.")

    outer_folds: Dict[str, Dict] = {}
    for test_key in sorted(groups):
        source_keys = sorted(key for key in groups if key != test_key)
        inner_folds = {}
        for validation_key in source_keys:
            train_indices: List[int] = []
            for key in source_keys:
                if key != validation_key:
                    train_indices.extend(groups[key])
            inner_fold = {
                "train": subset_by_indices(
                    x1, x2, y, meta, train_indices
                ),
                "val": subset_by_indices(
                    x1, x2, y, meta, groups[validation_key]
                ),
            }
            if any(len(split[2]) == 0 for split in inner_fold.values()):
                raise ValueError(
                    f"test_{test_key[1]}/val_{validation_key[1]} "
                    "produced an empty split."
                )
            _assert_split_cycle_isolation(inner_fold)
            inner_folds[f"val_{validation_key[1]}"] = inner_fold

        final_train_indices: List[int] = []
        for key in source_keys:
            final_train_indices.extend(groups[key])
        final_split = {
            "train": subset_by_indices(
                x1, x2, y, meta, final_train_indices
            ),
            "test": subset_by_indices(
                x1, x2, y, meta, groups[test_key]
            ),
        }
        _assert_split_cycle_isolation(final_split)
        outer_folds[f"test_{test_key[1]}"] = {
            "inner": inner_folds,
            "final": final_split,
        }
    return outer_folds


def interleaved_split_by_meta(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    meta: Sequence[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]]:
    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, item in enumerate(meta):
        groups.setdefault((item["dataset"], item["record"]), []).append(i)

    split_indices = {"train": [], "val": [], "test": []}
    for indices in groups.values():
        ordered = sorted(indices, key=lambda idx: meta[idx]["target_cycle"])
        n = len(ordered)
        if n == 0:
            continue
        target = {
            "train": int(round(n * train_ratio)),
            "val": int(round(n * val_ratio)),
        }
        target["test"] = max(0, n - target["train"] - target["val"])
        counts = {"train": 0, "val": 0, "test": 0}
        train_slots = max(0, int(round(train_ratio * 10)))
        val_slots = max(0, int(round(val_ratio * 10)))
        if train_ratio > 0 and train_slots == 0:
            train_slots = 1
        if val_ratio > 0 and val_slots == 0:
            val_slots = 1
        test_slots = max(0, 10 - train_slots - val_slots)
        if test_slots == 0 and target["test"] > 0:
            test_slots = 1
            if train_slots >= val_slots and train_slots > 1:
                train_slots -= 1
            elif val_slots > 1:
                val_slots -= 1
        pattern = ["train"] * train_slots + ["val"] * val_slots + ["test"] * test_slots
        if not pattern:
            pattern = ["train"]
        for pos, idx in enumerate(ordered):
            preferred = pattern[pos % len(pattern)]
            if counts[preferred] < target[preferred]:
                assigned = preferred
            else:
                assigned = None
                for split in pattern:
                    if counts[split] < target[split]:
                        assigned = split
                        break
            if assigned is None:
                assigned = min(counts, key=lambda key: counts[key] / max(target[key], 1))
            split_indices[assigned].append(idx)
            counts[assigned] += 1

    return {split: subset_by_indices(x1, x2, y, meta, indices) for split, indices in split_indices.items()}


def random_split_arrays(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    meta: Sequence[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_train = int(len(idx) * train_ratio)
    n_val = int(len(idx) * val_ratio)
    mapping = {
        "train": idx[:n_train],
        "val": idx[n_train : n_train + n_val],
        "test": idx[n_train + n_val :],
    }
    return {k: (x1[v], x2[v], y[v], [meta[i] for i in v]) for k, v in mapping.items()}


def fit_scalers(train_x1: np.ndarray, train_x2: np.ndarray, train_y: np.ndarray):
    return {
        "x1": StandardScaler1D.fit(train_x1),
        "x2": StandardScaler1D.fit(train_x2),
        "y": MinMaxScaler1D.fit(train_y),
    }


def fit_input_scalers_identity_target(
    train_x1: np.ndarray,
    train_x2: np.ndarray,
) -> Dict[str, object]:
    """Fit input scalers while keeping a physical-scale regression target.

    This is useful for transfer experiments where all stages must interpret the
    model output in the same target space.  In particular, a frozen regression
    head must not be paired with a newly fitted target scaler.
    """
    return {
        "x1": StandardScaler1D.fit(train_x1),
        "x2": StandardScaler1D.fit(train_x2),
        "y": IdentityScaler1D(),
    }


def identity_scalers():
    return {"x1": IdentityScaler1D(), "x2": IdentityScaler1D(), "y": IdentityScaler1D()}


def apply_scalers(x1: np.ndarray, x2: np.ndarray, y: np.ndarray, scalers: Dict):
    return (
        scalers["x1"].transform(x1),
        scalers["x2"].transform(x2),
        scalers["y"].transform(y),
    )


def make_loader(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x1).float(), torch.from_numpy(x2).float(), torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def model_output_to_prediction(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 3:
        output = output[:, -1, :]
    return output


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    patience: Optional[int] = None,
    train_mode_callback: Optional[Callable[[nn.Module], None]] = None,
) -> Tuple[nn.Module, List[Dict]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr, betas=DEFAULT_ADAM_BETAS)
    best_state = copy.deepcopy(model.state_dict())
    best_val = math.inf
    stale = 0
    history = []

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        if train_mode_callback is not None:
            train_mode_callback(model)
        train_losses = []
        for xb1, xb2, yb in train_loader:
            xb1, xb2, yb = xb1.to(device), xb2.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb1, xb2)
            if isinstance(output, tuple):
                output = output[0]
            pred = model_output_to_prediction(output)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = evaluate_loss(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else math.nan,
            "val_loss": val_loss,
        }
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if patience is not None and stale >= patience:
            break

    model.load_state_dict(best_state)
    return model, history


def train_model_fixed_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float = DEFAULT_LR,
) -> Tuple[nn.Module, List[Dict]]:
    """Train for a pre-selected epoch count without reusing training as validation."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        betas=DEFAULT_ADAM_BETAS,
    )
    history = []
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb1, xb2, yb in train_loader:
            xb1, xb2, yb = xb1.to(device), xb2.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb1, xb2)
            if isinstance(output, tuple):
                output = output[0]
            pred = model_output_to_prediction(output)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "epoch": epoch,
                "train_loss": (
                    float(np.mean(train_losses)) if train_losses else math.nan
                ),
            }
        )
    return model, history


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for xb1, xb2, yb in loader:
            xb1, xb2, yb = xb1.to(device), xb2.to(device), yb.to(device)
            output = model(xb1, xb2)
            if isinstance(output, tuple):
                output = output[0]
            pred = model_output_to_prediction(output)
            losses.append(float(criterion(pred, yb).detach().cpu()))
    return float(np.mean(losses)) if losses else math.nan


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb1, xb2, yb in loader:
            xb1, xb2 = xb1.to(device), xb2.to(device)
            output = model(xb1, xb2)
            if isinstance(output, tuple):
                output = output[0]
            pred = model_output_to_prediction(output).detach().cpu().numpy()
            preds.append(pred)
            targets.append(yb.numpy())
    return np.vstack(preds), np.vstack(targets)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    max_error = float(np.max(np.abs(err))) if len(err) else math.nan
    bias = float(np.mean(err))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(err**2) / denom) if denom > 1e-12 else math.nan
    value_range = float(y_true.max() - y_true.min()) if len(y_true) else math.nan
    nrmse_range = float(rmse / value_range) if value_range > 1e-12 else math.nan
    return {
        "R2": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MaxError": max_error,
        "Bias": bias,
        "NRMSE_range": nrmse_range,
    }


def evaluate_model(
    model: nn.Module,
    split_data: Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]],
    scalers: Dict,
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict]]:
    x1, x2, y, meta = split_data
    sx1, sx2, sy = apply_scalers(x1, x2, y, scalers)
    loader = make_loader(sx1, sx2, sy, batch_size, shuffle=False)
    pred_scaled, target_scaled = predict(model, loader, device)
    pred = scalers["y"].inverse_transform(pred_scaled)
    target = scalers["y"].inverse_transform(target_scaled)
    metrics = regression_metrics(target, pred)
    scaled_metrics = regression_metrics(target_scaled, pred_scaled)
    metrics.update({f"Normalized_{key}": value for key, value in scaled_metrics.items()})
    rows = []
    for item, t, p in zip(meta, target.reshape(-1), pred.reshape(-1)):
        row = dict(item)
        row["target"] = float(t)
        row["prediction"] = float(p)
        row["error"] = float(p - t)
        rows.append(row)
    return metrics, rows


def metrics_by_record(prediction_rows: Sequence[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for row in prediction_rows:
        grouped.setdefault((row["dataset"], row["record"]), []).append(row)
    out = []
    for (dataset, record), rows in grouped.items():
        y_true = np.asarray([r["target"] for r in rows], dtype=float)
        y_pred = np.asarray([r["prediction"] for r in rows], dtype=float)
        row = {"dataset": dataset, "record": record, "n": len(rows)}
        row.update(regression_metrics(y_true, y_pred))
        out.append(row)
    return out


def create_model(name: str) -> nn.Module:
    if name == "CnnMambaImproved":
        return CnnMambaImproved(d_model=64, d_state=16, d_conv=4, expand=2, output_dim=1)
    if name == "CnnSE":
        return CnnSE(output_dim=1)
    if name == "CnnMambaSimple":
        return CnnMambaSimple(d_model=64, d_state=16, d_conv=4, expand=2, output_dim=1)
    if name == "CnnLstmAttention":
        return CnnLstmAttention(d_model=64, hidden_size=64, num_layers=1, output_dim=1)
    if name == "CnnTransformer":
        return CnnTransformer(d_model=64, nhead=4, num_layers=2, output_dim=1)
    if name == "CnnLSTMControl":
        return CnnLSTMControl(d_model=64, hidden_size=128, output_dim=1)
    if name == "ResNet1d":
        return ResNet1d(output_dim=1)
    raise ValueError(f"Unknown model name: {name}")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit run directory. The default is a timestamped directory.",
    )
    parser.add_argument(
        "--purge-gap",
        type=int,
        default=None,
        help="Discarded window positions; defaults to window_size - 1.",
    )
    return parser


def prepare_train_val_loaders(splits: Dict, scalers: Dict, batch_size: int):
    tx1, tx2, ty, _ = splits["train"]
    vx1, vx2, vy, _ = splits["val"]
    stx1, stx2, sty = apply_scalers(tx1, tx2, ty, scalers)
    svx1, svx2, svy = apply_scalers(vx1, vx2, vy, scalers)
    return (
        make_loader(stx1, stx2, sty, batch_size, shuffle=True),
        make_loader(svx1, svx2, svy, batch_size, shuffle=False),
    )


def save_training_outputs(
    out_dir: Path,
    model_name: str,
    model: nn.Module,
    history: Sequence[Dict],
    metrics: Dict,
    by_record: Sequence[Dict],
    predictions: Sequence[Dict],
    scalers: Dict,
    config: Dict,
) -> None:
    ensure_dir(out_dir)
    torch.save(model.state_dict(), out_dir / f"{model_name}_best.pt")
    write_csv(out_dir / f"{model_name}_history.csv", history)
    write_csv(out_dir / f"{model_name}_metrics_by_record.csv", by_record)
    write_csv(out_dir / f"{model_name}_predictions.csv", predictions)
    write_csv(out_dir / f"{model_name}_metrics.csv", [metrics])
    save_json(
        out_dir / f"{model_name}_config.json",
        {
            **config,
            "optimizer": "Adam",
            "adam_betas": list(DEFAULT_ADAM_BETAS),
            "scalers": {k: v.to_dict() for k, v in scalers.items()},
        },
    )


def save_split_manifest(
    path: Path,
    splits: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]],
) -> None:
    """Save the exact sample membership used by one experiment or fold."""
    rows = []
    for split, (_, _, _, split_meta) in splits.items():
        for item in split_meta:
            rows.append({"split": split, **item})
    write_csv(path, rows)
