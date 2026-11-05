"""Leakage-safe loaders for the paper-facing experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CellWindows:
    cell_id: str
    path: Path
    voltage: np.ndarray
    current: np.ndarray
    duration: np.ndarray
    target: np.ndarray

    @property
    def size(self) -> int:
        return int(self.target.shape[0])


@dataclass(frozen=True)
class StandardScalers:
    voltage_mean: float
    voltage_std: float
    current_mean: float
    current_std: float
    duration_mean: float
    duration_std: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "StandardScalers":
        aliases = {
            "x1_mean": "voltage_mean",
            "x1_std": "voltage_std",
            "x2_mean": "current_mean",
            "x2_std": "current_std",
        }
        normalized = dict(value)
        for old, new in aliases.items():
            if old in normalized and new not in normalized:
                normalized[new] = normalized[old]
        return cls(**{field: float(normalized[field]) for field in cls.__dataclass_fields__})


def _safe_std(array: np.ndarray) -> float:
    value = float(np.std(array, dtype=np.float64))
    return value if value > 1e-8 else 1.0


def load_tju_cell(path: Path) -> CellWindows:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "window_row_indices",
            "window_target_soh_nominal",
            "charge_voltage_v",
            "charge_current_a",
            "charge_duration_s",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path.name} is missing keys: {missing}")
        rows = np.asarray(data["window_row_indices"], dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1] != 12 or rows.min() < 0:
            raise ValueError(f"Unexpected window_row_indices in {path.name}: {rows.shape}")
        voltage = np.asarray(data["charge_voltage_v"], dtype=np.float32)[rows]
        current = np.asarray(data["charge_current_a"], dtype=np.float32)[rows]
        duration = np.asarray(data["charge_duration_s"], dtype=np.float32)[rows]
        target = np.asarray(data["window_target_soh_nominal"], dtype=np.float32).reshape(-1, 1)
    expected = (target.shape[0], 12)
    if voltage.shape[:2] != expected or current.shape[:2] != expected:
        raise ValueError(f"Window/target mismatch in {path.name}")
    if duration.shape != (*expected, 1):
        raise ValueError(f"Unexpected duration shape in {path.name}: {duration.shape}")
    for name, array in (("voltage", voltage), ("current", current), ("duration", duration), ("target", target)):
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite {name} values in {path.name}")
    return CellWindows(path.stem, path.resolve(), voltage, current, duration, target)


def load_tju_cells(root: Path, rate: str) -> dict[str, CellWindows]:
    prefix = {"0.25C": "CY25-025_1-", "0.5C": "CY25-05_1-"}.get(rate)
    if prefix is None:
        raise ValueError(f"Unsupported TJU rate: {rate}")
    root = root.expanduser().resolve()
    cells = {path.stem: load_tju_cell(path) for path in sorted(root.glob(f"{prefix}*.npz"))}
    if not cells:
        raise FileNotFoundError(f"No {rate} cells found under {root}")
    return cells


def validate_roles(cells: dict[str, CellWindows], **roles: Iterable[str]) -> None:
    role_sets = {name: set(values) for name, values in roles.items()}
    for role, values in role_sets.items():
        unknown = sorted(values - set(cells))
        if unknown:
            raise ValueError(f"Unknown cells in {role}: {unknown}")
    names = list(role_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sorted(role_sets[left] & role_sets[right])
            if overlap:
                raise ValueError(f"Cell leakage between {left} and {right}: {overlap}")


def stack_cells(cells: dict[str, CellWindows], cell_ids: Iterable[str]):
    selected = [cells[cell_id] for cell_id in cell_ids]
    if not selected:
        raise ValueError("At least one cell is required")
    return (
        np.concatenate([cell.voltage for cell in selected]),
        np.concatenate([cell.current for cell in selected]),
        np.concatenate([cell.duration for cell in selected]),
        np.concatenate([cell.target for cell in selected]),
        np.concatenate([np.full(cell.size, cell.cell_id, dtype="U64") for cell in selected]),
    )


def fit_scalers(voltage: np.ndarray, current: np.ndarray, duration: np.ndarray) -> StandardScalers:
    return StandardScalers(
        float(np.mean(voltage, dtype=np.float64)),
        _safe_std(voltage),
        float(np.mean(current, dtype=np.float64)),
        _safe_std(current),
        float(np.mean(duration, dtype=np.float64)),
        _safe_std(duration),
    )


def apply_scalers(voltage, current, duration, scalers: StandardScalers):
    return (
        ((voltage - scalers.voltage_mean) / scalers.voltage_std).astype(np.float32),
        ((current - scalers.current_mean) / scalers.current_std).astype(np.float32),
        ((duration - scalers.duration_mean) / scalers.duration_std).astype(np.float32),
    )
