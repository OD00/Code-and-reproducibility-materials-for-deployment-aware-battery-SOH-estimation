"""A1 data access and split rules for the leakage-safe model_input_v1 files.

This module is intentionally local to A1.  The other experiment groups still
use their existing loaders until their protocols are migrated explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from src.common import IdentityScaler1D, StandardScaler1D


DEFAULT_MODEL_INPUT_ROOT = (
    Path(__file__).resolve().parents[2] / "data" / "model_input_v1"
)

CALCE_CONDITION_CELLS = {
    "CALCE_group_0.5C": ("CS2_8", "CS2_21", "CS2_33", "CS2_34"),
    "CALCE_group_1C": ("CS2_35", "CS2_36", "CS2_37", "CS2_38"),
}


@dataclass(frozen=True)
class CellWindows:
    """All valid 12-cycle windows from one physical cell."""

    dataset: str
    cell_id: str
    condition_group: str
    source_file: Path
    voltage: np.ndarray
    current: np.ndarray
    duration: np.ndarray
    soh: np.ndarray
    capacity_ah: np.ndarray
    metadata: tuple[dict, ...]

    @property
    def sample_count(self) -> int:
        return int(self.soh.shape[0])


@dataclass(frozen=True)
class A1Fold:
    """One outer fold and its source-only inner validation folds."""

    name: str
    condition_group: str
    test_cell: str
    source_cells: tuple[str, ...]

    @property
    def inner_validation_cells(self) -> tuple[str, ...]:
        return self.source_cells


def _scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(()).item())


def load_cell_windows(path: Path) -> CellWindows:
    """Load and validate one final NPZ without allowing pickle objects."""

    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version",
            "dataset",
            "cell_id",
            "condition_group",
            "interpolation_points",
            "window_size",
            "observed_cycle_index",
            "window_row_indices",
            "window_target_row_index",
            "window_target_cycle_uid",
            "window_target_capacity_ah",
            "window_target_soh_nominal",
            "charge_voltage_v",
            "charge_current_a",
            "charge_duration_s",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path.name} is missing fields: {missing}")

        schema = _scalar_text(data["schema_version"])
        if schema != "model_input_v1":
            raise ValueError(f"Unexpected schema in {path.name}: {schema}")

        window_size = int(np.asarray(data["window_size"]).reshape(()))
        points = int(np.asarray(data["interpolation_points"]).reshape(()))
        row_indices = np.asarray(data["window_row_indices"], dtype=np.int64)
        if row_indices.ndim != 2 or row_indices.shape[1] != window_size:
            raise ValueError(
                f"Invalid window_row_indices in {path.name}: {row_indices.shape}"
            )

        cycle_indices = np.asarray(data["observed_cycle_index"], dtype=np.int64)
        if row_indices.size and (
            row_indices.min() < 0 or row_indices.max() >= len(cycle_indices)
        ):
            raise ValueError(f"Window row index is out of range in {path.name}")
        observed_windows = cycle_indices[row_indices]
        if observed_windows.size and not np.all(np.diff(observed_windows, axis=1) == 1):
            raise ValueError(f"A window crosses a cycle gap in {path.name}")

        voltage_cycles = np.asarray(data["charge_voltage_v"], dtype=np.float32)
        current_cycles = np.asarray(data["charge_current_a"], dtype=np.float32)
        duration_cycles = np.asarray(data["charge_duration_s"], dtype=np.float32)
        if voltage_cycles.shape != current_cycles.shape:
            raise ValueError(f"Voltage/current shapes differ in {path.name}")
        if voltage_cycles.ndim != 2 or voltage_cycles.shape[1] != points:
            raise ValueError(f"Unexpected charge curve shape in {path.name}")
        if duration_cycles.shape != (voltage_cycles.shape[0], 1):
            raise ValueError(f"Unexpected duration shape in {path.name}")

        voltage = voltage_cycles[row_indices]
        current = current_cycles[row_indices]
        duration = duration_cycles[row_indices]
        soh = np.asarray(data["window_target_soh_nominal"], dtype=np.float32)
        capacity = np.asarray(data["window_target_capacity_ah"], dtype=np.float32)
        target_rows = np.asarray(data["window_target_row_index"], dtype=np.int64)
        target_uids = np.asarray(data["window_target_cycle_uid"])

        n = row_indices.shape[0]
        expected_shapes = {
            "voltage": (n, window_size, points),
            "current": (n, window_size, points),
            "duration": (n, window_size, 1),
            "soh": (n, 1),
            "capacity": (n, 1),
            "target_rows": (n,),
            "target_uids": (n,),
        }
        actual_shapes = {
            "voltage": voltage.shape,
            "current": current.shape,
            "duration": duration.shape,
            "soh": soh.shape,
            "capacity": capacity.shape,
            "target_rows": target_rows.shape,
            "target_uids": target_uids.shape,
        }
        if actual_shapes != expected_shapes:
            raise ValueError(
                f"Window field shapes are inconsistent in {path.name}: "
                f"{actual_shapes}"
            )
        arrays = (voltage, current, duration, soh, capacity)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError(f"Non-finite model input found in {path.name}")
        if np.any(duration <= 0):
            raise ValueError(f"Non-positive charge duration found in {path.name}")

        dataset = _scalar_text(data["dataset"])
        cell_id = _scalar_text(data["cell_id"])
        condition_group = _scalar_text(data["condition_group"])
        metadata = tuple(
            {
                "dataset": dataset,
                "record": cell_id,
                "condition_group": condition_group,
                "start_cycle": int(observed_windows[index, 0]),
                "target_cycle": int(observed_windows[index, -1]),
                "target_cycle_uid": str(target_uids[index]),
                "target_row_index": int(target_rows[index]),
                "capacity_ah": float(capacity[index, 0]),
                "soh_nominal": float(soh[index, 0]),
                "source_file": str(path),
                "source_sample_index": index,
            }
            for index in range(n)
        )

    return CellWindows(
        dataset=dataset,
        cell_id=cell_id,
        condition_group=condition_group,
        source_file=path,
        voltage=voltage,
        current=current,
        duration=duration,
        soh=soh,
        capacity_ah=capacity,
        metadata=metadata,
    )


def load_calce_cells(data_root: Path = DEFAULT_MODEL_INPUT_ROOT) -> Dict[str, CellWindows]:
    """Load the exact eight CALCE cells used by the revised A1 experiment."""

    calce_dir = Path(data_root) / "CALCE"
    expected = {
        cell
        for cells in CALCE_CONDITION_CELLS.values()
        for cell in cells
    }
    loaded = {
        cell: load_cell_windows(calce_dir / f"{cell}.npz")
        for cell in sorted(expected)
    }
    for group, cells in CALCE_CONDITION_CELLS.items():
        for cell in cells:
            record = loaded[cell]
            if record.dataset != "CALCE":
                raise ValueError(f"{cell} reports dataset={record.dataset}")
            if record.cell_id != cell:
                raise ValueError(f"{cell}.npz reports cell_id={record.cell_id}")
            if record.condition_group != group:
                raise ValueError(
                    f"{cell} reports {record.condition_group}; expected {group}"
                )
    return loaded


def build_stratified_fold_plan(
    condition_cells: Mapping[str, Sequence[str]] = CALCE_CONDITION_CELLS,
) -> Dict[str, A1Fold]:
    """Build independent within-condition nested LOCO folds.

    Holding out one cell leaves three same-condition source cells.  Each source
    cell then serves once as a complete inner validation cell while the other
    two source cells train the candidate epoch.
    """

    folds: Dict[str, A1Fold] = {}
    for group, raw_cells in condition_cells.items():
        cells = tuple(raw_cells)
        if len(cells) < 3 or len(set(cells)) != len(cells):
            raise ValueError(f"Condition {group} needs at least 3 unique cells")
        for test_cell in cells:
            sources = tuple(cell for cell in cells if cell != test_cell)
            name = f"{group}__test_{test_cell}"
            folds[name] = A1Fold(name, group, test_cell, sources)
    return folds


def stack_cells(
    cells: Mapping[str, CellWindows],
    selected: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[dict, ...]]:
    """Concatenate complete physical-cell window sets without re-windowing."""

    names = tuple(selected)
    if not names:
        raise ValueError("At least one cell must be selected")
    records = [cells[name] for name in names]
    return (
        np.concatenate([record.voltage for record in records], axis=0),
        np.concatenate([record.current for record in records], axis=0),
        np.concatenate([record.duration for record in records], axis=0),
        np.concatenate([record.soh for record in records], axis=0),
        tuple(item for record in records for item in record.metadata),
    )


def fit_a1_scalers(
    voltage: np.ndarray,
    current: np.ndarray,
    duration: np.ndarray,
) -> dict[str, object]:
    """Fit every input scaler on training cells only; keep SOH physical."""

    return {
        "x1": StandardScaler1D.fit(voltage),
        "x2": StandardScaler1D.fit(current),
        "duration": StandardScaler1D.fit(duration),
        "y": IdentityScaler1D(),
    }


def apply_a1_scalers(
    voltage: np.ndarray,
    current: np.ndarray,
    duration: np.ndarray,
    soh: np.ndarray,
    scalers: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        scalers["x1"].transform(voltage),
        scalers["x2"].transform(current),
        scalers["duration"].transform(duration),
        scalers["y"].transform(soh),
    )
