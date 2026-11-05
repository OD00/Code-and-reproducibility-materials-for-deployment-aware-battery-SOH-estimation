from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "cycle_dict_qc_v2"
DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data/cycle_dict_v1"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/cycle_dict_qc_v2"

DATASET_CONFIG = {
    "CALCE": {
        "index_name": "calce_cycle_dict.pkl",
        "nominal_capacity_ah": 1.1,
        "charge_upper_voltage_min_v": 4.05,
        "discharge_start_voltage_min_v": 3.75,
        "discharge_cutoff_max_v": 3.00,
    },
    "TJU": {
        "index_name": "tju_cycle_dict.pkl",
        "nominal_capacity_ah": 3.5,
        "charge_upper_voltage_min_v": 4.05,
        "discharge_start_voltage_min_v": 3.75,
        "discharge_cutoff_max_v": 2.85,
    },
}

MIN_PHASE_DURATION_S = 300.0
MIN_CAPACITY_FRACTION = 0.20
MAX_CAPACITY_FRACTION = 1.30
MAX_CHARGE_DISCHARGE_IMBALANCE = 0.15
MAX_REFERENCE_RELATIVE_ERROR = 0.08
STABLE_CURRENT_RELATIVE_TOLERANCE = 0.05
STABLE_CURRENT_ABSOLUTE_TOLERANCE_A = 0.02
STABLE_CURRENT_CONSECUTIVE_POINTS = 3
SPIKE_WINDOW = 7
SPIKE_SIGMA = 6.0


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding=encoding)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def finite_values(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.float64)
    result = np.asarray(values, dtype=np.float64)
    return result[np.isfinite(result)]


def first_finite(values: np.ndarray | None) -> float | None:
    values = finite_values(values)
    return float(values[0]) if values.size else None


def last_finite(values: np.ndarray | None) -> float | None:
    values = finite_values(values)
    return float(values[-1]) if values.size else None


def safe_relative_error(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or not math.isfinite(left) or not math.isfinite(right):
        return None
    denominator = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / denominator


def source_format_map(cell: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["source_id"]): str(item.get("file_format") or "").lower()
        for item in cell["source_registry"]
    }


def correct_segment_time(segment: dict[str, Any], scale: float) -> None:
    data = segment["data"]
    if scale != 1.0:
        raw = data.get("test_time_s_raw")
        current = data.get("test_time_s")
        base = raw if raw is not None else current
        if base is not None:
            data["test_time_s"] = np.asarray(base, dtype=np.float64) * scale

    times = data.get("test_time_s")
    if times is None:
        data["elapsed_s"] = None
        return
    times = np.asarray(times, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(times))
    elapsed = times.copy()
    if finite.size:
        elapsed = elapsed - times[finite[0]]
        start = float(times[finite[0]])
        end = float(times[finite[-1]])
        segment["boundary"]["start_time_s"] = start
        segment["boundary"]["end_time_s"] = end
        segment["boundary"]["duration_s"] = end - start
    data["elapsed_s"] = elapsed


def all_segments(cycle: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for phase in ("charge", "discharge"):
        yield from cycle[phase]["segments"]
    yield from cycle.get("rest_segments", [])
    yield from cycle.get("unknown_segments", [])


def update_cycle_time(cycle: dict[str, Any], scale: float) -> None:
    for segment in all_segments(cycle):
        correct_segment_time(segment, scale)
    if scale != 1.0:
        for key in ("gap_before_s", "gap_after_s"):
            value = cycle["boundary"].get(key)
            if value is not None:
                cycle["boundary"][key] = float(value) * scale


def valid_pair_arrays(segment: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = segment["data"]
    times = data.get("test_time_s")
    current = data.get("current_a")
    if times is None or current is None:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.bool_),
        )
    times = np.asarray(times, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    usable = data.get("usable_mask")
    if usable is None:
        usable = np.ones(len(times), dtype=np.bool_)
    else:
        usable = np.asarray(usable, dtype=np.bool_)
    mask = np.isfinite(times) & np.isfinite(current) & usable
    return times, current, mask


def integrate_segments(segments: list[dict[str, Any]]) -> tuple[float | None, dict[str, int]]:
    total_as = 0.0
    usable_pairs = 0
    nonpositive_dt = 0
    invalid_pairs = 0
    for segment in segments:
        times, current, mask = valid_pair_arrays(segment)
        if len(times) < 2:
            continue
        dt = np.diff(times)
        pair_mask = mask[:-1] & mask[1:] & np.isfinite(dt)
        nonpositive_dt += int(np.count_nonzero(pair_mask & (dt <= 0)))
        valid = pair_mask & (dt > 0)
        invalid_pairs += int(len(dt) - np.count_nonzero(valid))
        if np.any(valid):
            average_current = (np.abs(current[:-1]) + np.abs(current[1:])) * 0.5
            total_as += float(np.sum(average_current[valid] * dt[valid]))
            usable_pairs += int(np.count_nonzero(valid))
    capacity = total_as / 3600.0 if usable_pairs else None
    return capacity, {
        "usable_integration_pairs": usable_pairs,
        "nonpositive_time_intervals": nonpositive_dt,
        "invalid_integration_pairs": invalid_pairs,
    }


def column_delta(segments: list[dict[str, Any]], field: str) -> float | None:
    total = 0.0
    found = False
    for segment in segments:
        values = finite_values(segment["data"].get(field))
        if values.size:
            delta = float(np.nanmax(values) - np.nanmin(values))
            if delta >= 0:
                total += delta
                found = True
    return total if found else None


def phase_values(segments: list[dict[str, Any]], field: str) -> np.ndarray:
    arrays = []
    for segment in segments:
        values = finite_values(segment["data"].get(field))
        if values.size:
            arrays.append(values)
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64)


def total_duration(segments: list[dict[str, Any]]) -> float | None:
    values = [segment["boundary"].get("duration_s") for segment in segments]
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(sum(finite)) if finite else None


def stable_start_offset(current: np.ndarray | None) -> tuple[int | None, float | None]:
    values = finite_values(current)
    if values.size < STABLE_CURRENT_CONSECUTIVE_POINTS:
        return None, None
    magnitudes = np.abs(values)
    target = float(np.quantile(magnitudes, 0.90))
    tolerance = max(
        STABLE_CURRENT_ABSOLUTE_TOLERANCE_A,
        STABLE_CURRENT_RELATIVE_TOLERANCE * target,
    )
    stable = np.abs(magnitudes - target) <= tolerance
    run = STABLE_CURRENT_CONSECUTIVE_POINTS
    for idx in range(0, len(stable) - run + 1):
        if bool(np.all(stable[idx : idx + run])):
            return idx, target
    return None, target


def isolated_spike_indices(
    values: np.ndarray | None, start_offset: int | None, absolute_floor: float
) -> list[int]:
    if values is None:
        return []
    values = np.asarray(values, dtype=np.float64)
    if len(values) < SPIKE_WINDOW:
        return []
    radius = SPIKE_WINDOW // 2
    windows = np.lib.stride_tricks.sliding_window_view(values, SPIKE_WINDOW)
    finite_counts = np.count_nonzero(np.isfinite(windows), axis=1)
    with np.errstate(invalid="ignore"):
        local_steps = np.diff(windows, axis=1)
        step_medians = np.nanmedian(local_steps, axis=1)
        step_mads = np.nanmedian(
            np.abs(local_steps - step_medians[:, None]), axis=1
        )
    centers = values[radius : len(values) - radius]
    left = values[radius - 1 : len(values) - radius - 1]
    right = values[radius + 1 : len(values) - radius + 1]
    neighbor_difference = np.abs(left - right)
    impulse_residual = np.abs(centers - (left + right) * 0.5)
    scale = np.maximum(SPIKE_SIGMA * 1.4826 * step_mads, absolute_floor)
    original_indices = np.arange(radius, len(values) - radius)
    valid = (
        (finite_counts >= 5)
        & np.isfinite(centers)
        & np.isfinite(left)
        & np.isfinite(right)
        & (original_indices >= max(radius, int(start_offset or 0)))
    )
    spikes = valid & (impulse_residual > scale) & (impulse_residual > 3.0 * neighbor_difference)
    return original_indices[spikes].astype(int).tolist()


def enrich_segment_diagnostics(segment: dict[str, Any]) -> None:
    data = segment["data"]
    offset, target = stable_start_offset(data.get("current_a"))
    times = data.get("elapsed_s")
    stable_time = None
    if offset is not None and times is not None and offset < len(times):
        candidate = float(times[offset])
        stable_time = candidate if math.isfinite(candidate) else None
    current_spikes = isolated_spike_indices(data.get("current_a"), offset, 0.05)
    voltage_spikes = isolated_spike_indices(data.get("voltage_v"), offset, 0.02)
    segment["diagnostics"] = {
        "stable_current_start_offset": offset,
        "stable_current_start_elapsed_s": stable_time,
        "estimated_constant_current_a": target,
        "transition_prefix_points": int(offset or 0),
        "isolated_current_spike_indices": current_spikes,
        "isolated_voltage_spike_indices": voltage_spikes,
        "raw_signal_modified": False,
    }


def process_cycle(
    cycle: dict[str, Any], dataset: str, config: dict[str, Any], time_scale: float
) -> dict[str, Any]:
    original_targets = dict(cycle.get("targets", {}))
    original_quality = dict(cycle.get("quality", {}))
    update_cycle_time(cycle, time_scale)

    charge_segments = cycle["charge"]["segments"]
    discharge_segments = cycle["discharge"]["segments"]
    for segment in charge_segments + discharge_segments:
        enrich_segment_diagnostics(segment)

    charge_q, charge_time = integrate_segments(charge_segments)
    discharge_q, discharge_time = integrate_segments(discharge_segments)
    charge_column = column_delta(charge_segments, "charge_capacity_ah")
    discharge_column = column_delta(discharge_segments, "discharge_capacity_ah")
    if dataset == "CALCE" and not charge_segments:
        charge_column = None
    if dataset == "CALCE" and not discharge_segments:
        discharge_column = None

    charge_voltage = phase_values(charge_segments, "voltage_v")
    discharge_voltage = phase_values(discharge_segments, "voltage_v")
    charge_duration = total_duration(charge_segments)
    discharge_duration = total_duration(discharge_segments)
    charge_end = last_finite(charge_voltage)
    charge_max = float(np.max(charge_voltage)) if charge_voltage.size else None
    discharge_start = first_finite(discharge_voltage)
    discharge_end = last_finite(discharge_voltage)
    discharge_min = float(np.min(discharge_voltage)) if discharge_voltage.size else None

    balance = safe_relative_error(charge_q, discharge_q)
    charge_ref_error = safe_relative_error(charge_q, charge_column)
    discharge_ref_error = safe_relative_error(discharge_q, discharge_column)
    nominal = float(config["nominal_capacity_ah"])

    flags = set(str(value) for value in original_quality.get("flags", []))
    standard_phase_order = cycle["boundary"].get("active_phase_order") == ["charge", "discharge"]
    if not cycle["boundary"].get("is_complete_pair", False):
        flags.add("incomplete_charge_discharge_pair")
    if not standard_phase_order:
        flags.add("nonstandard_active_phase_order")
    if len(charge_segments) > 1 or len(discharge_segments) > 1:
        flags.add("multiple_active_segments")
    if charge_time["nonpositive_time_intervals"] or discharge_time["nonpositive_time_intervals"]:
        flags.add("nonmonotonic_time_within_active_phase")
    if charge_duration is None or charge_duration < MIN_PHASE_DURATION_S:
        flags.add("charge_duration_too_short")
    if discharge_duration is None or discharge_duration < MIN_PHASE_DURATION_S:
        flags.add("discharge_duration_too_short")
    if charge_max is None or charge_max < float(config["charge_upper_voltage_min_v"]):
        flags.add("charge_upper_voltage_not_reached")
    if discharge_start is None or discharge_start < float(config["discharge_start_voltage_min_v"]):
        flags.add("discharge_started_from_low_voltage")
    if discharge_min is None or discharge_min > float(config["discharge_cutoff_max_v"]):
        flags.add("discharge_cutoff_not_reached")
    if discharge_q is None:
        flags.add("missing_integrated_discharge_capacity")
    elif not (MIN_CAPACITY_FRACTION * nominal <= discharge_q <= MAX_CAPACITY_FRACTION * nominal):
        flags.add("discharge_capacity_outside_physical_range")
    if charge_q is None:
        flags.add("missing_integrated_charge_capacity")
    elif not (MIN_CAPACITY_FRACTION * nominal <= charge_q <= MAX_CAPACITY_FRACTION * nominal):
        flags.add("charge_capacity_outside_physical_range")
    if balance is not None and balance > MAX_CHARGE_DISCHARGE_IMBALANCE:
        flags.add("charge_discharge_capacity_imbalance")
    if charge_ref_error is not None and charge_ref_error > MAX_REFERENCE_RELATIVE_ERROR:
        flags.add("charge_capacity_reference_mismatch")
    if discharge_ref_error is not None and discharge_ref_error > MAX_REFERENCE_RELATIVE_ERROR:
        flags.add("discharge_capacity_reference_mismatch")

    current_spikes = sum(
        len(segment["diagnostics"]["isolated_current_spike_indices"])
        for segment in charge_segments + discharge_segments
    )
    voltage_spikes = sum(
        len(segment["diagnostics"]["isolated_voltage_spike_indices"])
        for segment in charge_segments + discharge_segments
    )
    transition_points = sum(
        segment["diagnostics"]["transition_prefix_points"]
        for segment in charge_segments + discharge_segments
    )
    if current_spikes:
        flags.add("isolated_current_spike_candidate")
    if voltage_spikes:
        flags.add("isolated_voltage_spike_candidate")
    if transition_points:
        flags.add("active_phase_start_transition_detected")

    hard_invalid = {
        "incomplete_charge_discharge_pair",
        "nonstandard_active_phase_order",
        "nonmonotonic_time_within_active_phase",
        "charge_duration_too_short",
        "discharge_duration_too_short",
        "charge_upper_voltage_not_reached",
        "discharge_started_from_low_voltage",
        "discharge_cutoff_not_reached",
        "missing_integrated_discharge_capacity",
        "missing_integrated_charge_capacity",
        "discharge_capacity_outside_physical_range",
        "charge_capacity_outside_physical_range",
        "charge_discharge_capacity_imbalance",
    }
    valid_label = not bool(hard_invalid & flags)

    cycle["targets"] = {
        "original_v1": original_targets,
        "charge_capacity_integrated_ah": charge_q,
        "discharge_capacity_integrated_ah": discharge_q,
        "charge_capacity_column_delta_ah": charge_column,
        "discharge_capacity_column_delta_ah": discharge_column,
        "capacity_label_ah": discharge_q,
        "capacity_label_qc_ah": discharge_q if valid_label else None,
        "capacity_label_source": "trapezoidal_integral_abs_current_over_corrected_time",
    }
    cycle["qc"] = {
        "time_scale_applied": time_scale,
        "charge_duration_s": charge_duration,
        "discharge_duration_s": discharge_duration,
        "charge_start_voltage_v": first_finite(charge_voltage),
        "charge_end_voltage_v": charge_end,
        "charge_max_voltage_v": charge_max,
        "discharge_start_voltage_v": discharge_start,
        "discharge_end_voltage_v": discharge_end,
        "discharge_min_voltage_v": discharge_min,
        "charge_discharge_balance_ratio": balance,
        "charge_capacity_reference_relative_error": charge_ref_error,
        "discharge_capacity_reference_relative_error": discharge_ref_error,
        "charge_time_diagnostics": charge_time,
        "discharge_time_diagnostics": discharge_time,
        "transition_prefix_points": transition_points,
        "isolated_current_spike_count": current_spikes,
        "isolated_voltage_spike_count": voltage_spikes,
        "capacity_delta_prev_ah": None,
        "capacity_delta_prev_fraction_nominal": None,
        "capacity_outlier_candidate": False,
        "possible_physical_recovery": False,
        "isolated_invalid_capacity_dip": False,
    }
    cycle["quality"] = {
        **original_quality,
        "flags_v1": sorted(set(str(value) for value in original_quality.get("flags", []))),
        "flags": sorted(flags),
        "valid_capacity_label": valid_label,
        "valid_for_primary_training": valid_label,
        "raw_signal_modified": False,
        "interpolated": False,
        "notes": "Raw voltage/current samples retained; QC flags are non-destructive.",
    }
    return cycle


def rolling_capacity_flags(cell: dict[str, Any], nominal: float) -> None:
    cycles = list(cell["cycles"].values())
    capacities = np.asarray(
        [
            float(cycle["targets"]["capacity_label_ah"])
            if cycle["targets"]["capacity_label_ah"] is not None
            else np.nan
            for cycle in cycles
        ],
        dtype=np.float64,
    )
    discharge_starts = np.asarray(
        [
            float(cycle["qc"]["discharge_start_voltage_v"])
            if cycle["qc"]["discharge_start_voltage_v"] is not None
            else np.nan
            for cycle in cycles
        ],
        dtype=np.float64,
    )
    minimum_deviation = 0.03 * nominal
    for idx, cycle in enumerate(cycles):
        q = capacities[idx]
        if idx > 0 and math.isfinite(q) and math.isfinite(capacities[idx - 1]):
            delta = float(q - capacities[idx - 1])
            cycle["qc"]["capacity_delta_prev_ah"] = delta
            cycle["qc"]["capacity_delta_prev_fraction_nominal"] = delta / nominal

        left = max(0, idx - 4)
        right = min(len(cycles), idx + 5)
        window = capacities[left:right]
        window = window[np.isfinite(window)]
        if not math.isfinite(q) or window.size < 5:
            continue
        median = float(np.median(window))
        mad = float(np.median(np.abs(window - median)))
        threshold = max(3.0 * 1.4826 * mad, minimum_deviation)
        is_candidate = abs(float(q) - median) > threshold
        cycle["qc"]["capacity_outlier_candidate"] = bool(is_candidate)
        if is_candidate:
            cycle["quality"]["flags"] = sorted(
                set(cycle["quality"]["flags"]) | {"capacity_outlier_candidate"}
            )

        if 0 < idx < len(cycles) - 1:
            previous_q = capacities[idx - 1]
            next_q = capacities[idx + 1]
            neighboring_start = np.asarray(
                [discharge_starts[idx - 1], discharge_starts[idx + 1]],
                dtype=np.float64,
            )
            neighboring_start = neighboring_start[np.isfinite(neighboring_start)]
            low_start_relative_to_neighbors = (
                math.isfinite(discharge_starts[idx])
                and neighboring_start.size == 2
                and discharge_starts[idx] < float(np.median(neighboring_start)) - 0.08
            )
            isolated_low_start_dip = (
                math.isfinite(previous_q)
                and math.isfinite(next_q)
                and q < previous_q - minimum_deviation
                and abs(float(next_q - previous_q)) <= minimum_deviation
                and low_start_relative_to_neighbors
            )
            if isolated_low_start_dip:
                cycle["quality"]["valid_capacity_label"] = False
                cycle["quality"]["valid_for_primary_training"] = False
                cycle["targets"]["capacity_label_qc_ah"] = None
                cycle["quality"]["flags"] = sorted(
                    set(cycle["quality"]["flags"])
                    | {"isolated_low_start_capacity_dip"}
                )
            isolated_dip = (
                math.isfinite(previous_q)
                and math.isfinite(next_q)
                and q < previous_q - minimum_deviation
                and abs(float(next_q - previous_q)) <= minimum_deviation
                and not cycle["quality"]["valid_capacity_label"]
            )
            cycle["qc"]["isolated_invalid_capacity_dip"] = bool(isolated_dip)
            if isolated_dip:
                cycle["quality"]["flags"] = sorted(
                    set(cycle["quality"]["flags"]) | {"isolated_invalid_capacity_dip"}
                )

        if idx > 0 and cycle["quality"]["valid_capacity_label"]:
            previous_q = capacities[idx - 1]
            future = capacities[idx : min(len(cycles), idx + 3)]
            future_cycles = cycles[idx : min(len(cycles), idx + 3)]
            persistent = (
                math.isfinite(previous_q)
                and cycles[idx - 1]["quality"]["valid_capacity_label"]
                and q > previous_q + minimum_deviation
                and len(future) == 3
                and all(item["quality"]["valid_capacity_label"] for item in future_cycles)
                and bool(np.all(np.isfinite(future)))
                and float(np.median(future)) > previous_q + minimum_deviation
            )
            cycle["qc"]["possible_physical_recovery"] = bool(persistent)
            if persistent:
                cycle["quality"]["flags"] = sorted(
                    set(cycle["quality"]["flags"]) | {"possible_physical_recovery"}
                )


def qc_row(dataset: str, cell_id: str, cycle_key: str, cycle: dict[str, Any]) -> dict[str, Any]:
    qc = cycle["qc"]
    target = cycle["targets"]
    return {
        "dataset": dataset,
        "cell_id": cell_id,
        "cycle_key": cycle_key,
        "observed_cycle_index": cycle["observed_cycle_index"],
        "cycle_uid": cycle["cycle_uid"],
        "source_file": cycle["source"]["source_file"],
        "source_cycle_id": cycle["raw_cycle_identity"]["source_cycle_id"],
        "time_scale_applied": qc["time_scale_applied"],
        "charge_capacity_integrated_ah": target["charge_capacity_integrated_ah"],
        "discharge_capacity_integrated_ah": target["discharge_capacity_integrated_ah"],
        "capacity_label_qc_ah": target["capacity_label_qc_ah"],
        "charge_capacity_column_delta_ah": target["charge_capacity_column_delta_ah"],
        "discharge_capacity_column_delta_ah": target["discharge_capacity_column_delta_ah"],
        "charge_duration_s": qc["charge_duration_s"],
        "discharge_duration_s": qc["discharge_duration_s"],
        "charge_max_voltage_v": qc["charge_max_voltage_v"],
        "discharge_start_voltage_v": qc["discharge_start_voltage_v"],
        "discharge_min_voltage_v": qc["discharge_min_voltage_v"],
        "charge_discharge_balance_ratio": qc["charge_discharge_balance_ratio"],
        "capacity_delta_prev_ah": qc["capacity_delta_prev_ah"],
        "capacity_outlier_candidate": qc["capacity_outlier_candidate"],
        "possible_physical_recovery": qc["possible_physical_recovery"],
        "isolated_invalid_capacity_dip": qc["isolated_invalid_capacity_dip"],
        "transition_prefix_points": qc["transition_prefix_points"],
        "isolated_current_spike_count": qc["isolated_current_spike_count"],
        "isolated_voltage_spike_count": qc["isolated_voltage_spike_count"],
        "valid_capacity_label": cycle["quality"]["valid_capacity_label"],
        "valid_for_primary_training": cycle["quality"]["valid_for_primary_training"],
        "flags": ";".join(cycle["quality"]["flags"]),
    }


def process_dataset(input_root: Path, output_root: Path, dataset: str) -> dict[str, Any]:
    config = DATASET_CONFIG[dataset]
    input_dataset = input_root / dataset
    output_dataset = output_root / dataset
    output_cells = output_dataset / "cells"
    output_cells.mkdir(parents=True, exist_ok=True)
    with (input_dataset / config["index_name"]).open("rb") as handle:
        old_index = pickle.load(handle)

    rows: list[dict[str, Any]] = []
    new_cells: dict[str, Any] = {}
    dataset_flags: Counter[str] = Counter()
    total_cycles = 0
    valid_cycles = 0
    time_corrected_cycles = 0

    for cell_id, item in old_index["cells"].items():
        input_path = input_dataset / item["file"]
        with input_path.open("rb") as handle:
            cell = pickle.load(handle)
        formats = source_format_map(cell)
        cell["schema_version"] = SCHEMA_VERSION
        cell["created_at"] = now_iso()
        cell["metadata"]["nominal_capacity_ah"] = config["nominal_capacity_ah"]
        cell["processing"] = {
            "input_schema": old_index["schema_version"],
            "capacity_method": "trapezoidal integral of absolute current over corrected test time",
            "raw_voltage_current_modified": False,
            "interpolation_performed": False,
            "qc_thresholds": {
                "minimum_phase_duration_s": MIN_PHASE_DURATION_S,
                "capacity_fraction_range": [MIN_CAPACITY_FRACTION, MAX_CAPACITY_FRACTION],
                "maximum_charge_discharge_imbalance": MAX_CHARGE_DISCHARGE_IMBALANCE,
                "maximum_reference_relative_error": MAX_REFERENCE_RELATIVE_ERROR,
                **{key: value for key, value in config.items() if key != "index_name"},
            },
        }
        for source in cell["source_registry"]:
            if dataset == "CALCE" and str(source.get("file_format", "")).lower() == "txt":
                source["test_time_correction"] = "test_time_s = test_time_s_raw * 60"
                source["raw_capacity_usage"] = "not used as Ah label"

        for cycle in cell["cycles"].values():
            source_format = formats.get(str(cycle["source"]["source_id"]), "")
            scale = 60.0 if dataset == "CALCE" and source_format == "txt" else 1.0
            if scale != 1.0:
                time_corrected_cycles += 1
            process_cycle(cycle, dataset, config, scale)
        rolling_capacity_flags(cell, float(config["nominal_capacity_ah"]))

        cell_flags: Counter[str] = Counter()
        cell_valid = 0
        for cycle_key, cycle in cell["cycles"].items():
            total_cycles += 1
            if cycle["quality"]["valid_for_primary_training"]:
                valid_cycles += 1
                cell_valid += 1
            for flag in cycle["quality"]["flags"]:
                dataset_flags[flag] += 1
                cell_flags[flag] += 1
            rows.append(qc_row(dataset, cell_id, cycle_key, cycle))
        cell["qc_summary"] = {
            "cycle_count": len(cell["cycles"]),
            "valid_for_primary_training": cell_valid,
            "invalid_for_primary_training": len(cell["cycles"]) - cell_valid,
            "flag_counts": dict(sorted(cell_flags.items())),
        }
        output_path = output_cells / f"{cell_id}.pkl"
        atomic_pickle(output_path, cell)
        new_cells[cell_id] = {
            "file": str(output_path.relative_to(output_dataset)).replace("\\", "/"),
            "summary": cell["qc_summary"],
            "sha256": sha256_file(output_path),
        }
        print(f"{dataset}: {cell_id}: {cell_valid}/{len(cell['cycles'])} primary-valid")

    fieldnames = list(rows[0].keys())
    csv_path = output_dataset / "cycle_qc_table.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, csv_path)

    index = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "dataset": dataset,
        "storage_mode": "per-cell full sequence dictionaries plus cycle QC table",
        "input_root": str(input_dataset),
        "raw_voltage_current_modified": False,
        "interpolation_performed": False,
        "cells": new_cells,
    }
    atomic_pickle(output_dataset / config["index_name"], index)
    atomic_text(
        output_dataset / "dataset_index.json",
        json.dumps(index, ensure_ascii=False, indent=2),
    )
    checksum_lines = [
        f"{item['sha256']}  {item['file']}" for item in new_cells.values()
    ]
    checksum_lines.append(f"{sha256_file(csv_path)}  cycle_qc_table.csv")
    atomic_text(output_dataset / "checksums.sha256", "\n".join(checksum_lines) + "\n")
    summary = {
        "dataset": dataset,
        "cells": len(new_cells),
        "cycles": total_cycles,
        "valid_for_primary_training": valid_cycles,
        "invalid_for_primary_training": total_cycles - valid_cycles,
        "time_corrected_cycles": time_corrected_cycles,
        "flag_counts": dict(sorted(dataset_flags.items())),
    }
    atomic_text(
        output_dataset / "qc_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    return summary


def write_method_document(output_root: Path, summaries: list[dict[str, Any]]) -> None:
    summary_lines = []
    for item in summaries:
        summary_lines.append(
            f"| {item['dataset']} | {item['cells']} | {item['cycles']} | "
            f"{item['valid_for_primary_training']} | {item['invalid_for_primary_training']} | "
            f"{item['time_corrected_cycles']} |"
        )
    text = f"""# 周期容量重建与质量控制说明（cycle_dict_qc_v2）

## 处理边界

- 输入：`cycle_dict_v1` 中已经完成阶段分割的变长周期字典。
- 输出：`cycle_dict_qc_v2`。
- 不覆盖原始字典，不删除原始循环。
- 不进行插值、重采样、归一化、平滑或数据集划分。
- 电压、电流数组及其采样点数量保持不变。

## 已执行的方法

1. CALCE TXT 的测试时间按分钟换算为秒：`test_time_s = test_time_s_raw × 60`。
2. CALCE TXT 的原 `Capacity` 字段不再解释为 Ah。
3. 所有电芯的充放电容量统一使用电流对修正时间的梯形积分重建。
4. CALCE XLSX/TJU 的容量列改用阶段内增量，仅作为积分结果的交叉检查。
5. 检查阶段完整性、时间单调性、持续时间、电压边界、容量物理范围和充放电容量平衡。
6. 容量跳变仅做候选标记；物理完整并持续存在的上升标为可能恢复，不做平滑。
7. 检测活动阶段起始过渡区和稳态段内孤立尖峰，只保存位置，不修改原始信号。

## 汇总

| 数据集 | 电芯 | 周期 | 主实验可用 | 主实验暂不使用 | 时间单位修正周期 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(summary_lines)}

## 主要字段

- `targets.capacity_label_ah`：积分重建后的放电容量，无论周期是否有效均保留。
- `targets.capacity_label_qc_ah`：仅在物理质量检查通过时保留，否则为 `None`。
- `quality.valid_for_primary_training`：后续构造主实验样本的入口标记。
- `quality.flags`：不通过或需要复核的原因。
- `qc`：持续时间、电压边界、容量平衡、跳变和恢复候选等指标。
- `segment.diagnostics`：稳定电流开始位置、起始过渡点数和孤立尖峰位置。

当前 QC 阈值是预注册式的物理筛查阈值，不代表最终神经网络阈值。任何阈值调整都应保存版本并重新生成汇总，不能只手工删除个别循环。
"""
    atomic_text(output_root / "周期容量重建与质量控制说明.md", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", choices=("all", "CALCE", "TJU"), default="all")
    args = parser.parse_args()
    selected = list(DATASET_CONFIG) if args.dataset == "all" else [args.dataset]
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = [process_dataset(args.input, args.output, dataset) for dataset in selected]
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "input_root": str(args.input),
        "output_root": str(args.output),
        "python": os.sys.version,
        "numpy": np.__version__,
        "interpolation_performed": False,
        "raw_voltage_current_modified": False,
        "summaries": summaries,
    }
    atomic_text(
        args.output / "processing_report.json",
        json.dumps(report, ensure_ascii=False, indent=2),
    )
    write_method_document(args.output, summaries)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
