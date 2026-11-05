from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl


SCHEMA_VERSION = "cycle_dict_v1"
CURRENT_THRESHOLD_A = 0.01
CHINA_TZ = timezone(timedelta(hours=8))

CALCE_INPUT = Path(__file__).resolve().parents[1] / "data/initial/CALCE/records/CALCE"
TJU_INPUT = Path(__file__).resolve().parents[1] / "data/initial/TJU/records/TJU_Dataset_1"

CALCE_CELLS = (
    "CS2_8",
    "CS2_21",
    "CS2_33",
    "CS2_34",
    "CS2_35",
    "CS2_36",
    "CS2_37",
    "CS2_38",
)
TJU_GROUPS = (
    "CY25-025_1",
    "CY25-05_1",
    "CY25-1_1",
    "CY35-05_1",
    "CY45-05_1",
)

# These pairs were found by a read-only, core-column equality audit. The script
# verifies equality again before excluding the duplicate from cycle generation.
KNOWN_DUPLICATES = {
    "CS2_35_2_4_11.parquet": "CS2_35_2_10_11.parquet",
    "CS2_38_2_4_11.parquet": "CS2_38_2_10_11.parquet",
}

DATA_COLUMNS = (
    "source_row",
    "test_time_s_raw",
    "test_time_s",
    "step_time_s",
    "timestamp",
    "step_id_raw",
    "voltage_v",
    "current_a",
    "temperature_c",
    "charge_capacity_ah",
    "discharge_capacity_ah",
    "capacity_raw_ah",
    "usable_for_model",
)


def now_iso() -> str:
    return datetime.now(CHINA_TZ).isoformat()


def py_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def first_non_null(df: pl.DataFrame, column: str) -> Any:
    if column not in df.columns:
        return None
    values = df[column].drop_nulls()
    return py_scalar(values[0]) if len(values) else None


def collapse(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if not result or result[-1] != value:
            result.append(value)
    return result


def parse_calce_filename_date(path: Path) -> str | None:
    match = re.search(r"_(\d{1,2})_(\d{1,2})[A-Za-z]?_(\d{2})$", path.stem)
    if not match:
        return None
    month, day, year = (int(part) for part in match.groups())
    try:
        return datetime(2000 + year, month, day).date().isoformat()
    except ValueError:
        return None


def source_time_bounds(df: pl.DataFrame) -> tuple[str | None, str | None]:
    if "timestamp" not in df.columns:
        return None, None
    timestamps = df["timestamp"].drop_nulls()
    if not len(timestamps):
        return None, None
    return str(timestamps.min()), str(timestamps.max())


def source_metadata(path: Path) -> dict[str, Any]:
    df = pl.read_parquet(path)
    timestamp_min, timestamp_max = source_time_bounds(df)
    source_format = first_non_null(df, "source_format")
    parsed_date = parse_calce_filename_date(path) if source_format == "txt" else None
    return {
        "path": path,
        "source_file": first_non_null(df, "source_file") or path.name,
        "source_sheet": first_non_null(df, "source_sheet"),
        "file_format": source_format,
        "row_count": df.height,
        "source_date": first_non_null(df, "source_date") or parsed_date,
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
        "time_segment_ids": sorted(
            int(x) for x in df["time_segment_id"].drop_nulls().unique().to_list()
        ),
    }


def source_sort_key(meta: dict[str, Any]) -> tuple[str, str]:
    date_key = meta.get("timestamp_min") or meta.get("source_date") or "9999-12-31"
    return str(date_key), str(meta["source_file"])


def verify_duplicate(duplicate: Path, original: Path) -> None:
    left = pl.read_parquet(duplicate)
    right = pl.read_parquet(original)
    excluded = {"source_file", "source_sheet", "source_date"}
    columns = [name for name in left.columns if name not in excluded]
    if columns != [name for name in right.columns if name not in excluded]:
        raise RuntimeError(f"Duplicate schema mismatch: {duplicate.name} vs {original.name}")
    if not left.select(columns).equals(right.select(columns), null_equal=True):
        raise RuntimeError(
            f"Known duplicate is no longer equal on core columns: "
            f"{duplicate.name} vs {original.name}"
        )


def assign_txt_cycle_ids(df: pl.DataFrame) -> np.ndarray:
    phase = np.asarray(df["phase"].to_list(), dtype=object)
    discharge_count = np.asarray(df["discharge_count"].to_list(), dtype=object)
    assigned = np.full(df.height, -1, dtype=np.int64)

    for idx, (phase_value, count_value) in enumerate(zip(phase, discharge_count)):
        if count_value is None:
            continue
        count = int(count_value)
        if phase_value == "charge":
            assigned[idx] = count + 1
        elif phase_value == "discharge":
            assigned[idx] = count

    # Rest/unknown rows within a cycle inherit the preceding active cycle.
    # Leading rows inherit the next active cycle. Between two different cycles,
    # the rest is assigned to the preceding cycle as post-discharge rest.
    last = -1
    for idx in range(len(assigned)):
        if assigned[idx] >= 0:
            last = int(assigned[idx])
        elif last >= 0:
            assigned[idx] = last
    next_value = -1
    for idx in range(len(assigned) - 1, -1, -1):
        if assigned[idx] >= 0:
            next_value = int(assigned[idx])
        elif next_value >= 0:
            assigned[idx] = next_value
    return assigned


def contiguous_cycle_slices(
    segment_ids: np.ndarray, cycle_ids: np.ndarray
) -> list[tuple[int, int, int, int, int]]:
    slices: list[tuple[int, int, int, int, int]] = []
    start: int | None = None
    current: tuple[int, int] | None = None
    occurrence: Counter[tuple[int, int]] = Counter()
    for idx, (segment_value, cycle_value) in enumerate(zip(segment_ids, cycle_ids)):
        if int(cycle_value) < 0:
            if start is not None and current is not None:
                occurrence[current] += 1
                slices.append((start, idx, current[0], current[1], occurrence[current]))
                start = None
                current = None
            continue
        key = (int(segment_value), int(cycle_value))
        if current is None:
            start, current = idx, key
        elif key != current:
            occurrence[current] += 1
            slices.append((int(start), idx, current[0], current[1], occurrence[current]))
            start, current = idx, key
    if start is not None and current is not None:
        occurrence[current] += 1
        slices.append((start, len(cycle_ids), current[0], current[1], occurrence[current]))
    return slices


def numeric_array(series: pl.Series, dtype: np.dtype | type) -> np.ndarray | None:
    if series.null_count() == len(series):
        return None
    values = series.cast(pl.Float64, strict=False).to_numpy()
    return np.asarray(values, dtype=dtype)


def timestamp_array(series: pl.Series) -> np.ndarray | None:
    if series.null_count() == len(series):
        return None
    values = [value if value is not None else "NaT" for value in series.to_list()]
    return np.asarray(values, dtype="datetime64[us]")


def build_data_arrays(df: pl.DataFrame) -> dict[str, Any]:
    test_time = numeric_array(df["test_time_s"], np.float64)
    if test_time is None:
        elapsed = None
    else:
        finite = np.flatnonzero(np.isfinite(test_time))
        elapsed = test_time.copy()
        if len(finite):
            elapsed = elapsed - test_time[finite[0]]

    result: dict[str, Any] = {
        "source_row": numeric_array(df["source_row"], np.int64),
        "test_time_s_raw": numeric_array(df["test_time_s_raw"], np.float64),
        "test_time_s": test_time,
        "elapsed_s": elapsed,
        "step_time_s": numeric_array(df["step_time_s"], np.float64),
        "timestamp": timestamp_array(df["timestamp"]),
        "step_id_raw": numeric_array(df["step_id_raw"], np.float64),
        "voltage_v": numeric_array(df["voltage_v"], np.float64),
        "current_a": numeric_array(df["current_a"], np.float64),
        "temperature_c": numeric_array(df["temperature_c"], np.float64),
        "charge_capacity_ah": numeric_array(df["charge_capacity_ah"], np.float64),
        "discharge_capacity_ah": numeric_array(
            df["discharge_capacity_ah"], np.float64
        ),
        "capacity_raw_ah": numeric_array(df["capacity_raw_ah"], np.float64),
        "usable_mask": np.asarray(
            [bool(value) if value is not None else False for value in df["usable_for_model"]],
            dtype=np.bool_,
        ),
    }
    return result


def quality_payload(df: pl.DataFrame) -> dict[str, Any]:
    flags = sorted(
        str(value)
        for value in df["quality_flag"].drop_nulls().unique().to_list()
        if value != "none"
    )
    events: list[dict[str, Any]] = []
    quality = df.select(
        "source_row", "quality_flag", "repair_action", "usable_for_model"
    ).to_dicts()
    for offset, item in enumerate(quality):
        if (
            item["quality_flag"] not in (None, "none")
            or item["repair_action"] not in (None, "none")
            or not bool(item["usable_for_model"])
        ):
            events.append(
                {
                    "offset": offset,
                    "source_row": int(item["source_row"]),
                    "quality_flag": item["quality_flag"],
                    "repair_action": item["repair_action"],
                    "usable_for_model": bool(item["usable_for_model"]),
                }
            )
    return {
        "flags": flags,
        "contains_invalid_rows": bool((~df["usable_for_model"]).any()),
        "events": events,
    }


def segment_payload(
    run_df: pl.DataFrame, segment_index: int, position: str | None = None
) -> dict[str, Any]:
    test_time = run_df["test_time_s"].drop_nulls()
    start_time = py_scalar(test_time[0]) if len(test_time) else None
    end_time = py_scalar(test_time[-1]) if len(test_time) else None
    duration = (
        float(end_time - start_time)
        if start_time is not None and end_time is not None
        else None
    )
    payload = {
        "segment_index": segment_index,
        "boundary": {
            "source_row_start": int(run_df["source_row"][0]),
            "source_row_end": int(run_df["source_row"][-1]),
            "start_time_s": start_time,
            "end_time_s": end_time,
            "duration_s": duration,
            "n_points": run_df.height,
        },
        "data": build_data_arrays(run_df),
        "quality": quality_payload(run_df),
    }
    if position is not None:
        payload["position"] = position
    return payload


def run_slices(phases: list[str]) -> list[tuple[int, int, str]]:
    if not phases:
        return []
    result: list[tuple[int, int, str]] = []
    start = 0
    current = phases[0]
    for idx in range(1, len(phases)):
        if phases[idx] != current:
            result.append((start, idx, current))
            start, current = idx, phases[idx]
    result.append((start, len(phases), current))
    return result


def rest_position(runs: list[tuple[int, int, str]], run_index: int) -> str:
    previous_active = None
    next_active = None
    for idx in range(run_index - 1, -1, -1):
        if runs[idx][2] in ("charge", "discharge"):
            previous_active = runs[idx][2]
            break
    for idx in range(run_index + 1, len(runs)):
        if runs[idx][2] in ("charge", "discharge"):
            next_active = runs[idx][2]
            break
    if previous_active is None and next_active == "charge":
        return "pre_charge"
    if previous_active is None and next_active == "discharge":
        return "pre_discharge"
    if previous_active == "charge" and next_active == "discharge":
        return "between_charge_discharge"
    if previous_active == "discharge" and next_active == "charge":
        return "between_discharge_charge"
    if previous_active == next_active == "charge":
        return "charge_interruption"
    if previous_active == next_active == "discharge":
        return "discharge_interruption"
    if previous_active == "discharge" and next_active is None:
        return "post_discharge"
    if previous_active == "charge" and next_active is None:
        return "post_charge"
    return "unassigned"


def phase_capacity(df: pl.DataFrame, phase: str) -> tuple[float | None, str | None]:
    selected = df.filter((pl.col("phase") == phase) & pl.col("usable_for_model"))
    if not selected.height:
        return None, None
    if phase == "charge" and selected["charge_capacity_ah"].null_count() < selected.height:
        value = selected["charge_capacity_ah"].max()
        return (float(value), "charge_capacity_ah:max") if value is not None else (None, None)
    if phase == "discharge" and selected["discharge_capacity_ah"].null_count() < selected.height:
        value = selected["discharge_capacity_ah"].max()
        return (float(value), "discharge_capacity_ah:max") if value is not None else (None, None)
    if selected["capacity_raw_ah"].null_count() < selected.height:
        value = selected["capacity_raw_ah"].max()
        return (float(value), f"capacity_raw_ah:{phase}_max") if value is not None else (None, None)
    return None, None


def cycle_clock(df: pl.DataFrame, source_id: str, time_segment_id: int) -> dict[str, Any]:
    timestamps = timestamp_array(df["timestamp"])
    if timestamps is not None:
        valid = timestamps[~np.isnat(timestamps)]
        if len(valid):
            return {
                "domain": "absolute_timestamp",
                "key": "absolute_timestamp",
                "start": int(valid[0].astype("datetime64[us]").astype(np.int64)),
                "end": int(valid[-1].astype("datetime64[us]").astype(np.int64)),
                "divisor": 1_000_000.0,
            }
    times = numeric_array(df["test_time_s"], np.float64)
    if times is not None:
        valid = times[np.isfinite(times)]
        if len(valid):
            return {
                "domain": "test_time_s",
                "key": f"{source_id}:{time_segment_id}",
                "start": float(valid[0]),
                "end": float(valid[-1]),
                "divisor": 1.0,
            }
    return {"domain": None, "key": None, "start": None, "end": None, "divisor": 1.0}


def build_cycle(
    cycle_df: pl.DataFrame,
    *,
    cell_id: str,
    observed_cycle_index: int,
    source_id: str,
    source_file: str,
    source_sheet: str | None,
    time_segment_id: int,
    source_cycle_id: int,
    occurrence: int,
    source_first_row: int,
    source_last_row: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    phases = [str(value) if value is not None else "unknown" for value in cycle_df["phase"]]
    runs = run_slices(phases)
    active_phase_order = collapse(value for value in phases if value in ("charge", "discharge"))
    complete_pair = bool(
        active_phase_order
        and active_phase_order[0] == "charge"
        and active_phase_order[-1] == "discharge"
        and "charge" in active_phase_order
        and "discharge" in active_phase_order
    )

    charge_segments: list[dict[str, Any]] = []
    discharge_segments: list[dict[str, Any]] = []
    rest_segments: list[dict[str, Any]] = []
    unknown_segments: list[dict[str, Any]] = []
    for run_idx, (start, end, phase) in enumerate(runs):
        run_df = cycle_df.slice(start, end - start)
        if phase == "charge":
            charge_segments.append(segment_payload(run_df, len(charge_segments)))
        elif phase == "discharge":
            discharge_segments.append(segment_payload(run_df, len(discharge_segments)))
        elif phase == "rest":
            rest_segments.append(
                segment_payload(run_df, len(rest_segments), rest_position(runs, run_idx))
            )
        else:
            unknown_segments.append(
                segment_payload(run_df, len(unknown_segments), rest_position(runs, run_idx))
            )

    charge_capacity, charge_source = phase_capacity(cycle_df, "charge")
    discharge_capacity, discharge_source = phase_capacity(cycle_df, "discharge")
    usable_charge = bool(
        cycle_df.filter((pl.col("phase") == "charge") & pl.col("usable_for_model")).height
    )
    usable_discharge = bool(
        cycle_df.filter((pl.col("phase") == "discharge") & pl.col("usable_for_model")).height
    )
    flags = sorted(
        str(value)
        for value in cycle_df["quality_flag"].drop_nulls().unique().to_list()
        if value != "none"
    )
    if not complete_pair:
        flags.append("incomplete_charge_discharge_pair")
    if len(active_phase_order) > 2:
        flags.append("interleaved_active_phases")
    flags = sorted(set(flags))

    uid = (
        f"{cell_id}:{source_id}:segment_{time_segment_id}:"
        f"raw_{source_cycle_id}:occurrence_{occurrence}"
    )
    cycle = {
        "observed_cycle_index": observed_cycle_index,
        "cycle_uid": uid,
        "raw_cycle_identity": {
            "time_segment_id": time_segment_id,
            "source_cycle_id": source_cycle_id,
            "occurrence": occurrence,
            "charge_count_min": first_non_null(cycle_df, "charge_count"),
            "discharge_count_min": first_non_null(cycle_df, "discharge_count"),
        },
        "source": {
            "source_id": source_id,
            "source_file": source_file,
            "source_sheet": source_sheet,
            "source_row_start": int(cycle_df["source_row"][0]),
            "source_row_end": int(cycle_df["source_row"][-1]),
        },
        "boundary": {
            "phase_order": collapse(phases),
            "active_phase_order": active_phase_order,
            "is_complete_pair": complete_pair,
            "starts_at_source_boundary": int(cycle_df["source_row"][0]) == source_first_row,
            "ends_at_source_boundary": int(cycle_df["source_row"][-1]) == source_last_row,
            "gap_before_s": None,
            "gap_after_s": None,
        },
        "charge": {"n_segments": len(charge_segments), "segments": charge_segments},
        "discharge": {
            "n_segments": len(discharge_segments),
            "segments": discharge_segments,
        },
        "rest_segments": rest_segments,
        "unknown_segments": unknown_segments,
        "targets": {
            "charge_capacity_ah": charge_capacity,
            "charge_capacity_source": charge_source,
            "discharge_capacity_ah": discharge_capacity,
            "discharge_capacity_source": discharge_source,
            "capacity_label_ah": discharge_capacity,
            "capacity_label_source": discharge_source,
        },
        "quality": {
            "usable_for_charge_model": usable_charge,
            "usable_for_discharge_model": usable_discharge,
            "usable_for_charge_discharge_pair": bool(
                complete_pair and usable_charge and usable_discharge
            ),
            "flags": flags,
            "notes": None,
        },
    }
    return cycle, cycle_clock(cycle_df, source_id, time_segment_id)


def missing_ranges(values: list[int]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    unique = sorted(set(values))
    for left, right in zip(unique, unique[1:]):
        if right - left > 1:
            result.append((left + 1, right - 1))
    return result


def atomic_pickle(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_cell_file(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        cell = pickle.load(handle)
    cycles = cell["cycles"]
    expected_indices = list(range(1, len(cycles) + 1))
    actual_indices = [cycle["observed_cycle_index"] for cycle in cycles.values()]
    if actual_indices != expected_indices:
        raise RuntimeError(f"Observed cycle indices are not continuous in {path.name}")
    segment_count = 0
    point_count = 0
    for cycle in cycles.values():
        for container in (cycle["charge"], cycle["discharge"]):
            segments = container["segments"]
            if container["n_segments"] != len(segments):
                raise RuntimeError(f"Segment count mismatch in {path.name}")
            for segment in segments:
                segment_count += 1
                expected = segment["boundary"]["n_points"]
                point_count += expected
                for value in segment["data"].values():
                    if isinstance(value, np.ndarray) and len(value) != expected:
                        raise RuntimeError(f"Array length mismatch in {path.name}")
        for group_name in ("rest_segments", "unknown_segments"):
            for segment in cycle[group_name]:
                segment_count += 1
                expected = segment["boundary"]["n_points"]
                point_count += expected
                for value in segment["data"].values():
                    if isinstance(value, np.ndarray) and len(value) != expected:
                        raise RuntimeError(f"Array length mismatch in {path.name}")
    return {
        "cell_id": cell["cell_id"],
        "reload_ok": True,
        "cycle_count": len(cycles),
        "segment_count": segment_count,
        "point_count": point_count,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_cell(
    dataset: str,
    cell_id: str,
    paths: list[Path],
    output_cells: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    metas = [source_metadata(path) for path in paths]
    metas.sort(key=source_sort_key)
    by_name = {meta["path"].name: meta for meta in metas}

    for duplicate_name, original_name in KNOWN_DUPLICATES.items():
        if duplicate_name in by_name:
            if original_name not in by_name:
                raise RuntimeError(f"Missing original for duplicate {duplicate_name}")
            verify_duplicate(by_name[duplicate_name]["path"], by_name[original_name]["path"])

    registry: list[dict[str, Any]] = []
    active_order = 0
    for registry_order, meta in enumerate(metas):
        duplicate_of_name = KNOWN_DUPLICATES.get(meta["path"].name)
        included = duplicate_of_name is None
        meta["source_id"] = f"source_{registry_order:04d}"
        meta["source_order"] = registry_order
        meta["active_source_order"] = active_order if included else None
        meta["is_duplicate"] = not included
        meta["duplicate_of_name"] = duplicate_of_name
        meta["included_in_cycles"] = included
        if included:
            active_order += 1
    name_to_source_id = {meta["path"].name: meta["source_id"] for meta in metas}
    for meta in metas:
        meta["duplicate_of"] = (
            name_to_source_id.get(meta["duplicate_of_name"])
            if meta["duplicate_of_name"]
            else None
        )

    cycles: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    unassigned_rows: list[dict[str, Any]] = []
    raw_cycle_reappearances: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    previous_cycle: dict[str, Any] | None = None
    previous_clock: dict[str, Any] | None = None
    observed_index = 0

    for meta in metas:
        registry_item = {key: value for key, value in meta.items() if key != "path"}
        registry.append(registry_item)
        if not meta["included_in_cycles"]:
            continue
        df = pl.read_parquet(meta["path"])
        if metadata is None:
            metadata = {
                "chemistry": first_non_null(df, "chemistry"),
                "nominal_capacity_ah": None,
                "temperature_setpoint_c": first_non_null(df, "temperature_setpoint_c"),
                "charge_rate_c": first_non_null(df, "charge_rate_c"),
                "discharge_rate_c": first_non_null(df, "discharge_rate_c"),
                "condition_group": first_non_null(df, "condition_group"),
            }

        source_format = meta["file_format"]
        if dataset == "CALCE" and source_format == "txt":
            cycle_ids = assign_txt_cycle_ids(df)
        else:
            raw = df["cycle_id_raw"].to_numpy()
            cycle_ids = np.asarray(
                [int(value) if value is not None else -1 for value in raw], dtype=np.int64
            )
        segment_ids = np.asarray(
            [int(value) if value is not None else 0 for value in df["time_segment_id"]],
            dtype=np.int64,
        )
        source_slices = contiguous_cycle_slices(segment_ids, cycle_ids)
        assigned_mask = cycle_ids >= 0
        if not bool(assigned_mask.all()):
            bad_rows = df.filter(pl.Series("assigned", ~assigned_mask))["source_row"].to_list()
            unassigned_rows.append(
                {
                    "source_id": meta["source_id"],
                    "source_file": meta["source_file"],
                    "count": len(bad_rows),
                    "source_row_samples": [int(value) for value in bad_rows[:20]],
                }
            )

        local_by_segment: dict[int, list[int]] = {}
        seen_keys: Counter[tuple[int, int]] = Counter()
        for start, end, time_segment_id, source_cycle_id, occurrence in source_slices:
            local_by_segment.setdefault(time_segment_id, []).append(source_cycle_id)
            seen_keys[(time_segment_id, source_cycle_id)] += 1
            if occurrence > 1:
                raw_cycle_reappearances.append(
                    {
                        "source_id": meta["source_id"],
                        "source_file": meta["source_file"],
                        "time_segment_id": time_segment_id,
                        "source_cycle_id": source_cycle_id,
                        "occurrence": occurrence,
                    }
                )
            cycle_df = df.slice(start, end - start)
            observed_index += 1
            cycle, clock = build_cycle(
                cycle_df,
                cell_id=cell_id,
                observed_cycle_index=observed_index,
                source_id=meta["source_id"],
                source_file=meta["source_file"],
                source_sheet=meta["source_sheet"],
                time_segment_id=time_segment_id,
                source_cycle_id=source_cycle_id,
                occurrence=occurrence,
                source_first_row=int(df["source_row"][0]),
                source_last_row=int(df["source_row"][-1]),
            )
            if (
                previous_clock is not None
                and previous_clock["key"] is not None
                and previous_clock["key"] == clock["key"]
                and previous_clock["end"] is not None
                and clock["start"] is not None
            ):
                gap = (clock["start"] - previous_clock["end"]) / clock["divisor"]
                cycle["boundary"]["gap_before_s"] = float(gap)
                if previous_cycle is not None:
                    previous_cycle["boundary"]["gap_after_s"] = float(gap)
            key = f"cycle_{observed_index:06d}"
            cycles[key] = cycle
            previous_cycle = cycle
            previous_clock = clock

        for time_segment_id, values in local_by_segment.items():
            for start_missing, end_missing in missing_ranges(values):
                missing.append(
                    {
                        "source_id": meta["source_id"],
                        "source_file": meta["source_file"],
                        "time_segment_id": time_segment_id,
                        "missing_from": start_missing,
                        "missing_to": end_missing,
                    }
                )

    pattern_counts = Counter()
    complete_count = 0
    incomplete_uids: list[str] = []
    interleaved_uids: list[str] = []
    total_points = 0
    for cycle in cycles.values():
        active = cycle["boundary"]["active_phase_order"]
        pattern = "->".join(active) if active else "no_active_phase"
        pattern_counts[pattern] += 1
        if cycle["boundary"]["is_complete_pair"]:
            complete_count += 1
        else:
            incomplete_uids.append(cycle["cycle_uid"])
        if len(active) > 2:
            interleaved_uids.append(cycle["cycle_uid"])
        for container in (cycle["charge"], cycle["discharge"]):
            total_points += sum(seg["boundary"]["n_points"] for seg in container["segments"])
        total_points += sum(seg["boundary"]["n_points"] for seg in cycle["rest_segments"])
        total_points += sum(seg["boundary"]["n_points"] for seg in cycle["unknown_segments"])

    cell_dict = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "cell_id": cell_id,
        "dataset": dataset,
        "metadata": metadata or {},
        "source_registry": registry,
        "continuity": {
            "ordering_basis": (
                "absolute timestamp then source file"
                if dataset == "CALCE" and cell_id not in ("CS2_8", "CS2_21")
                else "source date/file order; observed order is not absolute physical cycle number"
            ),
            "observed_cycle_count": len(cycles),
            "complete_pair_count": complete_count,
            "incomplete_cycle_count": len(cycles) - complete_count,
            "interleaved_cycle_count": len(interleaved_uids),
            "incomplete_cycle_uids": incomplete_uids,
            "interleaved_cycle_uids": interleaved_uids,
            "missing_source_cycle_ranges": missing,
            "duplicate_sources_excluded": [
                item for item in registry if item["is_duplicate"]
            ],
            "unassigned_rows": unassigned_rows,
            "raw_cycle_reappearances": raw_cycle_reappearances,
        },
        "cycles": cycles,
    }
    output_path = output_cells / f"{cell_id}.pkl"
    atomic_pickle(output_path, cell_dict)
    validation = validate_cell_file(output_path)
    summary = {
        "dataset": dataset,
        "cell_id": cell_id,
        "condition_group": (metadata or {}).get("condition_group"),
        "observed_cycles": len(cycles),
        "complete_pairs": complete_count,
        "incomplete_cycles": len(cycles) - complete_count,
        "interleaved_cycles": len(interleaved_uids),
        "source_files": len(registry),
        "included_source_files": sum(int(item["included_in_cycles"]) for item in registry),
        "excluded_duplicate_files": sum(int(item["is_duplicate"]) for item in registry),
        "missing_source_cycle_ranges": len(missing),
        "unassigned_rows": sum(int(item["count"]) for item in unassigned_rows),
        "rows_in_cycles": total_points,
        "phase_pattern_counts": dict(sorted(pattern_counts.items())),
        "cell_file": str(Path("cells") / output_path.name),
        "cell_file_size_bytes": output_path.stat().st_size,
        "cell_file_sha256": validation["sha256"],
        "reload_validation": True,
    }
    print(
        f"[{dataset}] {cell_id}: cycles={len(cycles)}, complete={complete_count}, "
        f"interleaved={len(interleaved_uids)}, rows={total_points}, "
        f"file={output_path.stat().st_size / (1024**2):.1f} MiB",
        flush=True,
    )
    return summary, registry, validation


def flatten_sources(dataset: str, cell_id: str, registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in registry:
        rows.append(
            {
                "dataset": dataset,
                "cell_id": cell_id,
                "source_id": item["source_id"],
                "source_file": item["source_file"],
                "source_order": item["source_order"],
                "active_source_order": item["active_source_order"],
                "file_format": item["file_format"],
                "row_count": item["row_count"],
                "source_date": item["source_date"],
                "timestamp_min": item["timestamp_min"],
                "timestamp_max": item["timestamp_max"],
                "time_segment_ids": json.dumps(item["time_segment_ids"]),
                "included_in_cycles": item["included_in_cycles"],
                "is_duplicate": item["is_duplicate"],
                "duplicate_of": item["duplicate_of"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            serializable = {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            writer.writerow(serializable)


def dataset_paths(dataset: str) -> dict[str, list[Path]]:
    if dataset == "CALCE":
        return {
            cell: sorted((CALCE_INPUT / cell).glob("*.parquet"))
            for cell in CALCE_CELLS
        }
    result: dict[str, list[Path]] = {}
    for group in TJU_GROUPS:
        for path in sorted((TJU_INPUT / group).glob("*.parquet")):
            result[path.stem] = [path]
    return result


def build_dataset(dataset: str, output_root: Path) -> dict[str, Any]:
    dataset_output = output_root / dataset
    cells_output = dataset_output / "cells"
    cells_output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    cells_index: dict[str, Any] = {}

    for cell_id, paths in dataset_paths(dataset).items():
        if not paths:
            raise FileNotFoundError(f"No standardized Parquet files found for {cell_id}")
        summary, registry, validation = build_cell(dataset, cell_id, paths, cells_output)
        summaries.append(summary)
        source_rows.extend(flatten_sources(dataset, cell_id, registry))
        validations.append(validation)
        cells_index[cell_id] = {
            "file": summary["cell_file"],
            "summary": {key: value for key, value in summary.items() if key != "cell_file"},
        }

    index = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "dataset": dataset,
        "storage_mode": "per-cell pickle dictionaries with a lightweight dataset index",
        "current_threshold_a": CURRENT_THRESHOLD_A,
        "input_root": str(CALCE_INPUT if dataset == "CALCE" else TJU_INPUT),
        "cells": cells_index,
        "rules": {
            "incomplete_cycles": "retained and marked unusable_for_charge_discharge_pair",
            "duplicates": "retained in source_registry but excluded from cycles after equality check",
            "multi_segment_cycles": "all contiguous charge/discharge segments retained",
            "rest": "stored separately and assigned a positional label",
            "sequence_length": "raw variable length; no interpolation, padding, truncation, or resampling",
            "cycle_number": "observed_cycle_index is sequential observed order, not an absolute physical cycle claim",
        },
    }
    index_name = "calce_cycle_dict.pkl" if dataset == "CALCE" else "tju_cycle_dict.pkl"
    atomic_pickle(dataset_output / index_name, index)
    (dataset_output / "dataset_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(dataset_output / "cycle_summary.csv", summaries)
    write_csv(dataset_output / "source_registry.csv", source_rows)
    validation_report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "dataset": dataset,
        "status": "VERIFIED",
        "checks": [
            "each cell pickle can be reloaded",
            "observed_cycle_index is consecutive from 1",
            "all stored arrays match their segment n_points",
            "known duplicate CALCE files match their originals on core columns",
            "SHA-256 recorded for every cell pickle",
        ],
        "cells": validations,
        "totals": {
            "cells": len(summaries),
            "cycles": sum(int(row["observed_cycles"]) for row in summaries),
            "complete_pairs": sum(int(row["complete_pairs"]) for row in summaries),
            "incomplete_cycles": sum(int(row["incomplete_cycles"]) for row in summaries),
            "interleaved_cycles": sum(int(row["interleaved_cycles"]) for row in summaries),
            "rows_in_cycles": sum(int(row["rows_in_cycles"]) for row in summaries),
        },
    }
    (dataset_output / "validation_report.json").write_text(
        json.dumps(validation_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checksum_lines = []
    for path in sorted(dataset_output.rglob("*.pkl")):
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(dataset_output).as_posix()}")
    (dataset_output / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(
        f"[{dataset}] complete: cells={len(summaries)}, "
        f"cycles={validation_report['totals']['cycles']}, "
        f"rows={validation_report['totals']['rows_in_cycles']}",
        flush=True,
    )
    return validation_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build per-cell charge/discharge cycle dictionaries from initial_v2 Parquet files."
    )
    parser.add_argument("--dataset", choices=("all", "calce", "tju"), default="all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/cycle_dict_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    selected = ["CALCE", "TJU"] if args.dataset == "all" else [args.dataset.upper()]
    for dataset in selected:
        target = output_root / dataset
        if target.exists() and any(target.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {target}. Use --overwrite only after review."
            )
    output_root.mkdir(parents=True, exist_ok=True)
    reports = [build_dataset(dataset, output_root) for dataset in selected]
    combined = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "python": sys.version,
        "python_executable": sys.executable,
        "polars": pl.__version__,
        "numpy": np.__version__,
        "reports": reports,
    }
    (output_root / "processing_report.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(combined, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
