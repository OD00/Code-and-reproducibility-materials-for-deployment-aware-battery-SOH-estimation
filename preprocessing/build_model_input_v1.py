from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "model_input_v1"
DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data/cycle_dict_qc_v2"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/model_input_v1"
INDEX_NAMES = {"CALCE": "calce_cycle_dict.pkl", "TJU": "tju_cycle_dict.pkl"}
NOMINAL_CAPACITY_AH = {"CALCE": 1.1, "TJU": 3.5}
DEFAULT_POINTS = 256
DEFAULT_WINDOW = 12


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def unicode_array(values: list[str]) -> np.ndarray:
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{width}")


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def trapz(values: np.ndarray, time_s: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, time_s))
    return float(np.trapz(values, time_s))


def phase_arrays(cycle: dict[str, Any], phase: str) -> dict[str, Any]:
    """Concatenate active segments on a cumulative active-time axis.

    Rest gaps are not converted into artificial current/voltage samples. Their
    effect remains auditable through wall_duration_s and segment_count.
    """

    all_time: list[np.ndarray] = []
    all_voltage: list[np.ndarray] = []
    all_current: list[np.ndarray] = []
    test_starts: list[float] = []
    test_ends: list[float] = []
    active_offset = 0.0
    raw_point_count = 0

    for segment in cycle[phase]["segments"]:
        data = segment["data"]
        elapsed = np.asarray(data["elapsed_s"], dtype=np.float64)
        voltage = np.asarray(data["voltage_v"], dtype=np.float64)
        current = np.asarray(data["current_a"], dtype=np.float64)
        usable = data.get("usable_mask")
        mask = np.isfinite(elapsed) & np.isfinite(voltage) & np.isfinite(current)
        if usable is not None:
            mask &= np.asarray(usable, dtype=bool)

        elapsed = elapsed[mask]
        voltage = voltage[mask]
        current = current[mask]
        raw_point_count += int(elapsed.size)
        if elapsed.size < 2:
            continue

        order = np.argsort(elapsed, kind="stable")
        elapsed = elapsed[order]
        voltage = voltage[order]
        current = current[order]
        elapsed = elapsed - elapsed[0] + active_offset
        if all_time and elapsed[0] <= all_time[-1][-1]:
            elapsed += all_time[-1][-1] - elapsed[0] + 1e-6

        all_time.append(elapsed)
        all_voltage.append(voltage)
        all_current.append(current)
        active_offset = float(elapsed[-1])

        boundary = segment["boundary"]
        start = boundary.get("start_time_s")
        end = boundary.get("end_time_s")
        if start is not None and np.isfinite(start):
            test_starts.append(float(start))
        if end is not None and np.isfinite(end):
            test_ends.append(float(end))

    if not all_time:
        raise ValueError(f"{phase} has fewer than two usable points")

    time_s = np.concatenate(all_time)
    voltage_v = np.concatenate(all_voltage)
    current_a = np.concatenate(all_current)

    unique_time, inverse = np.unique(time_s, return_inverse=True)
    duplicate_count = int(time_s.size - unique_time.size)
    if duplicate_count:
        voltage_sum = np.zeros(unique_time.size, dtype=np.float64)
        current_sum = np.zeros(unique_time.size, dtype=np.float64)
        counts = np.zeros(unique_time.size, dtype=np.float64)
        np.add.at(voltage_sum, inverse, voltage_v)
        np.add.at(current_sum, inverse, current_a)
        np.add.at(counts, inverse, 1.0)
        time_s = unique_time
        voltage_v = voltage_sum / counts
        current_a = current_sum / counts

    active_duration_s = float(time_s[-1] - time_s[0])
    if active_duration_s <= 0:
        raise ValueError(f"{phase} active duration is not positive")
    time_s = time_s - time_s[0]
    wall_duration_s = (
        float(max(test_ends) - min(test_starts))
        if test_starts and test_ends
        else active_duration_s
    )
    return {
        "time_s": time_s,
        "voltage_v": voltage_v,
        "current_a": current_a,
        "active_duration_s": active_duration_s,
        "wall_duration_s": wall_duration_s,
        "point_count": int(time_s.size),
        "raw_point_count": raw_point_count,
        "duplicate_time_count": duplicate_count,
        "segment_count": len(cycle[phase]["segments"]),
    }


def resample_phase(values: dict[str, Any], points: int) -> dict[str, Any]:
    time_s = values["time_s"]
    grid_s = np.linspace(0.0, values["active_duration_s"], points, dtype=np.float64)
    voltage = np.interp(grid_s, time_s, values["voltage_v"])
    current = np.interp(grid_s, time_s, values["current_a"])

    voltage_back = np.interp(time_s, grid_s, voltage)
    current_back = np.interp(time_s, grid_s, current)
    original_integral = abs(trapz(values["current_a"], time_s))
    resampled_integral = abs(trapz(current, grid_s))
    integral_error = abs(resampled_integral - original_integral) / max(
        original_integral, 1e-12
    )
    return {
        **values,
        "grid_s": grid_s.astype(np.float32),
        "voltage_resampled_v": voltage.astype(np.float32),
        "current_resampled_a": current.astype(np.float32),
        "voltage_mae_v": float(np.mean(np.abs(voltage_back - values["voltage_v"]))),
        "current_mae_a": float(np.mean(np.abs(current_back - values["current_a"]))),
        "voltage_max_abs_error_v": float(
            np.max(np.abs(voltage_back - values["voltage_v"]))
        ),
        "current_max_abs_error_a": float(
            np.max(np.abs(current_back - values["current_a"]))
        ),
        "current_integral_relative_error": float(integral_error),
    }


def consecutive_windows(observed_indices: np.ndarray, window: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for end in range(window - 1, len(observed_indices)):
        start = end - window + 1
        candidate = observed_indices[start : end + 1]
        if np.all(np.diff(candidate) == 1):
            rows.append(np.arange(start, end + 1, dtype=np.int32))
    if not rows:
        return np.empty((0, window), dtype=np.int32)
    return np.stack(rows, axis=0)


def write_cell_npz(
    destination: Path,
    cell: dict[str, Any],
    dataset: str,
    points: int,
    window: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    nominal = float(NOMINAL_CAPACITY_AH[dataset])
    cycle_keys: list[str] = []
    cycle_uids: list[str] = []
    observed_indices: list[int] = []
    quality_flags: list[str] = []
    capacity: list[float] = []
    phase_rows: dict[str, dict[str, list[Any]]] = {
        phase: {
            "voltage": [],
            "current": [],
            "time": [],
            "duration": [],
            "wall_duration": [],
            "point_count": [],
            "raw_point_count": [],
            "segment_count": [],
            "duplicate_time_count": [],
        }
        for phase in ("charge", "discharge")
    }
    index_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    excluded = 0

    ordered_cycles = sorted(
        cell["cycles"].items(),
        key=lambda item: int(item[1]["observed_cycle_index"]),
    )
    for cycle_key, cycle in ordered_cycles:
        if not cycle["quality"].get("valid_for_primary_training", False):
            excluded += 1
            continue
        label = cycle["targets"].get("capacity_label_qc_ah")
        if label is None or not np.isfinite(label):
            excluded += 1
            continue
        try:
            resampled = {
                phase: resample_phase(phase_arrays(cycle, phase), points)
                for phase in ("charge", "discharge")
            }
        except ValueError:
            excluded += 1
            continue

        row_index = len(cycle_keys)
        cycle_keys.append(cycle_key)
        cycle_uids.append(str(cycle["cycle_uid"]))
        observed_indices.append(int(cycle["observed_cycle_index"]))
        flags = sorted(str(value) for value in cycle["quality"].get("flags", []))
        quality_flags.append(json.dumps(flags, ensure_ascii=False))
        capacity.append(float(label))

        index_row = {
            "dataset": dataset,
            "cell_id": cell["cell_id"],
            "row_index": row_index,
            "cycle_key": cycle_key,
            "cycle_uid": cycle["cycle_uid"],
            "observed_cycle_index": int(cycle["observed_cycle_index"]),
            "capacity_ah": float(label),
            "soh_nominal": float(label) / nominal,
            "nominal_capacity_ah": nominal,
            "condition_group": cell["metadata"].get("condition_group"),
            "temperature_setpoint_c": cell["metadata"].get(
                "temperature_setpoint_c"
            ),
            "charge_rate_c": cell["metadata"].get("charge_rate_c"),
            "discharge_rate_c": cell["metadata"].get("discharge_rate_c"),
            "quality_flags_json": quality_flags[-1],
        }
        for phase in ("charge", "discharge"):
            item = resampled[phase]
            phase_rows[phase]["voltage"].append(item["voltage_resampled_v"])
            phase_rows[phase]["current"].append(item["current_resampled_a"])
            phase_rows[phase]["time"].append(item["grid_s"])
            phase_rows[phase]["duration"].append(item["active_duration_s"])
            phase_rows[phase]["wall_duration"].append(item["wall_duration_s"])
            phase_rows[phase]["point_count"].append(item["point_count"])
            phase_rows[phase]["raw_point_count"].append(item["raw_point_count"])
            phase_rows[phase]["segment_count"].append(item["segment_count"])
            phase_rows[phase]["duplicate_time_count"].append(
                item["duplicate_time_count"]
            )
            index_row[f"{phase}_duration_s"] = item["active_duration_s"]
            index_row[f"{phase}_wall_duration_s"] = item["wall_duration_s"]
            index_row[f"{phase}_original_points"] = item["point_count"]
            quality_rows.append(
                {
                    "dataset": dataset,
                    "cell_id": cell["cell_id"],
                    "row_index": row_index,
                    "cycle_uid": cycle["cycle_uid"],
                    "phase": phase,
                    "original_points": item["point_count"],
                    "resampled_points": points,
                    "voltage_mae_mv": item["voltage_mae_v"] * 1000.0,
                    "current_mae_ma": item["current_mae_a"] * 1000.0,
                    "voltage_max_abs_error_mv": item["voltage_max_abs_error_v"]
                    * 1000.0,
                    "current_max_abs_error_ma": item["current_max_abs_error_a"]
                    * 1000.0,
                    "current_integral_relative_error_pct": item[
                        "current_integral_relative_error"
                    ]
                    * 100.0,
                }
            )
        index_rows.append(index_row)

    if not cycle_keys:
        raise RuntimeError(f"{dataset}/{cell['cell_id']} has no usable cycles")

    observed = np.asarray(observed_indices, dtype=np.int32)
    capacity_array = np.asarray(capacity, dtype=np.float32).reshape(-1, 1)
    soh_array = (capacity_array / nominal).astype(np.float32)
    window_rows = consecutive_windows(observed, window)
    target_rows = (
        window_rows[:, -1]
        if len(window_rows)
        else np.empty(0, dtype=np.int32)
    )
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "dataset": np.asarray(dataset),
        "cell_id": np.asarray(cell["cell_id"]),
        "condition_group": np.asarray(
            cell["metadata"].get("condition_group") or ""
        ),
        "nominal_capacity_ah": np.asarray(nominal, dtype=np.float32),
        "interpolation_points": np.asarray(points, dtype=np.int32),
        "window_size": np.asarray(window, dtype=np.int32),
        "cycle_key": unicode_array(cycle_keys),
        "cycle_uid": unicode_array(cycle_uids),
        "quality_flags_json": unicode_array(quality_flags),
        "observed_cycle_index": observed,
        "capacity_ah": capacity_array,
        "label_soh_nominal": soh_array,
        "window_row_indices": window_rows,
        "window_target_row_index": target_rows,
        "window_target_capacity_ah": capacity_array[target_rows]
        if len(target_rows)
        else np.empty((0, 1), dtype=np.float32),
        "window_target_soh_nominal": soh_array[target_rows]
        if len(target_rows)
        else np.empty((0, 1), dtype=np.float32),
        "window_target_cycle_uid": unicode_array(
            [cycle_uids[index] for index in target_rows]
        ),
    }
    for phase in ("charge", "discharge"):
        rows = phase_rows[phase]
        arrays[f"{phase}_voltage_v"] = np.stack(rows["voltage"]).astype(np.float32)
        arrays[f"{phase}_current_a"] = np.stack(rows["current"]).astype(np.float32)
        arrays[f"{phase}_elapsed_time_s"] = np.stack(rows["time"]).astype(
            np.float32
        )
        arrays[f"{phase}_duration_s"] = np.asarray(
            rows["duration"], dtype=np.float32
        ).reshape(-1, 1)
        arrays[f"{phase}_wall_duration_s"] = np.asarray(
            rows["wall_duration"], dtype=np.float32
        ).reshape(-1, 1)
        for field in (
            "point_count",
            "raw_point_count",
            "segment_count",
            "duplicate_time_count",
        ):
            arrays[f"{phase}_{field}"] = np.asarray(rows[field], dtype=np.int32)

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)

    summary = {
        "cell_id": cell["cell_id"],
        "dataset": dataset,
        "file": destination.name,
        "valid_cycles": len(cycle_keys),
        "excluded_cycles": excluded,
        "valid_12_cycle_windows": int(len(window_rows)),
        "observed_cycle_min": int(observed.min()),
        "observed_cycle_max": int(observed.max()),
        "capacity_ah_min": float(capacity_array.min()),
        "capacity_ah_max": float(capacity_array.max()),
        "soh_nominal_min": float(soh_array.min()),
        "soh_nominal_max": float(soh_array.max()),
        "sha256": sha256_file(destination),
        "metadata": cell["metadata"],
    }
    return summary, index_rows, quality_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def build_readme(
    dataset_summaries: dict[str, Any], quality_rows: list[dict[str, Any]], points: int
) -> str:
    lines = [
        "# 神经网络输入数据集说明（model_input_v1）",
        "",
        "## 数据边界",
        "",
        "- 来源：`cycle_dict_qc_v2` 中 `valid_for_primary_training=True` 的循环。",
        "- 插值：按真实活动阶段累计时间进行线性插值，不按原始点序号插值。",
        f"- 序列长度：每个充电/放电阶段 `{points}` 点。",
        "- 不执行信号平滑、离群点替换或容量曲线单调化。",
        "- 不执行全数据归一化；归一化器必须在后续实验中只用训练电芯拟合。",
        "- 主标签：`label_soh_nominal = capacity_ah / nominal_capacity_ah`，不裁剪到 `[0,1]`。",
        "- 附加标签：`capacity_ah`，来自放电电流对修正时间的梯形积分。",
        "",
        "## 推荐主输入",
        "",
        "主模型优先使用充电阶段：",
        "",
        "```text",
        f"charge_voltage_v       [cycle, {points}]",
        f"charge_current_a       [cycle, {points}]",
        "charge_duration_s      [cycle, 1]",
        "label_soh_nominal      [cycle, 1]",
        "```",
        "",
        "放电阶段同样完整保存，供阶段对照或数据审计使用。不要在未做消融的情况下同时增加为主模型输入。",
        "",
        "## 12循环窗口",
        "",
        "`window_row_indices` 保存每个连续12循环窗口对应的循环行号。只有12个原始观测循环编号严格连续时才生成窗口，不会跨过被QC排除的循环。窗口标签由最后一个循环给出。",
        "",
        "## 时间字段",
        "",
        "- `*_elapsed_time_s`：从0到实际活动阶段持续时间的插值时间网格。",
        "- `*_duration_s`：活动阶段累计持续时间，建议作为网络第三输入。",
        "- `*_wall_duration_s`：首段开始至末段结束的墙钟跨度，包含阶段中断，仅用于审计。",
        "",
        "多段活动阶段会在累计活动时间轴上连接，静置间隔不会被伪造为充放电采样点。",
        "",
        "## 数据量",
        "",
        "| 数据集 | 电芯 | 有效循环 | 连续12循环窗口 |",
        "|---|---:|---:|---:|",
    ]
    for dataset, summary in dataset_summaries.items():
        lines.append(
            f"| {dataset} | {summary['cell_count']} | {summary['valid_cycles']} | {summary['valid_12_cycle_windows']} |"
        )

    lines.extend(
        [
            "",
            "## 插值质量",
            "",
            "下列指标为全部输出循环在原采样时刻的插值回放误差95分位数：",
            "",
            "| 阶段 | 电压MAE P95 (mV) | 电流MAE P95 (mA) | 电流积分相对误差P95 (%) |",
            "|---|---:|---:|---:|",
        ]
    )
    for phase in ("charge", "discharge"):
        subset = [row for row in quality_rows if row["phase"] == phase]
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} |".format(
                phase,
                percentile([row["voltage_mae_mv"] for row in subset], 95),
                percentile([row["current_mae_ma"] for row in subset], 95),
                percentile(
                    [row["current_integral_relative_error_pct"] for row in subset],
                    95,
                ),
            )
        )
    lines.extend(
        [
            "",
            "## 加载示例",
            "",
            "```python",
            "import numpy as np",
            "",
            "data = np.load(r\"D:\\learning\\小论文\\数据\\model_input_v1\\CALCE\\CS2_35.npz\", allow_pickle=False)",
            "rows = data[\"window_row_indices\"][0]",
            "x_voltage = data[\"charge_voltage_v\"][rows]       # [12, 256]",
            "x_current = data[\"charge_current_a\"][rows]       # [12, 256]",
            "x_duration = data[\"charge_duration_s\"][rows]     # [12, 1]",
            "y_soh = data[\"window_target_soh_nominal\"][0]     # [1]",
            "y_capacity = data[\"window_target_capacity_ah\"][0]",
            "```",
            "",
            "Pickle仅用于读取本地可信的上游字典；最终模型输入为不需要 `allow_pickle=True` 的NPZ。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    args = parser.parse_args()

    if args.points < 2:
        raise ValueError("--points must be at least 2")
    if args.window < 1:
        raise ValueError("--window must be positive")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    staging = args.output.with_name(args.output.name + ".building")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)

    all_index_rows: list[dict[str, Any]] = []
    all_quality_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "source_root": str(args.input),
        "output_root": str(args.output),
        "interpolation": {
            "method": "linear_on_cumulative_active_elapsed_time",
            "points": args.points,
            "rest_gaps_interpolated": False,
            "signal_smoothing": False,
            "normalization": False,
        },
        "labels": {
            "primary": "label_soh_nominal",
            "definition": "capacity_label_qc_ah / dataset_nominal_capacity_ah",
            "capacity_auxiliary": "capacity_ah",
            "clipped": False,
        },
        "window": {
            "size": args.window,
            "requires_consecutive_observed_cycle_indices": True,
            "target": "last cycle in window",
        },
        "datasets": {},
    }

    for dataset, index_name in INDEX_NAMES.items():
        source_root = args.input / dataset
        with (source_root / index_name).open("rb") as handle:
            source_index = pickle.load(handle)
        dataset_output = staging / dataset
        dataset_output.mkdir()
        cell_summaries: list[dict[str, Any]] = []
        for cell_id, entry in sorted(source_index["cells"].items()):
            with (source_root / entry["file"]).open("rb") as handle:
                cell = pickle.load(handle)
            summary, index_rows, quality_rows = write_cell_npz(
                dataset_output / f"{cell_id}.npz",
                cell,
                dataset,
                args.points,
                args.window,
            )
            cell_summaries.append(summary)
            all_index_rows.extend(index_rows)
            all_quality_rows.extend(quality_rows)

        dataset_summary = {
            "cell_count": len(cell_summaries),
            "valid_cycles": sum(item["valid_cycles"] for item in cell_summaries),
            "excluded_cycles": sum(
                item["excluded_cycles"] for item in cell_summaries
            ),
            "valid_12_cycle_windows": sum(
                item["valid_12_cycle_windows"] for item in cell_summaries
            ),
            "nominal_capacity_ah": NOMINAL_CAPACITY_AH[dataset],
            "cells": cell_summaries,
        }
        manifest["datasets"][dataset] = dataset_summary
        atomic_json(dataset_output / "dataset_index.json", dataset_summary)

    write_csv(staging / "cycle_index.csv", all_index_rows)
    write_csv(staging / "interpolation_quality.csv", all_quality_rows)
    atomic_json(staging / "dataset_manifest.json", manifest)
    atomic_text(
        staging / "README.md",
        build_readme(manifest["datasets"], all_quality_rows, args.points),
    )

    checksums = []
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksums.append(
                f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}"
            )
    atomic_text(staging / "checksums.sha256", "\n".join(checksums) + "\n")
    os.replace(staging, args.output)

    totals = Counter()
    for summary in manifest["datasets"].values():
        totals["cells"] += summary["cell_count"]
        totals["cycles"] += summary["valid_cycles"]
        totals["windows"] += summary["valid_12_cycle_windows"]
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "output": str(args.output),
                "points": args.points,
                "window": args.window,
                **totals,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
