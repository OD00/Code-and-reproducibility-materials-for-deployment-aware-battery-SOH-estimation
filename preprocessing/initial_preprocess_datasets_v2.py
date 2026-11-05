from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import openpyxl
import polars as pl


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

OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "dataset": pl.String,
    "chemistry": pl.String,
    "cell_id": pl.String,
    "condition_group": pl.String,
    "temperature_setpoint_c": pl.Float64,
    "charge_rate_c": pl.Float64,
    "discharge_rate_c": pl.Float64,
    "source_file": pl.String,
    "source_sheet": pl.String,
    "source_format": pl.String,
    "source_row": pl.Int64,
    "source_date": pl.String,
    "timestamp": pl.String,
    "cycle_id_raw": pl.Int64,
    "cycle_uid": pl.String,
    "step_id_raw": pl.Int64,
    "test_time_s_raw": pl.Float64,
    "test_time_s": pl.Float64,
    "time_segment_id": pl.Int64,
    "step_time_s": pl.Float64,
    "voltage_v": pl.Float64,
    "current_a": pl.Float64,
    "temperature_c": pl.Float64,
    "charge_capacity_ah": pl.Float64,
    "discharge_capacity_ah": pl.Float64,
    "capacity_raw_ah": pl.Float64,
    "control_value_raw": pl.Float64,
    "control_voltage_v": pl.Float64,
    "control_current_a": pl.Float64,
    "status_code": pl.Int64,
    "status_category": pl.Int64,
    "program_cycle": pl.Int64,
    "charge_count": pl.Int64,
    "discharge_count": pl.Int64,
    "phase": pl.String,
    "quality_flag": pl.String,
    "repair_action": pl.String,
    "valid_core": pl.Boolean,
    "usable_for_model": pl.Boolean,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preliminary normalization of selected CALCE and TJU battery records."
    )
    parser.add_argument(
        "--calce-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/raw/CALCE",
    )
    parser.add_argument(
        "--tju-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/raw/TJU",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=("all", "calce", "tju"), default="all"
    )
    parser.add_argument(
        "--current-threshold-a",
        type=float,
        default=0.01,
        help="Absolute current below this threshold is provisionally labelled rest.",
    )
    return parser.parse_args()


def typed_null(name: str) -> pl.Expr:
    return pl.lit(None, dtype=OUTPUT_SCHEMA[name]).alias(name)


def cast_expr(source: str, target: str, dtype: pl.DataType) -> pl.Expr:
    return pl.col(source).cast(dtype, strict=False).alias(target)


def finalize_frame(df: pl.DataFrame, threshold_a: float) -> pl.DataFrame:
    defaults: list[pl.Expr] = []
    if "test_time_s_raw" not in df.columns:
        defaults.append(pl.col("test_time_s").alias("test_time_s_raw"))
    if "time_segment_id" not in df.columns:
        defaults.append(pl.lit(0, dtype=pl.Int64).alias("time_segment_id"))
    if "quality_flag" not in df.columns:
        defaults.append(pl.lit("none").alias("quality_flag"))
    if "repair_action" not in df.columns:
        defaults.append(pl.lit("none").alias("repair_action"))
    if defaults:
        df = df.with_columns(defaults)

    missing = [
        typed_null(name) for name in OUTPUT_SCHEMA if name not in df.columns
    ]
    if missing:
        df = df.with_columns(missing)

    df = df.with_columns(
        [
            pl.col(name).cast(dtype, strict=False).alias(name)
            for name, dtype in OUTPUT_SCHEMA.items()
            if name not in ("cycle_uid", "phase", "valid_core", "usable_for_model")
        ]
    )
    df = df.with_columns(
        pl.when(pl.col("current_a").is_null())
        .then(pl.lit("unknown"))
        .when(pl.col("current_a") > threshold_a)
        .then(pl.lit("charge"))
        .when(pl.col("current_a") < -threshold_a)
        .then(pl.lit("discharge"))
        .otherwise(pl.lit("rest"))
        .alias("phase"),
        pl.when(pl.col("cycle_id_raw").is_not_null())
        .then(
            pl.concat_str(
                [
                    pl.col("time_segment_id").cast(pl.String),
                    pl.col("cycle_id_raw").cast(pl.String),
                ],
                separator=":",
            )
        )
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("cycle_uid"),
        (
            pl.col("test_time_s").is_not_null()
            & pl.col("voltage_v").is_not_null()
            & pl.col("current_a").is_not_null()
            & (pl.col("quality_flag") != "corrupt_placeholder")
        ).alias("valid_core"),
    )
    df = df.with_columns(
        (
            pl.col("valid_core")
            & ~pl.col("quality_flag").is_in(
                ["corrupt_placeholder", "suspect_low_voltage_rest"]
            )
        ).alias("usable_for_model")
    )
    return df.select(list(OUTPUT_SCHEMA))


def parse_source_date(path: Path, cell_id: str) -> str | None:
    tail = path.stem[len(cell_id) :].lstrip("_")
    match = re.match(r"^(\d{1,2})_(\d{1,2})_(\d{2,4})$", tail)
    if not match:
        return None
    month, day, year = (int(value) for value in match.groups())
    if year < 100:
        year += 2000
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def iso_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds")
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time()).isoformat()
    text = str(value).strip()
    return text or None


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    numeric = as_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def safe_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.#-]+", "_", value)


def summarize_frame(
    df: pl.DataFrame,
    *,
    source_path: Path,
    output_path: Path,
    input_root: Path,
    source_sheet: str | None,
) -> dict[str, Any]:
    row = df.select(
        pl.len().alias("rows"),
        pl.col("valid_core").sum().alias("valid_core_rows"),
        pl.col("usable_for_model").sum().alias("usable_for_model_rows"),
        (pl.col("quality_flag") != "none").sum().alias("quality_flagged_rows"),
        pl.col("time_segment_id").n_unique().alias("time_segments"),
        (pl.col("phase") == "charge").sum().alias("charge_rows"),
        (pl.col("phase") == "discharge").sum().alias("discharge_rows"),
        (pl.col("phase") == "rest").sum().alias("rest_rows"),
        (pl.col("phase") == "unknown").sum().alias("unknown_rows"),
        pl.col("cycle_id_raw").drop_nulls().n_unique().alias("cycle_ids_in_source"),
        pl.col("cycle_id_raw").min().alias("cycle_id_min"),
        pl.col("cycle_id_raw").max().alias("cycle_id_max"),
        pl.col("test_time_s").min().alias("time_s_min"),
        pl.col("test_time_s").max().alias("time_s_max"),
        pl.col("voltage_v").min().alias("voltage_v_min"),
        pl.col("voltage_v").max().alias("voltage_v_max"),
        pl.col("current_a").min().alias("current_a_min"),
        pl.col("current_a").max().alias("current_a_max"),
        pl.col("timestamp").min().alias("timestamp_min"),
        pl.col("timestamp").max().alias("timestamp_max"),
    ).row(0, named=True)
    row.update(
        {
            "dataset": df["dataset"][0] if df.height else None,
            "cell_id": df["cell_id"][0] if df.height else None,
            "condition_group": df["condition_group"][0] if df.height else None,
            "source_file": source_path.relative_to(input_root).as_posix(),
            "source_sheet": source_sheet,
            "source_format": source_path.suffix.lower().lstrip("."),
            "input_bytes": source_path.stat().st_size,
            "output_file": output_path.as_posix(),
            "output_bytes": output_path.stat().st_size,
            "status": "ok",
            "error": None,
        }
    )
    quality_counts = {
        item["quality_flag"]: int(item["len"])
        for item in df.group_by("quality_flag").len().sort("quality_flag").to_dicts()
    }
    row["quality_flag_counts_json"] = json.dumps(
        quality_counts, ensure_ascii=False, sort_keys=True
    )
    return row


def write_parquet_and_summarize(
    df: pl.DataFrame,
    *,
    source_path: Path,
    output_path: Path,
    input_root: Path,
    source_sheet: str | None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd", statistics=True)
    return summarize_frame(
        df,
        source_path=source_path,
        output_path=output_path,
        input_root=input_root,
        source_sheet=source_sheet,
    )


def calce_condition(cell_id: str) -> str:
    if cell_id in {"CS2_8", "CS2_21", "CS2_33", "CS2_34"}:
        return "CALCE_group_0.5C"
    return "CALCE_group_1C"


def read_calce_txt(
    path: Path, cell_id: str, input_root: Path, threshold_a: float
) -> pl.DataFrame:
    raw = pl.read_csv(
        path,
        separator="\t",
        infer_schema=False,
        encoding="utf8-lossy",
        truncate_ragged_lines=True,
    ).with_row_index("source_row", offset=2)
    relative = path.relative_to(input_root).as_posix()
    df = raw.select(
        pl.lit("CALCE").alias("dataset"),
        pl.lit("LCO").alias("chemistry"),
        pl.lit(cell_id).alias("cell_id"),
        pl.lit(calce_condition(cell_id)).alias("condition_group"),
        pl.lit(relative).alias("source_file"),
        pl.lit(None, dtype=pl.String).alias("source_sheet"),
        pl.lit("txt").alias("source_format"),
        pl.col("source_row"),
        pl.lit(parse_source_date(path, cell_id)).alias("source_date"),
        cast_expr("Time", "test_time_s_raw", pl.Float64),
        cast_expr("Time", "test_time_s", pl.Float64),
        (pl.col("mV").cast(pl.Float64, strict=False) / 1000.0).alias("voltage_v"),
        (pl.col("mA").cast(pl.Float64, strict=False) / 1000.0).alias("current_a"),
        cast_expr("Temperature", "temperature_c", pl.Float64),
        (pl.col("Capacity").cast(pl.Float64, strict=False) / 1000.0).alias("capacity_raw_ah"),
        cast_expr("Status code", "status_code", pl.Int64),
        cast_expr("Status category", "status_category", pl.Int64),
        cast_expr("Pgm step", "step_id_raw", pl.Int64),
        cast_expr("Pgm cycle", "program_cycle", pl.Int64),
        cast_expr("Charge count", "charge_count", pl.Int64),
        cast_expr("Discharge count", "discharge_count", pl.Int64),
    )
    return finalize_frame(df, threshold_a)


def read_calce_xlsx_sheet(
    path: Path,
    sheet_name: str,
    cell_id: str,
    input_root: Path,
    threshold_a: float,
) -> tuple[pl.DataFrame, int]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook[sheet_name]
        rows = sheet.iter_rows(values_only=True)
        headers = [str(value).strip() if value is not None else "" for value in next(rows)]
        indices = {name: index for index, name in enumerate(headers)}
        needed = {
            "Data_Point",
            "Test_Time(s)",
            "Date_Time",
            "Step_Time(s)",
            "Step_Index",
            "Cycle_Index",
            "Current(A)",
            "Voltage(V)",
            "Charge_Capacity(Ah)",
            "Discharge_Capacity(Ah)",
        }
        missing = sorted(needed - set(indices))
        if missing:
            raise ValueError(f"missing XLSX columns: {missing}")

        data: dict[str, list[Any]] = defaultdict(list)
        skipped_blank_rows = 0

        def value(row: tuple[Any, ...], name: str) -> Any:
            index = indices[name]
            return row[index] if index < len(row) else None

        for source_row, row in enumerate(rows, start=2):
            time_s = as_float(value(row, "Test_Time(s)"))
            voltage_v = as_float(value(row, "Voltage(V)"))
            current_a = as_float(value(row, "Current(A)"))
            if time_s is None and voltage_v is None and current_a is None:
                skipped_blank_rows += 1
                continue
            data["source_row"].append(source_row)
            data["timestamp"].append(iso_timestamp(value(row, "Date_Time")))
            data["cycle_id_raw"].append(as_int(value(row, "Cycle_Index")))
            data["step_id_raw"].append(as_int(value(row, "Step_Index")))
            data["test_time_s_raw"].append(time_s)
            data["test_time_s"].append(time_s)
            data["step_time_s"].append(as_float(value(row, "Step_Time(s)")))
            data["voltage_v"].append(voltage_v)
            data["current_a"].append(current_a)
            data["charge_capacity_ah"].append(as_float(value(row, "Charge_Capacity(Ah)")))
            data["discharge_capacity_ah"].append(as_float(value(row, "Discharge_Capacity(Ah)")))

        relative = path.relative_to(input_root).as_posix()
        df = pl.DataFrame(data).with_columns(
            pl.lit("CALCE").alias("dataset"),
            pl.lit("LCO").alias("chemistry"),
            pl.lit(cell_id).alias("cell_id"),
            pl.lit(calce_condition(cell_id)).alias("condition_group"),
            pl.lit(relative).alias("source_file"),
            pl.lit(sheet_name).alias("source_sheet"),
            pl.lit("xlsx").alias("source_format"),
            pl.lit(parse_source_date(path, cell_id)).alias("source_date"),
        )
        impute_first_time = (
            pl.col("test_time_s").is_null()
            & (pl.col("source_row") == 2)
            & pl.col("step_time_s").is_not_null()
        )
        df = df.with_columns(
            pl.when(impute_first_time)
            .then(pl.col("step_time_s"))
            .otherwise(pl.col("test_time_s"))
            .alias("test_time_s"),
            pl.when(impute_first_time)
            .then(pl.lit("imputed_missing_test_time"))
            .otherwise(pl.lit("none"))
            .alias("quality_flag"),
            pl.when(impute_first_time)
            .then(pl.lit("test_time_s_from_step_time_s"))
            .otherwise(pl.lit("none"))
            .alias("repair_action"),
        )
        return finalize_frame(df, threshold_a), skipped_blank_rows
    finally:
        workbook.close()


def process_calce(
    input_root: Path, output_root: Path, threshold_a: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifests: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for cell_id in CALCE_CELLS:
        cell_dir = input_root / cell_id
        files = sorted(cell_dir.glob("*.txt")) + sorted(cell_dir.glob("*.xlsx"))
        print(f"[CALCE] {cell_id}: {len(files)} source files", flush=True)
        for index, path in enumerate(files, start=1):
            try:
                if path.suffix.lower() == ".txt":
                    df = read_calce_txt(path, cell_id, input_root, threshold_a)
                    out = output_root / "records" / "CALCE" / cell_id / f"{path.stem}.parquet"
                    manifests.append(
                        write_parquet_and_summarize(
                            df,
                            source_path=path,
                            output_path=out,
                            input_root=input_root,
                            source_sheet=None,
                        )
                    )
                else:
                    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
                    channel_sheets = [name for name in workbook.sheetnames if name.startswith("Channel_")]
                    workbook.close()
                    if not channel_sheets:
                        raise ValueError("no Channel_* sheet found")
                    for sheet_name in channel_sheets:
                        df, skipped = read_calce_xlsx_sheet(
                            path, sheet_name, cell_id, input_root, threshold_a
                        )
                        suffix = "" if len(channel_sheets) == 1 else f"__{safe_part(sheet_name)}"
                        out = output_root / "records" / "CALCE" / cell_id / f"{path.stem}{suffix}.parquet"
                        item = write_parquet_and_summarize(
                            df,
                            source_path=path,
                            output_path=out,
                            input_root=input_root,
                            source_sheet=sheet_name,
                        )
                        item["skipped_blank_rows"] = skipped
                        manifests.append(item)
                print(f"  [{index}/{len(files)}] {path.name}", flush=True)
            except Exception as exc:  # retain the rest of the run for inspection
                error = {
                    "dataset": "CALCE",
                    "cell_id": cell_id,
                    "source_file": path.relative_to(input_root).as_posix(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                errors.append(error)
                print(f"  ERROR {path.name}: {type(exc).__name__}: {exc}", flush=True)
    return manifests, errors


def parse_tju_metadata(stem: str) -> dict[str, Any]:
    match = re.match(
        r"^(CY(?P<temperature>\d+)-(?P<charge>025|05|1)_(?P<discharge>\d+))-#(?P<cell>\d+)$",
        stem,
    )
    if not match:
        raise ValueError(f"unexpected TJU filename: {stem}")
    rate_map = {"025": 0.25, "05": 0.5, "1": 1.0}
    return {
        "cell_id": stem,
        "condition_group": match.group(1),
        "temperature_setpoint_c": float(match.group("temperature")),
        "charge_rate_c": rate_map[match.group("charge")],
        "discharge_rate_c": float(match.group("discharge")),
    }


def read_tju_csv(
    path: Path, input_root: Path, threshold_a: float
) -> pl.DataFrame:
    metadata = parse_tju_metadata(path.stem)
    raw = pl.read_csv(
        path,
        infer_schema_length=10000,
        null_values=["", "nan", "NaN"],
    ).with_row_index("source_row", offset=2)
    relative = path.relative_to(input_root).as_posix()
    df = raw.select(
        pl.lit("TJU_Dataset_1").alias("dataset"),
        pl.lit("NCA").alias("chemistry"),
        pl.lit(metadata["cell_id"]).alias("cell_id"),
        pl.lit(metadata["condition_group"]).alias("condition_group"),
        pl.lit(metadata["temperature_setpoint_c"]).alias("temperature_setpoint_c"),
        pl.lit(metadata["charge_rate_c"]).alias("charge_rate_c"),
        pl.lit(metadata["discharge_rate_c"]).alias("discharge_rate_c"),
        pl.lit(relative).alias("source_file"),
        pl.lit(None, dtype=pl.String).alias("source_sheet"),
        pl.lit("csv").alias("source_format"),
        pl.col("source_row"),
        pl.col("cycle number").round(0).cast(pl.Int64, strict=False).alias("cycle_id_raw"),
        cast_expr("time/s", "test_time_s_raw", pl.Float64),
        cast_expr("time/s", "test_time_s", pl.Float64),
        cast_expr("Ecell/V", "voltage_v", pl.Float64),
        (pl.col("<I>/mA").cast(pl.Float64, strict=False) / 1000.0).alias("current_a"),
        (pl.col("Q charge/mA.h").cast(pl.Float64, strict=False) / 1000.0).alias("charge_capacity_ah"),
        (pl.col("Q discharge/mA.h").cast(pl.Float64, strict=False) / 1000.0).alias("discharge_capacity_ah"),
        cast_expr("control/V/mA", "control_value_raw", pl.Float64),
        cast_expr("control/V", "control_voltage_v", pl.Float64),
        (pl.col("control/mA").cast(pl.Float64, strict=False) / 1000.0).alias("control_current_a"),
    )
    corrupt_placeholder = (
        (pl.col("test_time_s") < 0)
        & (pl.col("voltage_v") == 0)
        & (pl.col("current_a") == 0)
        & (pl.col("charge_capacity_ah") == 0)
        & (pl.col("discharge_capacity_ah") == 0)
    )
    df = df.with_columns(
        corrupt_placeholder.alias("_corrupt_placeholder"),
        pl.when(corrupt_placeholder)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("test_time_s"))
        .alias("_clean_time_s"),
    )
    df = df.with_columns(
        pl.col("_clean_time_s").forward_fill().shift(1).alias("_previous_clean_time_s")
    )
    df = df.with_columns(
        (
            pl.col("_clean_time_s").is_not_null()
            & pl.col("_previous_clean_time_s").is_not_null()
            & (pl.col("_clean_time_s") < pl.col("_previous_clean_time_s"))
        )
        .fill_null(False)
        .alias("_time_reset")
    )
    df = df.with_columns(
        pl.col("_time_reset").cum_sum().cast(pl.Int64).alias("time_segment_id"),
        pl.when(pl.col("_corrupt_placeholder"))
        .then(pl.lit("corrupt_placeholder"))
        .when((pl.col("voltage_v") < 2.0) & (pl.col("current_a").abs() <= threshold_a))
        .then(pl.lit("suspect_low_voltage_rest"))
        .when(pl.col("_time_reset"))
        .then(pl.lit("time_reset_segment_start"))
        .otherwise(pl.lit("none"))
        .alias("quality_flag"),
        pl.when(pl.col("_corrupt_placeholder"))
        .then(pl.lit("retained_but_excluded"))
        .when((pl.col("voltage_v") < 2.0) & (pl.col("current_a").abs() <= threshold_a))
        .then(pl.lit("retained_no_interpolation"))
        .when(pl.col("_time_reset"))
        .then(pl.lit("new_time_segment"))
        .otherwise(pl.lit("none"))
        .alias("repair_action"),
    ).drop(
        "_corrupt_placeholder",
        "_clean_time_s",
        "_previous_clean_time_s",
        "_time_reset",
    )
    return finalize_frame(df, threshold_a)


def process_tju(
    input_root: Path, output_root: Path, threshold_a: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifests: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    files = sorted(
        path
        for group in TJU_GROUPS
        for path in input_root.glob(f"{group}-#*.csv")
    )
    print(f"[TJU] Dataset 1 selected CSV files: {len(files)}", flush=True)
    for index, path in enumerate(files, start=1):
        try:
            df = read_tju_csv(path, input_root, threshold_a)
            metadata = parse_tju_metadata(path.stem)
            out = (
                output_root
                / "records"
                / "TJU_Dataset_1"
                / metadata["condition_group"]
                / f"{path.stem}.parquet"
            )
            manifests.append(
                write_parquet_and_summarize(
                    df,
                    source_path=path,
                    output_path=out,
                    input_root=input_root,
                    source_sheet=None,
                )
            )
            print(f"  [{index}/{len(files)}] {path.name}", flush=True)
        except Exception as exc:  # retain the rest of the run for inspection
            error = {
                "dataset": "TJU_Dataset_1",
                "cell_id": path.stem,
                "source_file": path.relative_to(input_root).as_posix(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            errors.append(error)
            print(f"  ERROR {path.name}: {type(exc).__name__}: {exc}", flush=True)
    return manifests, errors


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(
    manifests: list[dict[str, Any]], key_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in manifests:
        grouped[tuple(row.get(key) for key in key_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        item = dict(zip(key_fields, key))
        item.update(
            {
                "source_files": len(rows),
                "rows": sum(int(row.get("rows") or 0) for row in rows),
                "valid_core_rows": sum(int(row.get("valid_core_rows") or 0) for row in rows),
                "usable_for_model_rows": sum(int(row.get("usable_for_model_rows") or 0) for row in rows),
                "quality_flagged_rows": sum(int(row.get("quality_flagged_rows") or 0) for row in rows),
                "time_segments_sum": sum(int(row.get("time_segments") or 0) for row in rows),
                "charge_rows": sum(int(row.get("charge_rows") or 0) for row in rows),
                "discharge_rows": sum(int(row.get("discharge_rows") or 0) for row in rows),
                "rest_rows": sum(int(row.get("rest_rows") or 0) for row in rows),
                "unknown_rows": sum(int(row.get("unknown_rows") or 0) for row in rows),
                "input_bytes": sum(int(row.get("input_bytes") or 0) for row in rows),
                "output_bytes": sum(int(row.get("output_bytes") or 0) for row in rows),
            }
        )
        output.append(item)
    return output


def write_metadata(
    output_root: Path,
    manifests: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    manifests = sorted(
        manifests,
        key=lambda row: (
            str(row.get("dataset")),
            str(row.get("cell_id")),
            str(row.get("source_file")),
            str(row.get("source_sheet")),
        ),
    )
    write_csv(output_root / "manifest.csv", manifests)
    write_csv(output_root / "cell_summary.csv", aggregate_rows(manifests, ("dataset", "cell_id", "condition_group")))
    write_csv(output_root / "condition_summary.csv", aggregate_rows(manifests, ("dataset", "condition_group")))
    write_csv(output_root / "errors.csv", errors)

    mapping_rows = [
        {"standard_column": "test_time_s", "CALCE_TXT": "Time", "CALCE_XLSX": "Test_Time(s)", "TJU_CSV": "time/s", "unit": "s"},
        {"standard_column": "test_time_s_raw", "CALCE_TXT": "Time", "CALCE_XLSX": "Test_Time(s)", "TJU_CSV": "time/s", "unit": "s; unmodified source value"},
        {"standard_column": "time_segment_id", "CALCE_TXT": "0", "CALCE_XLSX": "0", "TJU_CSV": "increments when time/s restarts", "unit": "segment id"},
        {"standard_column": "timestamp", "CALCE_TXT": "not available", "CALCE_XLSX": "Date_Time", "TJU_CSV": "not available", "unit": "ISO-8601 text"},
        {"standard_column": "voltage_v", "CALCE_TXT": "mV / 1000", "CALCE_XLSX": "Voltage(V)", "TJU_CSV": "Ecell/V", "unit": "V"},
        {"standard_column": "current_a", "CALCE_TXT": "mA / 1000", "CALCE_XLSX": "Current(A)", "TJU_CSV": "<I>/mA / 1000", "unit": "A"},
        {"standard_column": "temperature_c", "CALCE_TXT": "Temperature", "CALCE_XLSX": "not available in Channel sheet", "TJU_CSV": "not measured; setpoint parsed from filename", "unit": "degC"},
        {"standard_column": "charge_capacity_ah", "CALCE_TXT": "not separately available", "CALCE_XLSX": "Charge_Capacity(Ah)", "TJU_CSV": "Q charge/mA.h / 1000", "unit": "Ah"},
        {"standard_column": "discharge_capacity_ah", "CALCE_TXT": "not separately available", "CALCE_XLSX": "Discharge_Capacity(Ah)", "TJU_CSV": "Q discharge/mA.h / 1000", "unit": "Ah"},
        {"standard_column": "capacity_raw_ah", "CALCE_TXT": "Capacity / 1000", "CALCE_XLSX": "not used", "TJU_CSV": "not used", "unit": "Ah"},
        {"standard_column": "cycle_id_raw", "CALCE_TXT": "not inferred; retain charge_count/discharge_count", "CALCE_XLSX": "Cycle_Index", "TJU_CSV": "cycle number", "unit": "raw id"},
        {"standard_column": "phase", "CALCE_TXT": "current sign", "CALCE_XLSX": "current sign", "TJU_CSV": "current sign", "unit": f"charge > {args.current_threshold_a} A; discharge < -{args.current_threshold_a} A"},
        {"standard_column": "quality_flag", "CALCE_TXT": "none", "CALCE_XLSX": "missing first Test_Time may be imputed", "TJU_CSV": "corrupt placeholder, low-voltage rest, or time reset", "unit": "diagnostic label"},
        {"standard_column": "usable_for_model", "CALCE_TXT": "derived", "CALCE_XLSX": "derived", "TJU_CSV": "false for corrupt placeholder and suspect low-voltage rest", "unit": "boolean"},
    ]
    write_csv(output_root / "column_mapping.csv", mapping_rows)

    config = {
        "processing_version": "initial_v2",
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "polars": pl.__version__,
        "openpyxl": openpyxl.__version__,
        "dataset_selection": args.dataset,
        "calce_input_root": str(args.calce_root),
        "tju_input_root": str(args.tju_root),
        "output_root": str(args.output_root),
        "current_threshold_a": args.current_threshold_a,
        "calce_cells": CALCE_CELLS,
        "tju_groups": TJU_GROUPS,
        "operations_applied": [
            "column mapping",
            "unit conversion to SI-friendly V/A/Ah/s",
            "provisional phase label from current sign",
            "retain raw test time and add time-segment identifiers",
            "impute a missing first CALCE Test_Time from Step_Time",
            "tag low-voltage TJU rest rows without interpolation",
            "retain but exclude an all-zero negative-time placeholder row",
            "Parquet compression using zstd",
        ],
        "operations_not_applied": [
            "outlier removal",
            "interpolation or resampling",
            "sequence truncation or padding",
            "normalization or scaling",
            "cycle renumbering",
            "train/validation/test splitting",
        ],
        "successful_outputs": len(manifests),
        "errors": len(errors),
    }
    (output_root / "processing_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    output_root: Path = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        print(f"Output directory is not empty: {output_root}", file=sys.stderr)
        print("Use a new output directory to keep processing runs immutable.", file=sys.stderr)
        return 2
    output_root.mkdir(parents=True, exist_ok=True)

    manifests: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if args.dataset in ("all", "calce"):
        calce_manifest, calce_errors = process_calce(
            args.calce_root.resolve(), output_root, args.current_threshold_a
        )
        manifests.extend(calce_manifest)
        errors.extend(calce_errors)
    if args.dataset in ("all", "tju"):
        tju_manifest, tju_errors = process_tju(
            args.tju_root.resolve(), output_root, args.current_threshold_a
        )
        manifests.extend(tju_manifest)
        errors.extend(tju_errors)

    write_metadata(output_root, manifests, errors, args)
    print(
        f"Completed: {len(manifests)} outputs, {len(errors)} errors -> {output_root}",
        flush=True,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
