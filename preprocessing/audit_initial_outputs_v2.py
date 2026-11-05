from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit initial_v2 battery outputs.")
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    manifest = pl.read_csv(root / "manifest.csv")

    schema_counts: dict[tuple[tuple[str, str], ...], int] = defaultdict(int)
    action_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    action_samples: list[dict[str, Any]] = []
    unhandled: list[dict[str, Any]] = []
    total_rows = 0
    valid_core_rows = 0
    usable_rows = 0
    missing_outputs = 0

    for item in manifest.to_dicts():
        path = Path(item["output_file"])
        if not path.exists():
            missing_outputs += 1
            continue
        schema = pl.read_parquet_schema(path)
        signature = tuple((name, str(dtype)) for name, dtype in schema.items())
        schema_counts[signature] += 1
        frame = pl.read_parquet(
            path,
            columns=[
                "source_row",
                "cycle_id_raw",
                "cycle_uid",
                "test_time_s_raw",
                "test_time_s",
                "time_segment_id",
                "voltage_v",
                "current_a",
                "phase",
                "quality_flag",
                "repair_action",
                "valid_core",
                "usable_for_model",
            ],
        )
        total_rows += frame.height
        valid_core_rows += int(frame["valid_core"].sum())
        usable_rows += int(frame["usable_for_model"].sum())

        for action in (
            frame.filter(pl.col("quality_flag") != "none")
            .group_by("quality_flag", "repair_action")
            .len()
            .to_dicts()
        ):
            key = (
                item["dataset"],
                item["cell_id"],
                action["quality_flag"],
                action["repair_action"],
            )
            action_counts[key] += int(action["len"])

        for row in (
            frame.filter(pl.col("quality_flag") != "none")
            .group_by("quality_flag", maintain_order=True)
            .head(3)
            .to_dicts()
        ):
            action_samples.append(
                {
                    "dataset": item["dataset"],
                    "cell_id": item["cell_id"],
                    "source_file": item["source_file"],
                    **row,
                }
            )

        raw_time_back = (pl.col("test_time_s_raw").diff() < 0).fill_null(False)
        segment_increment = (pl.col("time_segment_id").diff() > 0).fill_null(False)
        checks = {
            "unhandled_missing_core": (
                ~pl.col("valid_core")
                & (pl.col("quality_flag") != "corrupt_placeholder")
            ),
            "untagged_voltage_below_2v": (
                (pl.col("voltage_v") < 2.0)
                & (pl.col("quality_flag") == "none")
            ),
            "untagged_voltage_above_5v": (
                (pl.col("voltage_v") > 5.0)
                & (pl.col("quality_flag") == "none")
            ),
            "untagged_abs_current_above_5a": (
                (pl.col("current_a").abs() > 5.0)
                & (pl.col("quality_flag") == "none")
            ),
            "unsegmented_time_decrease": (
                raw_time_back
                & ~segment_increment
                & (pl.col("quality_flag") != "corrupt_placeholder")
            ),
        }
        for issue, expression in checks.items():
            rows = frame.filter(expression)
            if rows.height:
                unhandled.append(
                    {
                        "dataset": item["dataset"],
                        "cell_id": item["cell_id"],
                        "source_file": item["source_file"],
                        "issue": issue,
                        "rows": rows.height,
                        "first_source_row": rows["source_row"].min(),
                    }
                )

    action_rows = [
        {
            "dataset": key[0],
            "cell_id": key[1],
            "quality_flag": key[2],
            "repair_action": key[3],
            "rows": count,
        }
        for key, count in sorted(action_counts.items())
    ]
    write_csv(root / "quality_actions.csv", action_rows)
    write_csv(root / "quality_action_samples.csv", action_samples)
    write_csv(root / "quality_issues_unhandled.csv", unhandled)

    report = {
        "processing_version": "initial_v2",
        "created_at": datetime.now().astimezone().isoformat(),
        "manifest_outputs": manifest.height,
        "missing_output_files": missing_outputs,
        "schema_variants": len(schema_counts),
        "schema_variant_counts": list(schema_counts.values()),
        "total_rows": total_rows,
        "valid_core_rows": valid_core_rows,
        "usable_for_model_rows": usable_rows,
        "quality_action_rows": sum(action_counts.values()),
        "unhandled_issue_groups": len(unhandled),
        "note": "Source values are retained. Suspect rows are labelled and excluded through usable_for_model; no interpolation was applied.",
    }
    (root / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if missing_outputs or len(schema_counts) != 1 or unhandled else 0


if __name__ == "__main__":
    raise SystemExit(main())
