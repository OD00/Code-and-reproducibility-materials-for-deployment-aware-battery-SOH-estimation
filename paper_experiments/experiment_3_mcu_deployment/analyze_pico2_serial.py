"""Summarize one Pico 2 INT8 deployment log and compare it with local TFLite."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from src.common.experiment import regression_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--configured-arena-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8")
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    matches = re.findall(
        r"sample=(\d+) yq=(-?\d+) y=([0-9.\-]+) target=([0-9.\-]+)", text
    )
    timing = re.search(
        r"timing runs=(\d+) cold_us=(\d+) mean_us=([0-9.]+) sd_us=([0-9.]+) "
        r"p50_us=(\d+) p95_us=(\d+) max_us=(\d+)",
        text,
    )
    arena_used = re.search(r"arena_used=(\d+)", text)
    firmware_provenance = re.search(
        r"firmware build=([^\s]+) sys_clock_hz=(\d+)", text
    )
    allocation_failure = re.search(
        r"Failed to resize buffer\. Requested: (\d+), available (\d+), missing: (\d+)",
        text,
    )

    result: dict = {
        "board": "Raspberry Pi Pico 2 / RP2350",
        "firmware_build_type": (
            firmware_provenance.group(1) if firmware_provenance else None
        ),
        "system_clock_hz": (
            int(firmware_provenance.group(2)) if firmware_provenance else None
        ),
        "configured_arena_bytes": args.configured_arena_bytes,
        "arena_used_bytes": int(arena_used.group(1)) if arena_used else None,
        "fixture_samples_expected": int(fixture["sample_count"]),
        "fixture_samples_completed": len(matches),
        "status": "success" if len(matches) == fixture["sample_count"] and timing else "failed_or_incomplete",
        "allocation_failure": None,
        "metrics": None,
        "local_int8_metrics_on_fixture": None,
        "board_vs_local_int8": None,
        "timing": None,
        "serial_log": str(args.log.resolve()),
    }
    if allocation_failure:
        result["allocation_failure"] = {
            "requested_bytes": int(allocation_failure.group(1)),
            "available_bytes": int(allocation_failure.group(2)),
            "missing_bytes": int(allocation_failure.group(3)),
        }
    if matches:
        board = np.asarray([float(item[2]) for item in matches], dtype=np.float64)
        target = np.asarray([float(item[3]) for item in matches], dtype=np.float64)
        result["metrics"] = regression_metrics(target, board)
        local_rows = list(csv.DictReader(args.predictions.open(encoding="utf-8")))
        local = np.asarray(
            [
                float(local_rows[index]["int8_prediction_soh"])
                for index in fixture["source_test_indices"][: len(matches)]
            ],
            dtype=np.float64,
        )
        delta = board - local
        result["local_int8_metrics_on_fixture"] = regression_metrics(target, local)
        result["board_vs_local_int8"] = {
            "RMSE": float(np.sqrt(np.mean(np.square(delta)))),
            "MAE": float(np.mean(np.abs(delta))),
            "MaxAbs": float(np.max(np.abs(delta))),
        }
    if timing:
        result["timing"] = {
            "runs": int(timing.group(1)),
            "cold_us": int(timing.group(2)),
            "mean_us": float(timing.group(3)),
            "sd_us": float(timing.group(4)),
            "p50_us": int(timing.group(5)),
            "p95_us": int(timing.group(6)),
            "max_us": int(timing.group(7)),
        }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
