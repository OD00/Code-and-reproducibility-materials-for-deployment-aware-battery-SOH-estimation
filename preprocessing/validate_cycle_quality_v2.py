from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np


INDEX_NAMES = {"CALCE": "calce_cycle_dict.pkl", "TJU": "tju_cycle_dict.pkl"}


def load(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def segments(cycle: dict):
    for phase in ("charge", "discharge"):
        yield from cycle[phase]["segments"]
    yield from cycle.get("rest_segments", [])
    yield from cycle.get("unknown_segments", [])


def validate_dataset(old_root: Path, new_root: Path, dataset: str) -> dict:
    index_name = INDEX_NAMES[dataset]
    old_index = load(old_root / dataset / index_name)
    new_index = load(new_root / dataset / index_name)
    if set(old_index["cells"]) != set(new_index["cells"]):
        raise RuntimeError(f"{dataset}: cell set changed")

    checksum_expected = {}
    for line in (new_root / dataset / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksum_expected[relative] = digest
    for relative, digest in checksum_expected.items():
        if sha256(new_root / dataset / relative) != digest:
            raise RuntimeError(f"{dataset}: checksum mismatch: {relative}")

    cycles_checked = 0
    points_checked = 0
    corrected_txt_segments = 0
    finite_labels = []
    valid_labels = 0
    for cell_id in old_index["cells"]:
        old_cell = load(old_root / dataset / old_index["cells"][cell_id]["file"])
        new_cell = load(new_root / dataset / new_index["cells"][cell_id]["file"])
        if list(old_cell["cycles"]) != list(new_cell["cycles"]):
            raise RuntimeError(f"{dataset}/{cell_id}: cycle identities changed")
        formats = {item["source_id"]: item["file_format"] for item in old_cell["source_registry"]}
        for cycle_key in old_cell["cycles"]:
            old_cycle = old_cell["cycles"][cycle_key]
            new_cycle = new_cell["cycles"][cycle_key]
            if new_cycle["quality"].get("interpolated") is not False:
                raise RuntimeError(f"{dataset}/{cell_id}/{cycle_key}: interpolation marker invalid")
            old_segments = list(segments(old_cycle))
            new_segments = list(segments(new_cycle))
            if len(old_segments) != len(new_segments):
                raise RuntimeError(f"{dataset}/{cell_id}/{cycle_key}: segment count changed")
            source_format = formats[new_cycle["source"]["source_id"]]
            time_scale = 60.0 if dataset == "CALCE" and source_format == "txt" else 1.0
            for old_segment, new_segment in zip(old_segments, new_segments):
                for field in ("voltage_v", "current_a"):
                    old_values = old_segment["data"].get(field)
                    new_values = new_segment["data"].get(field)
                    if old_values is None and new_values is None:
                        continue
                    if not np.array_equal(old_values, new_values, equal_nan=True):
                        raise RuntimeError(
                            f"{dataset}/{cell_id}/{cycle_key}: raw {field} changed"
                        )
                    points_checked += len(old_values)
                old_time = old_segment["data"].get("test_time_s")
                new_time = new_segment["data"].get("test_time_s")
                if old_time is not None and new_time is not None:
                    expected = np.asarray(old_time) * time_scale
                    if not np.allclose(expected, new_time, rtol=0, atol=1e-8, equal_nan=True):
                        raise RuntimeError(
                            f"{dataset}/{cell_id}/{cycle_key}: time correction mismatch"
                        )
                    if time_scale == 60.0:
                        corrected_txt_segments += 1
            label = new_cycle["targets"]["capacity_label_ah"]
            if label is not None and math.isfinite(label):
                finite_labels.append(float(label))
            if new_cycle["quality"]["valid_capacity_label"]:
                if new_cycle["targets"]["capacity_label_qc_ah"] is None:
                    raise RuntimeError(f"{dataset}/{cell_id}/{cycle_key}: missing valid QC label")
                valid_labels += 1
            elif new_cycle["targets"]["capacity_label_qc_ah"] is not None:
                raise RuntimeError(f"{dataset}/{cell_id}/{cycle_key}: invalid QC label retained")
            cycles_checked += 1

    with (new_root / dataset / "cycle_qc_table.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        csv_rows = sum(1 for _ in csv.DictReader(handle))
    if csv_rows != cycles_checked:
        raise RuntimeError(f"{dataset}: QC CSV row count mismatch")
    return {
        "dataset": dataset,
        "status": "VERIFIED",
        "cells": len(old_index["cells"]),
        "cycles_checked": cycles_checked,
        "raw_voltage_current_values_checked": points_checked,
        "corrected_txt_segments": corrected_txt_segments,
        "finite_integrated_labels": len(finite_labels),
        "valid_qc_labels": valid_labels,
        "capacity_median_ah": float(np.median(finite_labels)),
        "checksums_verified": len(checksum_expected),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old", type=Path, default=Path(__file__).resolve().parents[1] / "data/cycle_dict_v1"
    )
    parser.add_argument(
        "--new", type=Path, default=Path(__file__).resolve().parents[1] / "data/cycle_dict_qc_v2"
    )
    args = parser.parse_args()
    reports = [validate_dataset(args.old, args.new, dataset) for dataset in INDEX_NAMES]
    report = {"status": "VERIFIED", "datasets": reports}
    (args.new / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
