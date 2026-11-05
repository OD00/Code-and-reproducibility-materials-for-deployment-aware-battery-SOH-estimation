from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


EXPECTED = {
    "CALCE": {
        "cells": 8,
        "cycles": 7704,
        "complete_pairs": 7638,
        "incomplete_cycles": 66,
        "interleaved_cycles": 0,
        "rows_in_cycles": 2337684,
    },
    "TJU": {
        "cells": 66,
        "cycles": 22916,
        "complete_pairs": 22915,
        "incomplete_cycles": 1,
        "interleaved_cycles": 11,
        "rows_in_cycles": 19235196,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def verify_dataset(root: Path, dataset: str) -> dict:
    dataset_root = root / dataset
    index_name = "calce_cycle_dict.pkl" if dataset == "CALCE" else "tju_cycle_dict.pkl"
    index = load_pickle(dataset_root / index_name)
    json_index = json.loads((dataset_root / "dataset_index.json").read_text(encoding="utf-8"))
    if index != json_index:
        raise RuntimeError(f"{dataset}: pickle index and JSON index differ")

    report = json.loads((dataset_root / "validation_report.json").read_text(encoding="utf-8"))
    totals = report["totals"]
    if totals != EXPECTED[dataset]:
        raise RuntimeError(
            f"{dataset}: totals differ from the prior read-only boundary audit: "
            f"actual={totals}, expected={EXPECTED[dataset]}"
        )
    if len(index["cells"]) != totals["cells"]:
        raise RuntimeError(f"{dataset}: index cell count mismatch")

    summary_path = dataset_root / "cycle_summary.csv"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        summaries = list(csv.DictReader(handle))
    source_path = dataset_root / "source_registry.csv"
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sources = list(csv.DictReader(handle))
    included_source_rows = sum(
        int(row["row_count"])
        for row in sources
        if row["included_in_cycles"].lower() == "true"
    )
    unassigned_rows = sum(int(row["unassigned_rows"]) for row in summaries)
    if included_source_rows != totals["rows_in_cycles"] + unassigned_rows:
        raise RuntimeError(
            f"{dataset}: source coverage mismatch: included={included_source_rows}, "
            f"cycles={totals['rows_in_cycles']}, unassigned={unassigned_rows}"
        )

    checksum_lines = (dataset_root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    expected_checksums = {}
    for line in checksum_lines:
        digest, relative = line.split("  ", 1)
        expected_checksums[relative] = digest
    checked_files = 0
    checked_bytes = 0
    for relative, expected_digest in expected_checksums.items():
        path = dataset_root / Path(relative)
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise RuntimeError(f"{dataset}: checksum mismatch for {relative}")
        checked_files += 1
        checked_bytes += path.stat().st_size

    for cell_id, item in index["cells"].items():
        cell_path = dataset_root / item["file"]
        cell = load_pickle(cell_path)
        if cell["cell_id"] != cell_id or cell["dataset"] != dataset:
            raise RuntimeError(f"{dataset}: cell identity mismatch in {cell_path.name}")
        cycles = cell["cycles"]
        actual = [cycle["observed_cycle_index"] for cycle in cycles.values()]
        if actual != list(range(1, len(cycles) + 1)):
            raise RuntimeError(f"{dataset}: non-continuous observed indices in {cell_id}")

    return {
        "dataset": dataset,
        "status": "VERIFIED",
        "totals": totals,
        "included_source_rows": included_source_rows,
        "unassigned_rows": unassigned_rows,
        "checksum_files_verified": checked_files,
        "checksum_bytes_verified": checked_bytes,
        "checks": [
            "pickle index equals JSON index",
            "totals equal the independent read-only boundary audit",
            "included source row count equals cycle rows plus unassigned rows",
            "all recorded SHA-256 checksums match",
            "all cell dictionaries reload with continuous observed_cycle_index",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/cycle_dict_v1",
    )
    args = parser.parse_args()
    reports = [verify_dataset(args.root, dataset) for dataset in ("CALCE", "TJU")]
    combined = {
        "schema_version": "cycle_dict_v1",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "status": "VERIFIED",
        "execution_note": "Executed by the current Python runtime; see validator_python.",
        "validator_python": sys.version,
        "datasets": reports,
    }
    (args.root / "processing_report.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(combined, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
