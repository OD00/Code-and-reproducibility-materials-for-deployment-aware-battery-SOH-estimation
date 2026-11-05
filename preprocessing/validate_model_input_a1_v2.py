from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data/model_input_a1_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    manifest = json.loads((args.root / "dataset_manifest.json").read_text(encoding="utf-8"))
    expected = {}
    for line in (args.root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    summaries = []
    for relative, digest in expected.items():
        path = args.root / relative
        if sha256_file(path) != digest:
            raise RuntimeError(f"Checksum mismatch: {relative}")
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema_version"]) != "model_input_a1_v2":
                raise RuntimeError(f"Bad schema: {path}")
            rows = data["window_row_indices"]
            observed = data["observed_cycle_index"]
            gaps = np.diff(observed[rows], axis=1)
            if not np.all((gaps > 0) & (gaps <= 5)):
                raise RuntimeError(f"Invalid cycle gap: {path}")
            expected_delta = np.zeros(rows.shape, dtype=np.int16)
            expected_delta[:, 1:] = gaps
            if not np.array_equal(expected_delta, data["window_cycle_delta"]):
                raise RuntimeError(f"Cycle delta mismatch: {path}")
            roles = data["a1_source_split"]
            stages = data["a1_lifecycle_stage"]
            if not set(np.unique(roles)).issubset({0, 1, 2}):
                raise RuntimeError(f"Invalid split code: {path}")
            train_rows = set(rows[roles == 0].ravel().tolist())
            val_rows = set(rows[roles == 1].ravel().tolist())
            if train_rows.intersection(val_rows):
                raise RuntimeError(f"Train/validation leakage: {path}")
            for stage in range(3):
                mask = stages == stage
                if not np.any(roles[mask] == 0) or not np.any(roles[mask] == 1):
                    raise RuntimeError(f"Stage coverage failure: {path}, stage={stage}")
            targets = rows[:, -1]
            if not np.array_equal(targets, data["window_target_row_index"]):
                raise RuntimeError(f"Target row mismatch: {path}")
            if not np.allclose(data["window_target_soh_nominal"], data["label_soh_nominal"][targets]):
                raise RuntimeError(f"Target SOH mismatch: {path}")
            summaries.append(
                {
                    "cell_id": str(data["cell_id"]),
                    "windows": len(rows),
                    "train": int(np.sum(roles == 0)),
                    "validation": int(np.sum(roles == 1)),
                    "purged": int(np.sum(roles == 2)),
                    "ratio": float(np.sum(roles == 0) / np.sum(roles == 1)),
                }
            )
    with (args.root / "split_coverage_audit.csv").open(encoding="utf-8-sig", newline="") as handle:
        split_audit = list(csv.DictReader(handle))
    if len(split_audit) != len(summaries) * 9:
        raise RuntimeError("split_coverage_audit.csv row count mismatch")
    report = {
        "status": "VERIFIED",
        "schema_version": manifest["schema_version"],
        "cell_files": len(summaries),
        "checksums_verified": len(expected),
        "total_windows": sum(item["windows"] for item in summaries),
        "total_train": sum(item["train"] for item in summaries),
        "total_validation": sum(item["validation"] for item in summaries),
        "total_purged": sum(item["purged"] for item in summaries),
        "cells": summaries,
    }
    (args.root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
