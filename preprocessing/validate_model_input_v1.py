from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data/model_input_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_npz(path: Path, expected_points: int, expected_window: int) -> dict:
    with np.load(path, allow_pickle=False) as data:
        cell_id = str(data["cell_id"])
        dataset = str(data["dataset"])
        observed = data["observed_cycle_index"]
        capacity = data["capacity_ah"]
        soh = data["label_soh_nominal"]
        nominal = float(data["nominal_capacity_ah"])
        n = len(observed)
        if capacity.shape != (n, 1) or soh.shape != (n, 1):
            raise RuntimeError(f"{path}: label shape mismatch")
        if not np.all(np.diff(observed) > 0):
            raise RuntimeError(f"{path}: observed cycles are not strictly ordered")
        if not np.all(np.isfinite(capacity)) or not np.all(capacity > 0):
            raise RuntimeError(f"{path}: invalid capacity labels")
        if not np.allclose(soh, capacity / nominal, rtol=1e-6, atol=1e-7):
            raise RuntimeError(f"{path}: SOH and capacity labels disagree")

        for phase in ("charge", "discharge"):
            voltage = data[f"{phase}_voltage_v"]
            current = data[f"{phase}_current_a"]
            elapsed = data[f"{phase}_elapsed_time_s"]
            duration = data[f"{phase}_duration_s"]
            if voltage.shape != (n, expected_points):
                raise RuntimeError(f"{path}: {phase} voltage shape mismatch")
            if current.shape != voltage.shape or elapsed.shape != voltage.shape:
                raise RuntimeError(f"{path}: {phase} input shape mismatch")
            if duration.shape != (n, 1):
                raise RuntimeError(f"{path}: {phase} duration shape mismatch")
            for name, values in (
                ("voltage", voltage),
                ("current", current),
                ("elapsed", elapsed),
                ("duration", duration),
            ):
                if not np.all(np.isfinite(values)):
                    raise RuntimeError(f"{path}: non-finite {phase} {name}")
            if not np.allclose(elapsed[:, 0], 0.0, atol=1e-4):
                raise RuntimeError(f"{path}: {phase} time does not start at zero")
            if not np.all(np.diff(elapsed, axis=1) > 0):
                raise RuntimeError(f"{path}: {phase} time is not monotonic")
            if not np.allclose(
                elapsed[:, -1], duration[:, 0], rtol=2e-6, atol=1e-3
            ):
                raise RuntimeError(f"{path}: {phase} time endpoint mismatch")

        windows = data["window_row_indices"]
        targets = data["window_target_row_index"]
        if windows.ndim != 2 or windows.shape[1] != expected_window:
            raise RuntimeError(f"{path}: invalid window shape")
        if targets.shape != (len(windows),):
            raise RuntimeError(f"{path}: target-row shape mismatch")
        if len(windows):
            if np.any(windows < 0) or np.any(windows >= n):
                raise RuntimeError(f"{path}: window row outside cycle array")
            if not np.array_equal(windows[:, -1], targets):
                raise RuntimeError(f"{path}: window target is not final row")
            if not np.all(np.diff(observed[windows], axis=1) == 1):
                raise RuntimeError(f"{path}: a window crosses an excluded cycle")
            if not np.allclose(
                data["window_target_capacity_ah"], capacity[targets]
            ):
                raise RuntimeError(f"{path}: window capacity label mismatch")
            if not np.allclose(data["window_target_soh_nominal"], soh[targets]):
                raise RuntimeError(f"{path}: window SOH label mismatch")

    return {
        "dataset": dataset,
        "cell_id": cell_id,
        "cycles": n,
        "windows": int(len(windows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    manifest = json.loads(
        (args.root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    points = int(manifest["interpolation"]["points"])
    window = int(manifest["window"]["size"])

    expected_checksums: dict[str, str] = {}
    for line in (args.root / "checksums.sha256").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, relative = line.split("  ", 1)
        expected_checksums[relative] = digest
    for relative, expected in expected_checksums.items():
        actual = sha256_file(args.root / relative)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}")

    summaries = [
        validate_npz(path, points, window)
        for dataset in ("CALCE", "TJU")
        for path in sorted((args.root / dataset).glob("*.npz"))
    ]
    cycles = sum(item["cycles"] for item in summaries)
    windows = sum(item["windows"] for item in summaries)
    manifest_cycles = sum(
        int(item["valid_cycles"]) for item in manifest["datasets"].values()
    )
    manifest_windows = sum(
        int(item["valid_12_cycle_windows"])
        for item in manifest["datasets"].values()
    )
    if cycles != manifest_cycles or windows != manifest_windows:
        raise RuntimeError("Manifest totals do not match NPZ files")

    with (args.root / "cycle_index.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        index_count = sum(1 for _ in csv.DictReader(handle))
    if index_count != cycles:
        raise RuntimeError("cycle_index.csv row count mismatch")

    with (args.root / "interpolation_quality.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        quality_rows = list(csv.DictReader(handle))
    if len(quality_rows) != 2 * cycles:
        raise RuntimeError("interpolation_quality.csv row count mismatch")

    quality_summary = {}
    for phase in ("charge", "discharge"):
        rows = [row for row in quality_rows if row["phase"] == phase]
        quality_summary[phase] = {
            "voltage_mae_mv_p95": float(
                np.percentile([float(row["voltage_mae_mv"]) for row in rows], 95)
            ),
            "current_mae_ma_p95": float(
                np.percentile([float(row["current_mae_ma"]) for row in rows], 95)
            ),
            "current_integral_relative_error_pct_p95": float(
                np.percentile(
                    [
                        float(row["current_integral_relative_error_pct"])
                        for row in rows
                    ],
                    95,
                )
            ),
        }

    report = {
        "status": "VERIFIED",
        "root": str(args.root),
        "schema_version": manifest["schema_version"],
        "cell_files": len(summaries),
        "cycles": cycles,
        "windows": windows,
        "points": points,
        "window_size": window,
        "checksums_verified": len(expected_checksums),
        "interpolation_quality": quality_summary,
    }
    (args.root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
