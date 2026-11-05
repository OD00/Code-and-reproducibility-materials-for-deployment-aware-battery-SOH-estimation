"""Validate and aggregate the formal CALCE 1C frontend ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 123, 2026)
CELLS = ("CS2_35", "CS2_36", "CS2_37", "CS2_38")
FOLDS = tuple(f"CALCE_group_1C__test_{cell}" for cell in CELLS)
BASELINE_MODELS = ("CnnMambaImproved", "CnnLSTMControl")
MODIFIED_MODELS = ("CnnMambaSimpleCNN", "CnnLSTMSimpleCNN")
MODEL_PAIRS = {
    "CnnMamba": ("CnnMambaImproved", "CnnMambaSimpleCNN"),
    "CnnLSTM": ("CnnLSTMControl", "CnnLSTMSimpleCNN"),
}
PARAMETERS = {
    "CnnMambaImproved": 138026,
    "CnnMambaSimpleCNN": 71106,
    "CnnLSTMControl": 303210,
    "CnnLSTMSimpleCNN": 236290,
}
METRICS = ("R2", "MSE", "RMSE", "MAE", "MaxError", "Bias", "NRMSE_range")
EXPECTED_BATCH_SIZE = 64
EXPECTED_EPOCHS = 50
EXPECTED_LR = 0.0005211317875270432


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-run-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def seed_dir(root: Path, seed: int) -> Path:
    candidates = (root / f"seed_{seed}", root / f"a2_block_validation_seed{seed}")
    found = [path for path in candidates if path.is_dir()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"Expected exactly one directory for seed {seed} under {root}; found {found}"
        )
    return found[0]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_config(path: Path, model: str, seed: int, modified: bool) -> str:
    config = read_json(path)
    if int(config["seed"]) != seed:
        raise ValueError(f"Seed mismatch in {path}")
    if int(config["batch_size"]) != EXPECTED_BATCH_SIZE:
        raise ValueError(f"Expected batch size {EXPECTED_BATCH_SIZE} in {path}")
    if int(config["selection"]["maximum_epochs"]) != EXPECTED_EPOCHS:
        raise ValueError(f"Expected {EXPECTED_EPOCHS} selection epochs in {path}")
    if not np.isclose(float(config["lr"]), EXPECTED_LR, rtol=0.0, atol=1e-15):
        raise ValueError(f"Learning-rate mismatch in {path}")
    if bool(config.get("use_simple_cnn_frontend", False)) is not modified:
        raise ValueError(f"Frontend flag mismatch for {model} in {path}")
    if config["test_record"] not in CELLS:
        raise ValueError(f"Unexpected test cell in {path}")
    return str(config["dataset_manifest_sha256"])


def load_runs(root: Path, models: tuple[str, ...], source: str) -> tuple[pd.DataFrame, set[str]]:
    frames: list[pd.DataFrame] = []
    hashes: set[str] = set()
    for seed in SEEDS:
        run_dir = seed_dir(root, seed)
        metric_path = run_dir / "A2_model_comparison_fold_metrics.csv"
        frame = pd.read_csv(metric_path)
        frame = frame.loc[
            frame["model"].isin(models)
            & frame["fold"].isin(FOLDS)
            & frame["test_record"].isin(CELLS)
        ].copy()
        expected_pairs = {(model, cell) for model in models for cell in CELLS}
        actual_pairs = set(zip(frame["model"], frame["test_record"]))
        if actual_pairs != expected_pairs or len(frame) != len(expected_pairs):
            raise ValueError(
                f"Incomplete or duplicate model/cell rows for seed {seed}: "
                f"missing={sorted(expected_pairs - actual_pairs)}, "
                f"extra={sorted(actual_pairs - expected_pairs)}"
            )
        frame["seed"] = seed
        frame["source"] = source
        for row in frame.itertuples(index=False):
            if int(row.parameter_count) != PARAMETERS[row.model]:
                raise ValueError(f"Parameter count mismatch for {row.model}")
            config_path = (
                run_dir
                / row.model
                / row.fold
                / f"{row.model}_{row.test_record}_config.json"
            )
            hashes.add(
                validate_config(
                    config_path,
                    row.model,
                    seed,
                    modified=row.model in MODIFIED_MODELS,
                )
            )
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    for metric in METRICS:
        result[metric] = pd.to_numeric(result[metric], errors="raise")
    if result[list(METRICS)].isna().any().any():
        raise ValueError(f"NaN metrics found in {source} results")
    result["AbsBias"] = result["Bias"].abs()
    return result, hashes


def flatten_aggregate(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    aggregate = frame.groupby(group_columns)[list(METRICS) + ["AbsBias"]].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    return aggregate.reset_index()


def main() -> None:
    args = parse_args()
    new_root = args.new_run_root.resolve()
    baseline_root = args.baseline_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    baseline, baseline_hashes = load_runs(baseline_root, BASELINE_MODELS, "baseline")
    modified, modified_hashes = load_runs(new_root, MODIFIED_MODELS, "modified")
    all_hashes = baseline_hashes | modified_hashes
    if len(all_hashes) != 1:
        raise ValueError(f"Results use different dataset manifests: {sorted(all_hashes)}")

    raw = pd.concat([baseline, modified], ignore_index=True)
    if len(raw) != 4 * 4 * 3:
        raise ValueError(f"Expected 48 formal rows, found {len(raw)}")
    raw.to_csv(output_dir / "fold_metrics_all_models.csv", index=False)

    # The experimental unit for uncertainty is the independent training seed.
    per_seed = (
        raw.groupby(["model", "parameter_count", "seed"], as_index=False)[
            list(METRICS) + ["AbsBias"]
        ]
        .mean()
        .sort_values(["model", "seed"])
    )
    per_seed.to_csv(output_dir / "metrics_per_seed_loco_macro.csv", index=False)
    aggregate = flatten_aggregate(per_seed, ["model", "parameter_count"])
    aggregate["n_training_seeds"] = len(SEEDS)
    aggregate["folds_per_seed"] = len(CELLS)
    aggregate.to_csv(output_dir / "metrics_three_seed_aggregate.csv", index=False)

    by_cell = flatten_aggregate(raw, ["model", "parameter_count", "test_record"])
    by_cell["n_training_seeds"] = len(SEEDS)
    by_cell.to_csv(output_dir / "metrics_by_test_cell.csv", index=False)

    paired_frames = []
    for family, (baseline_name, modified_name) in MODEL_PAIRS.items():
        left = raw.loc[raw["model"] == baseline_name].copy()
        right = raw.loc[raw["model"] == modified_name].copy()
        paired = left.merge(
            right,
            on=["seed", "fold", "condition_group", "test_record"],
            suffixes=("_baseline", "_modified"),
            validate="one_to_one",
        )
        paired.insert(0, "family", family)
        for metric in METRICS + ("AbsBias",):
            paired[f"Delta_{metric}_modified_minus_baseline"] = (
                paired[f"{metric}_modified"] - paired[f"{metric}_baseline"]
            )
        paired["parameter_reduction"] = (
            paired["parameter_count_baseline"] - paired["parameter_count_modified"]
        )
        paired["parameter_reduction_fraction"] = 1.0 - (
            paired["parameter_count_modified"] / paired["parameter_count_baseline"]
        )
        paired_frames.append(paired)
    paired_all = pd.concat(paired_frames, ignore_index=True)
    if len(paired_all) != 2 * 4 * 3:
        raise ValueError(f"Expected 24 paired rows, found {len(paired_all)}")
    paired_all.to_csv(output_dir / "paired_fold_differences.csv", index=False)

    delta_columns = [column for column in paired_all if column.startswith("Delta_")]
    paired_per_seed = (
        paired_all.groupby(["family", "seed"], as_index=False)[delta_columns]
        .mean()
        .sort_values(["family", "seed"])
    )
    paired_per_seed.to_csv(output_dir / "paired_differences_per_seed.csv", index=False)
    paired_aggregate = paired_per_seed.groupby("family")[delta_columns].agg(["mean", "std"])
    paired_aggregate.columns = [
        f"{column}_{stat}" for column, stat in paired_aggregate.columns
    ]
    paired_aggregate = paired_aggregate.reset_index()
    paired_aggregate["n_training_seeds"] = len(SEEDS)
    paired_aggregate.to_csv(output_dir / "paired_differences_three_seed_aggregate.csv", index=False)

    lines = [
        "# CALCE 1C frontend ablation formal summary",
        "",
        "Aggregation: first macro-average four LOCO folds within each training seed, then report mean and sample standard deviation across seeds 42, 123, and 2026.",
        "",
        "| Model | Parameters | R2 | RMSE | MAE | MaxError | Bias |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.sort_values("model").itertuples(index=False):
        lines.append(
            f"| {row.model} | {int(row.parameter_count):,} | "
            f"{row.R2_mean:.5f} ± {row.R2_std:.5f} | "
            f"{row.RMSE_mean:.5f} ± {row.RMSE_std:.5f} | "
            f"{row.MAE_mean:.5f} ± {row.MAE_std:.5f} | "
            f"{row.MaxError_mean:.5f} ± {row.MaxError_std:.5f} | "
            f"{row.Bias_mean:+.5f} ± {row.Bias_std:.5f} |"
        )
    lines.extend(
        [
            "",
            "Only three independent training seeds were used. These descriptive results do not establish statistical significance.",
        ]
    )
    (output_dir / "formal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(aggregate.to_string(index=False))
    print(f"Validated dataset manifest: {next(iter(all_hashes))}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
