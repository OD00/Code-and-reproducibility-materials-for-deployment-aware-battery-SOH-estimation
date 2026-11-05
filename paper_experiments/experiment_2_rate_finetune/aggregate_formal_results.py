from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["RMSE", "MAE", "Bias", "MacroAbsCellBias", "R2", "MaxError"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Experiment 2 across target-cell subsets and training seeds."
    )
    parser.add_argument("run_root", type=Path, help="Directory containing seed_42, seed_123, and seed_2026.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    metric_files = sorted(run_root.glob("seed_*/metrics_summary.csv"))
    if not metric_files:
        raise FileNotFoundError(f"No seed_*/metrics_summary.csv found under {run_root}")

    raw = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    expected_seeds = {42, 123, 2026}
    actual_seeds = set(raw["training_seed"].astype(int).unique())
    if actual_seeds != expected_seeds:
        raise ValueError(f"Expected training seeds {sorted(expected_seeds)}, found {sorted(actual_seeds)}")
    if raw[METRICS].isna().any().any():
        raise ValueError("NaN found in formal metrics")

    raw.to_csv(run_root / "metrics_all_seeds.csv", index=False)

    # Target-subset seeds are correlated within one trained source model. Average
    # them first, then quantify variation across the three independent training seeds.
    per_training_seed = (
        raw.groupby(["model", "training_seed", "stage", "target_cell_count"], as_index=False)[METRICS]
        .mean()
        .sort_values(["model", "target_cell_count", "training_seed"])
    )
    per_training_seed.to_csv(run_root / "metrics_per_training_seed.csv", index=False)

    aggregate = (
        per_training_seed.groupby(["model", "stage", "target_cell_count"])[METRICS]
        .agg(["mean", "std"])
    )
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index().sort_values(["target_cell_count", "RMSE_mean", "model"])
    aggregate["n_training_seeds"] = 3
    aggregate.to_csv(run_root / "metrics_hierarchical_aggregate.csv", index=False)

    subset_summary = (
        raw.loc[raw["target_cell_count"].isin([1, 3, 6])]
        .groupby(["model", "target_cell_count", "subset_seed"], as_index=False)[METRICS]
        .mean()
        .sort_values(["model", "target_cell_count", "subset_seed"])
    )
    subset_summary.to_csv(run_root / "metrics_by_subset_selection.csv", index=False)

    lines = [
        "# Experiment 2 formal aggregate",
        "",
        "Aggregation: target-cell subset choices are averaged within each training seed, followed by mean and sample standard deviation across seeds 42, 123, and 2026.",
        "",
        "| Target cells | Model | RMSE | MAE | Bias | Abs. cell bias | R2 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.target_cell_count} | {row.model} | "
            f"{row.RMSE_mean:.4f} ± {row.RMSE_std:.4f} | "
            f"{row.MAE_mean:.4f} ± {row.MAE_std:.4f} | "
            f"{row.Bias_mean:.4f} ± {row.Bias_std:.4f} | "
            f"{row.MacroAbsCellBias_mean:.4f} ± {row.MacroAbsCellBias_std:.4f} | "
            f"{row.R2_mean:.4f} ± {row.R2_std:.4f} |"
        )
    (run_root / "formal_result_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
