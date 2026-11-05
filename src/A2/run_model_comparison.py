"""Current A2: architecture comparison on the exact current A1 protocol.

Only the neural-network architecture changes.  Data, eight outer folds,
source-cell block validation, duration input, optimizer, seed and final refit
rules are shared with ``src.A1.run_block_validation_loco``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.A1.model_input_a1_v2 import (
    CALCE_CONDITION_CELLS,
    DEFAULT_MODEL_INPUT_ROOT,
    apply_a1_scalers,
    build_stratified_fold_plan,
    fit_a1_scalers,
    load_calce_cells,
    stack_cells,
    validate_outer_plan,
)
from src.A1.run_block_validation_loco import (
    _scaled,
    dataset_hash,
    evaluate,
    train_with_cell_macro_validation,
    write_selection_manifest,
)
from src.A1.run_nested_loco import make_a1_loader, train_a1_model
from src.common import (
    add_common_args,
    create_run_dir,
    ensure_dir,
    get_device,
    metrics_by_record,
    save_json,
    save_training_outputs,
    set_seed,
    write_csv,
)
from src.models import CnnLSTMControl, CnnMambaImproved, CnnTransformer, ResNet1d


FORMAL_SEEDS = (42, 123, 2026)
MODEL_NAMES = (
    "CnnTransformer",
    "CnnLSTMControl",
    "ResNet1d",
    "CnnMambaImproved",
)
EXPERIMENTAL_MODEL_NAMES = (
    "CnnMambaSimpleCNN",
    "CnnLSTMSimpleCNN",
)
MULTI_MODEL_COMPARISON = tuple((name, name) for name in MODEL_NAMES)
REFERENCE_MODEL_NAME = "CnnMambaImproved"
EXPERIMENT = "A2_CALCE_A1_matched_block_validation_model_comparison"


def create_a2_model(name: str):
    """Create every comparison model with the same duration information."""

    if name == "CnnTransformer":
        return CnnTransformer(
            d_model=64,
            nhead=4,
            num_layers=2,
            output_dim=1,
            use_duration_fusion=True,
        )
    if name == "CnnLSTMControl":
        return CnnLSTMControl(
            d_model=64,
            hidden_size=128,
            output_dim=1,
            use_duration_fusion=True,
        )
    if name == "ResNet1d":
        return ResNet1d(output_dim=1, use_duration_fusion=True)
    if name == "CnnMambaImproved":
        return CnnMambaImproved(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
            output_dim=1,
            use_duration_fusion=True,
        )
    if name == "CnnMambaSimpleCNN":
        return CnnMambaImproved(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
            output_dim=1,
            use_duration_fusion=True,
            use_simple_cnn_frontend=True,
        )
    if name == "CnnLSTMSimpleCNN":
        return CnnLSTMControl(
            d_model=64,
            hidden_size=128,
            output_dim=1,
            use_duration_fusion=True,
            use_simple_cnn_frontend=True,
        )
    raise ValueError(f"Unknown A2 model: {name}")


def build_a2_outer_folds(*_unused):
    """Compatibility helper returning the exact current A1 outer plan."""

    return build_stratified_fold_plan()


def _validate_model_set(names: tuple[str, ...]) -> list[dict]:
    rows = []
    for name in names:
        model = create_a2_model(name)
        if not getattr(model, "use_duration_fusion", False):
            raise ValueError(f"{name} was created without duration fusion")
        rows.append(
            {
                "model": name,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "trainable_parameter_count": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT)
    add_common_args(parser)
    parser.set_defaults(points=256, window=12)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_MODEL_INPUT_ROOT)
    parser.add_argument("--outer-folds", nargs="*", default=None)
    parser.add_argument(
        "--models",
        nargs="*",
        choices=MODEL_NAMES + EXPERIMENTAL_MODEL_NAMES,
        default=None,
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.patience:
        raise ValueError("Use --patience 0; every model needs a complete epoch history")

    root = args.data_root.expanduser().resolve()
    cells = load_calce_cells(root)
    all_folds = build_stratified_fold_plan()
    split_rows = validate_outer_plan(cells, all_folds)
    folds = all_folds
    if args.outer_folds:
        unknown = sorted(set(args.outer_folds) - set(all_folds))
        if unknown:
            raise ValueError(f"Unknown outer folds: {unknown}")
        folds = {name: all_folds[name] for name in args.outer_folds}
    model_names = tuple(args.models or MODEL_NAMES)
    model_rows = _validate_model_set(model_names)

    print("A2 protocol validation passed: exact A1 data and outer folds")
    print(f"  data_root={root}")
    print(f"  folds={list(folds)}")
    for row in split_rows:
        print(
            f"  {row['cell']}: train={row['train']}, validation={row['validation']}, "
            f"purged={row['purged']}"
        )
    for row in model_rows:
        print(f"  {row['model']}: parameters={row['parameter_count']}")
    if args.validate_only:
        print("Validation-only mode complete; no training output was created.")
        return

    device = get_device()
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the Mamba comparison; use --validate-only locally")
    out_dir = create_run_dir(
        "A2",
        "calce_A1_matched_block_validation_model_comparison",
        args.seed,
        output_dir=args.output_dir,
    )
    save_json(
        out_dir / "A2_run_manifest.json",
        {
            "experiment": EXPERIMENT,
            "data_root": str(root),
            "dataset_manifest_sha256": dataset_hash(root),
            "seed": args.seed,
            "formal_seed_set": list(FORMAL_SEEDS),
            "models": list(model_names),
            "selected_outer_folds": list(folds),
            "all_formal_outer_folds_selected": set(folds) == set(all_folds),
            "comparison_control": (
                "Only architecture changes; A1 v2 data, folds, inputs, label, "
                "scalers, optimizer, epoch budget and selection rule are fixed."
            ),
        },
    )

    fold_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    for model_name in model_names:
        parameter_count = next(
            row["parameter_count"] for row in model_rows if row["model"] == model_name
        )
        model_fold_rows = []
        for fold in folds.values():
            fold_dir = ensure_dir(out_dir / model_name / fold.name)
            write_selection_manifest(
                fold_dir / "selection_split_manifest.csv", cells, fold.source_cells
            )

            train_raw = stack_cells(cells, fold.source_cells, "train")
            selection_scalers = fit_a1_scalers(*train_raw[:3])
            train_data = _scaled(cells, fold.source_cells, "train", selection_scalers)
            train_loader = make_a1_loader(
                *train_data[:4], args.batch_size, shuffle=True
            )
            validation_loaders = {}
            for cell_id in fold.source_cells:
                data = _scaled(cells, (cell_id,), "validation", selection_scalers)
                validation_loaders[cell_id] = make_a1_loader(
                    *data[:4], args.batch_size, shuffle=False
                )

            set_seed(args.seed)
            selection_model = create_a2_model(model_name)
            history = train_with_cell_macro_validation(
                selection_model,
                train_loader,
                validation_loaders,
                device,
                args.epochs,
                args.lr,
            )
            write_csv(fold_dir / "block_validation_history.csv", history)
            selected = min(history, key=lambda item: item["macro_val_loss"])
            selected_epoch = int(selected["epoch"])

            final_raw = stack_cells(cells, fold.source_cells, None)
            final_scalers = fit_a1_scalers(*final_raw[:3])
            final_data = (
                *apply_a1_scalers(*final_raw[:4], final_scalers),
                final_raw[4],
            )
            final_loader = make_a1_loader(
                *final_data[:4], args.batch_size, shuffle=True
            )
            set_seed(args.seed)
            model = create_a2_model(model_name)
            model, final_history = train_a1_model(
                model, final_loader, device, selected_epoch, args.lr
            )
            metrics, predictions = evaluate(
                model, cells, fold.test_cell, final_scalers, args.batch_size, device
            )
            save_training_outputs(
                fold_dir,
                f"{model_name}_{fold.test_cell}",
                model,
                final_history,
                metrics,
                metrics_by_record(predictions),
                predictions,
                final_scalers,
                {
                    "experiment": EXPERIMENT,
                    "model_name": model_name,
                    "parameter_count": parameter_count,
                    "use_duration_fusion": True,
                    "use_simple_cnn_frontend": bool(
                        getattr(model, "use_simple_cnn_frontend", False)
                    ),
                    "outer_fold": fold.name,
                    "condition_group": fold.condition_group,
                    "source_cells": list(fold.source_cells),
                    "test_record": fold.test_cell,
                    "selection": {
                        "split": "same source-cell lifecycle blocks as current A1",
                        "boundary_purge_windows": 11,
                        "criterion": "equal-source-cell macro validation MSE",
                        "maximum_epochs": args.epochs,
                        "selected_epoch": selected_epoch,
                        "selected_macro_validation_loss": float(
                            selected["macro_val_loss"]
                        ),
                    },
                    "final_training": {
                        "records": list(fold.source_cells),
                        "epochs": selected_epoch,
                        "from_scratch": True,
                        "uses_all_source_windows_after_selection": True,
                    },
                    "inputs": {
                        "voltage": "charge_voltage_v [B,12,256]",
                        "current": "charge_current_a [B,12,256]",
                        "duration": "charge_duration_s [B,12,1]",
                    },
                    "label": "window_target_soh_nominal, identity-scaled",
                    "test_isolation": (
                        "held-out cell absent from training, validation, scaler fitting "
                        "and epoch selection"
                    ),
                    "data_root": str(root),
                    "dataset_manifest_sha256": dataset_hash(root),
                    "device": str(device),
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "window": args.window,
                    "points": args.points,
                    "seed": args.seed,
                },
            )
            save_json(
                fold_dir / "selection_summary.json",
                {
                    "model": model_name,
                    "fold": fold.name,
                    "test_cell": fold.test_cell,
                    "selected_epoch": selected_epoch,
                    "macro_val_loss": float(selected["macro_val_loss"]),
                },
            )
            row = {
                "model": model_name,
                "parameter_count": parameter_count,
                "fold": fold.name,
                "condition_group": fold.condition_group,
                "test_record": fold.test_cell,
                "selected_epoch": selected_epoch,
                "macro_val_loss": float(selected["macro_val_loss"]),
                **metrics,
            }
            fold_rows.append(row)
            model_fold_rows.append(row)
            print(f"{model_name} {fold.name}: selected_epoch={selected_epoch}, metrics={metrics}")

        metric_keys = [
            key
            for key in model_fold_rows[0]
            if key
            not in {
                "model",
                "parameter_count",
                "fold",
                "condition_group",
                "test_record",
            }
        ]
        for group in CALCE_CONDITION_CELLS:
            rows = [row for row in model_fold_rows if row["condition_group"] == group]
            if rows:
                aggregate_rows.append(
                    {
                        "model": model_name,
                        "parameter_count": parameter_count,
                        "scope": group,
                        **{
                            key: float(np.mean([row[key] for row in rows]))
                            for key in metric_keys
                        },
                    }
                )
        aggregate_rows.append(
            {
                "model": model_name,
                "parameter_count": parameter_count,
                "scope": "selected_cells_macro_mean",
                **{
                    key: float(np.mean([row[key] for row in model_fold_rows]))
                    for key in metric_keys
                },
            }
        )

    reference = next(
        (
            row
            for row in aggregate_rows
            if row["model"] == REFERENCE_MODEL_NAME
            and row["scope"] == "selected_cells_macro_mean"
        ),
        None,
    )
    if reference is not None:
        for row in aggregate_rows:
            if row["scope"] == "selected_cells_macro_mean":
                row["Delta_R2_vs_CnnMambaImproved"] = row["R2"] - reference["R2"]
                row["Delta_RMSE_vs_CnnMambaImproved"] = (
                    row["RMSE"] - reference["RMSE"]
                )
    write_csv(out_dir / "A2_model_comparison_fold_metrics.csv", fold_rows)
    write_csv(out_dir / "A2_model_comparison_summary.csv", aggregate_rows)
    print(f"A2 model comparison complete. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
