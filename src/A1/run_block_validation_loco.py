"""Current A1: condition-stratified LOCO with within-cell block validation.

The held-out target cell is test-only. Each of the three source cells supplies
chronological train and validation blocks covering early/middle/late life.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
from src.A1.run_nested_loco import (
    MODEL_NAME,
    _forward_prediction,
    create_a1_model,
    evaluate_a1_loss,
    make_a1_loader,
    train_a1_model,
)
from src.common import (
    DEFAULT_ADAM_BETAS,
    add_common_args,
    create_run_dir,
    ensure_dir,
    get_device,
    metrics_by_record,
    regression_metrics,
    save_json,
    save_training_outputs,
    set_seed,
    write_csv,
)


EXPERIMENT = "A1_CALCE_8cell_condition_stratified_block_validation_LOCO"


def _scaled(cells, names, split, scalers):
    raw = stack_cells(cells, names, split)
    return (*apply_a1_scalers(*raw[:4], scalers), raw[4])


def train_with_cell_macro_validation(
    model: nn.Module,
    train_loader,
    validation_loaders: dict[str, object],
    device: torch.device,
    epochs: int,
    lr: float,
) -> list[dict]:
    """Select epochs by equal-cell macro MSE, not sample-weighted MSE."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
        betas=DEFAULT_ADAM_BETAS,
    )
    model.to(device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for voltage, current, duration, target in train_loader:
            voltage, current, duration, target = (
                voltage.to(device), current.to(device), duration.to(device), target.to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(_forward_prediction(model, voltage, current, duration), target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * int(target.shape[0])
            sample_count += int(target.shape[0])
        row = {"epoch": epoch, "train_loss": loss_sum / sample_count}
        cell_losses = []
        for cell_id, loader in sorted(validation_loaders.items()):
            value = evaluate_a1_loss(model, loader, device)
            row[f"val_loss_{cell_id}"] = value
            cell_losses.append(value)
        row["macro_val_loss"] = float(np.mean(cell_losses))
        history.append(row)
    return history


def evaluate(model, cells, test_cell, scalers, batch_size, device):
    data = _scaled(cells, (test_cell,), None, scalers)
    loader = make_a1_loader(*data[:4], batch_size, shuffle=False)
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for voltage, current, duration, target in loader:
            prediction = _forward_prediction(
                model, voltage.to(device), current.to(device), duration.to(device)
            )
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(target.numpy())
    pred = scalers["y"].inverse_transform(np.vstack(predictions))
    target = scalers["y"].inverse_transform(np.vstack(targets))
    rows = []
    for metadata, y_true, y_pred in zip(data[4], target.reshape(-1), pred.reshape(-1)):
        rows.append({**metadata, "target": float(y_true), "prediction": float(y_pred),
                     "error": float(y_pred - y_true)})
    return regression_metrics(target, pred), rows


def write_selection_manifest(path: Path, cells, source_cells):
    rows = []
    for cell_id in source_cells:
        record = cells[cell_id]
        for item in record.metadata:
            if item["source_split"] in {"train", "validation", "purged"}:
                rows.append({"outer_role": "source", **item})
    write_csv(path, rows)


def dataset_hash(root: Path) -> str:
    return hashlib.sha256((root / "dataset_manifest.json").read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT)
    add_common_args(parser)
    parser.set_defaults(points=256, window=12)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_MODEL_INPUT_ROOT)
    parser.add_argument("--outer-folds", nargs="*", default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.patience:
        raise ValueError("Use --patience 0; epoch selection requires complete histories")

    root = args.data_root.expanduser().resolve()
    cells = load_calce_cells(root)
    all_folds = build_stratified_fold_plan()
    rows = validate_outer_plan(cells, all_folds)
    print("A1 model_input_a1_v2 validation passed:")
    for row in rows:
        print(f"  {row['cell']}: windows={row['windows']}, train={row['train']}, "
              f"validation={row['validation']}, purged={row['purged']}, ratio={row['ratio']:.3f}")
    folds = all_folds
    if args.outer_folds:
        unknown = sorted(set(args.outer_folds) - set(all_folds))
        if unknown:
            raise ValueError(f"Unknown outer folds: {unknown}")
        folds = {name: all_folds[name] for name in args.outer_folds}
    for fold in folds.values():
        print(f"  {fold.name}: source={list(fold.source_cells)}, test-only={fold.test_cell}")
    if args.validate_only:
        print("Validation-only mode complete; no training output was created.")
        return

    device = get_device()
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Mamba training; use --validate-only locally")
    out_dir = create_run_dir(
        "A1", "calce_8cell_condition_stratified_block_validation_loco",
        args.seed, output_dir=args.output_dir,
    )
    save_json(out_dir / "A1_run_manifest.json", {
        "experiment": EXPERIMENT, "data_root": str(root),
        "dataset_manifest_sha256": dataset_hash(root), "seed": args.seed,
        "selected_outer_folds": list(folds),
        "all_formal_outer_folds_selected": set(folds) == set(all_folds),
    })
    summary = []
    for fold in folds.values():
        fold_dir = ensure_dir(out_dir / fold.name)
        write_selection_manifest(fold_dir / "selection_split_manifest.csv", cells, fold.source_cells)

        train_raw = stack_cells(cells, fold.source_cells, "train")
        selection_scalers = fit_a1_scalers(*train_raw[:3])
        train_data = _scaled(cells, fold.source_cells, "train", selection_scalers)
        train_loader = make_a1_loader(*train_data[:4], args.batch_size, shuffle=True)
        validation_loaders = {}
        for cell_id in fold.source_cells:
            data = _scaled(cells, (cell_id,), "validation", selection_scalers)
            validation_loaders[cell_id] = make_a1_loader(*data[:4], args.batch_size, shuffle=False)
        set_seed(args.seed)
        selection_model = create_a1_model()
        history = train_with_cell_macro_validation(
            selection_model, train_loader, validation_loaders, device, args.epochs, args.lr
        )
        write_csv(fold_dir / "block_validation_history.csv", history)
        selected = min(history, key=lambda item: item["macro_val_loss"])
        selected_epoch = int(selected["epoch"])

        # Final source-only refit uses every source window after epoch selection.
        # The held-out target cell remains completely untouched.
        final_raw = stack_cells(cells, fold.source_cells, None)
        final_scalers = fit_a1_scalers(*final_raw[:3])
        final_data = _scaled(cells, fold.source_cells, None, final_scalers)
        final_loader = make_a1_loader(*final_data[:4], args.batch_size, shuffle=True)
        set_seed(args.seed)
        model = create_a1_model()
        model, final_history = train_a1_model(
            model, final_loader, device, selected_epoch, args.lr
        )
        metrics, predictions = evaluate(
            model, cells, fold.test_cell, final_scalers, args.batch_size, device
        )
        save_training_outputs(
            fold_dir, f"{MODEL_NAME}_{fold.test_cell}", model, final_history,
            metrics, metrics_by_record(predictions), predictions, final_scalers,
            {
                "experiment": EXPERIMENT, "outer_fold": fold.name,
                "condition_group": fold.condition_group,
                "source_cells": list(fold.source_cells), "test_record": fold.test_cell,
                "selection": {
                    "split": "each source cell: lifecycle-stratified contiguous blocks, nominal 2:1",
                    "boundary_purge_windows": 11,
                    "criterion": "equal-source-cell macro validation MSE",
                    "maximum_epochs": args.epochs, "selected_epoch": selected_epoch,
                    "selected_macro_validation_loss": float(selected["macro_val_loss"]),
                },
                "final_training": {
                    "records": list(fold.source_cells), "epochs": selected_epoch,
                    "from_scratch": True, "uses_all_source_windows_after_selection": True,
                },
                "test_isolation": "held-out cell absent from train, validation, scaler fitting, and epoch selection",
                "window": "12 retained cycles; adjacent observed-cycle gap <=5; deltas stored",
                "data_root": str(root), "dataset_manifest_sha256": dataset_hash(root),
                "device": str(device), "lr": args.lr, "batch_size": args.batch_size,
                "seed": args.seed,
            },
        )
        save_json(fold_dir / "selection_summary.json", {
            "fold": fold.name, "source_cells": list(fold.source_cells),
            "test_cell": fold.test_cell, "selected_epoch": selected_epoch,
            "macro_val_loss": float(selected["macro_val_loss"]),
        })
        summary.append({"fold": fold.name, "condition_group": fold.condition_group,
                        "test_record": fold.test_cell, "selected_epoch": selected_epoch,
                        "macro_val_loss": float(selected["macro_val_loss"]), **metrics})
        print(f"{fold.name}: selected_epoch={selected_epoch}, metrics={metrics}")

    metric_keys = [key for key in summary[0] if key not in {"fold", "condition_group", "test_record"}]
    result = list(summary)
    for group in CALCE_CONDITION_CELLS:
        group_rows = [row for row in summary if row["condition_group"] == group]
        if group_rows:
            result.append({"fold": "condition_cell_macro_mean", "condition_group": group,
                           "test_record": "", **{key: float(np.mean([row[key] for row in group_rows])) for key in metric_keys}})
    result.append({"fold": "selected_cells_macro_mean", "condition_group": "selected",
                   "test_record": "", **{key: float(np.mean([row[key] for row in summary])) for key in metric_keys}})
    write_csv(out_dir / "A1_block_validation_LOCO_summary.csv", result)
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
