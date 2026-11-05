"""Revised A1: condition-stratified nested LOCO on eight CALCE cells.

The main experiment uses charge voltage/current curves and charge duration from
``model_input_v1``.  The target is discharge-integrated capacity divided by the
dataset nominal capacity (``label_soh_nominal``).  No ablation is run here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.A1.model_input_v1 import (
    DEFAULT_MODEL_INPUT_ROOT,
    A1Fold,
    CellWindows,
    apply_a1_scalers,
    build_stratified_fold_plan,
    fit_a1_scalers,
    load_calce_cells,
    stack_cells,
)
from src.common import (
    DEFAULT_ADAM_BETAS,
    add_common_args,
    create_run_dir,
    ensure_dir,
    get_device,
    metrics_by_record,
    model_output_to_prediction,
    regression_metrics,
    save_json,
    save_training_outputs,
    set_seed,
    write_csv,
)
from src.models import CnnMambaImproved


MODEL_NAME = "CnnMambaImproved"


def aggregate_inner_histories(inner_histories):
    """Macro-average each epoch so every validation cell has equal weight."""

    lengths = {len(history) for history in inner_histories.values()}
    if len(lengths) != 1:
        raise ValueError(
            "All inner folds must cover the same number of epochs for "
            "macro validation selection."
        )

    rows = []
    total_epochs = lengths.pop()
    for position in range(total_epochs):
        row = {"epoch": position + 1}
        val_losses = []
        train_losses = []
        for validation_record, history in sorted(inner_histories.items()):
            history_row = history[position]
            if int(history_row["epoch"]) != position + 1:
                raise ValueError("Inner histories have inconsistent epochs.")
            val_loss = float(history_row["val_loss"])
            train_loss = float(history_row["train_loss"])
            row[f"val_loss_{validation_record}"] = val_loss
            row[f"train_loss_{validation_record}"] = train_loss
            val_losses.append(val_loss)
            train_losses.append(train_loss)
        row["macro_val_loss"] = float(np.mean(val_losses))
        row["macro_train_loss"] = float(np.mean(train_losses))
        rows.append(row)
    return rows


def create_a1_model() -> CnnMambaImproved:
    """Create only the duration-fusion main model; no ablation variants."""

    return CnnMambaImproved(
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        output_dim=1,
        use_duration_fusion=True,
    )


def make_a1_loader(
    voltage: np.ndarray,
    current: np.ndarray,
    duration: np.ndarray,
    soh: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(voltage).float(),
        torch.from_numpy(current).float(),
        torch.from_numpy(duration).float(),
        torch.from_numpy(soh).float(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def _forward_prediction(
    model: nn.Module,
    voltage: torch.Tensor,
    current: torch.Tensor,
    duration: torch.Tensor,
) -> torch.Tensor:
    output = model(voltage, current, duration)
    if isinstance(output, tuple):
        output = output[0]
    return model_output_to_prediction(output)


def evaluate_a1_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Return exact sample-level MSE in physical nominal-SOH space."""

    model.eval()
    squared_error_sum = 0.0
    element_count = 0
    with torch.no_grad():
        for voltage, current, duration, target in loader:
            voltage = voltage.to(device)
            current = current.to(device)
            duration = duration.to(device)
            target = target.to(device)
            prediction = _forward_prediction(model, voltage, current, duration)
            squared_error_sum += float(
                torch.sum((prediction - target) ** 2).detach().cpu()
            )
            element_count += int(target.numel())
    return squared_error_sum / element_count if element_count else math.nan


def train_a1_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    val_loader: DataLoader | None = None,
    restore_best: bool = False,
) -> tuple[nn.Module, list[dict]]:
    """Train the three-input A1 model for a fixed candidate epoch range."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
        betas=DEFAULT_ADAM_BETAS,
    )
    best_state = copy.deepcopy(model.state_dict()) if restore_best else None
    best_val = math.inf
    history = []
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for voltage, current, duration, target in train_loader:
            voltage = voltage.to(device)
            current = current.to(device)
            duration = duration.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward_prediction(model, voltage, current, duration)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            batch_samples = int(target.shape[0])
            loss_sum += float(loss.detach().cpu()) * batch_samples
            sample_count += batch_samples

        row = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count if sample_count else math.nan,
        }
        if val_loader is not None:
            row["val_loss"] = evaluate_a1_loss(model, val_loader, device)
            if restore_best and row["val_loss"] < best_val:
                best_val = float(row["val_loss"])
                best_state = copy.deepcopy(model.state_dict())
        history.append(row)

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _scaled_cell_data(
    cells: Mapping[str, CellWindows],
    names: Sequence[str],
    scalers: Mapping[str, object],
):
    voltage, current, duration, soh, metadata = stack_cells(cells, names)
    scaled = apply_a1_scalers(voltage, current, duration, soh, scalers)
    return (*scaled, metadata)


def evaluate_a1_model(
    model: nn.Module,
    cells: Mapping[str, CellWindows],
    test_cell: str,
    scalers: Mapping[str, object],
    batch_size: int,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    voltage, current, duration, soh, metadata = _scaled_cell_data(
        cells, (test_cell,), scalers
    )
    loader = make_a1_loader(
        voltage, current, duration, soh, batch_size, shuffle=False
    )
    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for xb1, xb2, xd, yb in loader:
            prediction = _forward_prediction(
                model,
                xb1.to(device),
                xb2.to(device),
                xd.to(device),
            )
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(yb.numpy())
    pred = scalers["y"].inverse_transform(np.vstack(predictions))
    target = scalers["y"].inverse_transform(np.vstack(targets))
    metrics = regression_metrics(target, pred)
    rows = []
    for item, target_value, prediction_value in zip(
        metadata, target.reshape(-1), pred.reshape(-1)
    ):
        row = dict(item)
        row["target"] = float(target_value)
        row["prediction"] = float(prediction_value)
        row["error"] = float(prediction_value - target_value)
        rows.append(row)
    return metrics, rows


def _write_split_manifest(
    path: Path,
    cells: Mapping[str, CellWindows],
    split_cells: Mapping[str, Sequence[str]],
) -> None:
    rows = []
    for split, names in split_cells.items():
        for name in names:
            rows.extend({"split": split, **item} for item in cells[name].metadata)
    write_csv(path, rows)


def _dataset_manifest_sha256(data_root: Path) -> str:
    manifest = Path(data_root) / "dataset_manifest.json"
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def validate_a1_plan(
    cells: Mapping[str, CellWindows],
    folds: Mapping[str, A1Fold],
    window: int,
    points: int,
) -> list[dict]:
    """Validate shapes and complete-cell isolation before any training."""

    rows = []
    for cell_id, record in sorted(cells.items()):
        if record.voltage.shape[1:] != (window, points):
            raise ValueError(
                f"{cell_id} has {record.voltage.shape[1:]}; "
                f"expected {(window, points)}"
            )
        rows.append(
            {
                "cell": cell_id,
                "condition_group": record.condition_group,
                "windows": record.sample_count,
                "window": window,
                "points": points,
            }
        )

    for fold in folds.values():
        if fold.test_cell in fold.source_cells:
            raise ValueError(f"Test leakage in {fold.name}")
        if len(fold.source_cells) != 3:
            raise ValueError(f"{fold.name} must have exactly three source cells")
        involved = (fold.test_cell, *fold.source_cells)
        if any(cells[name].condition_group != fold.condition_group for name in involved):
            raise ValueError(f"Mixed charging condition in {fold.name}")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "A1: two independent four-cell condition-stratified nested LOCO "
            "experiments using charge voltage, current, and duration."
        )
    )
    add_common_args(parser)
    parser.set_defaults(points=256, window=12)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_MODEL_INPUT_ROOT,
        help="Directory containing dataset_manifest.json and CALCE/*.npz.",
    )
    parser.add_argument(
        "--outer-folds",
        nargs="*",
        default=None,
        help="Optional fold names for a smoke run; omit for all eight folds.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate data and split isolation without training.",
    )
    args = parser.parse_args()
    if args.patience:
        raise ValueError(
            "Nested macro validation requires equal-length histories; "
            "use --patience 0."
        )

    data_root = args.data_root.expanduser().resolve()
    cells = load_calce_cells(data_root)
    all_folds = build_stratified_fold_plan()
    validation_rows = validate_a1_plan(cells, all_folds, args.window, args.points)
    print("A1 model_input_v1 validation passed:")
    for row in validation_rows:
        print(
            f"  {row['cell']}: {row['condition_group']}, "
            f"windows={row['windows']}, shape=[12,{row['points']}]"
        )

    if args.outer_folds:
        unknown = sorted(set(args.outer_folds).difference(all_folds))
        if unknown:
            raise ValueError(f"Unknown outer folds: {unknown}")
        folds = {name: all_folds[name] for name in args.outer_folds}
    else:
        folds = all_folds

    print("Outer folds:")
    for fold in folds.values():
        print(
            f"  {fold.name}: train/inner={list(fold.source_cells)}, "
            f"test={fold.test_cell}"
        )
    if args.validate_only:
        print("Validation-only mode complete; no training output was created.")
        return

    device = get_device()
    if device.type != "cuda":
        raise RuntimeError(
            "The installed mamba_ssm build requires CUDA for training. "
            "Use --validate-only for data checks on CPU."
        )
    set_seed(args.seed)
    out_dir = create_run_dir(
        "A1",
        "calce_8cell_condition_stratified_nested_loco",
        args.seed,
        output_dir=args.output_dir,
    )
    save_json(
        out_dir / "A1_run_manifest.json",
        {
            "experiment": "A1_CALCE_8cell_condition_stratified_nested_LOCO",
            "data_root": str(data_root),
            "dataset_manifest_sha256": _dataset_manifest_sha256(data_root),
            "selected_outer_folds": list(folds),
            "all_formal_outer_folds_selected": set(folds) == set(all_folds),
            "ablation_included": False,
            "seed": args.seed,
        },
    )

    summary_rows = []
    for fold in folds.values():
        outer_dir = ensure_dir(out_dir / fold.name)
        inner_histories = {}

        for validation_cell in fold.inner_validation_cells:
            inner_train_cells = tuple(
                cell for cell in fold.source_cells if cell != validation_cell
            )
            inner_dir = ensure_dir(outer_dir / f"val_{validation_cell}")
            _write_split_manifest(
                inner_dir / "selection_split_manifest.csv",
                cells,
                {"train": inner_train_cells, "val": (validation_cell,)},
            )
            train_raw = stack_cells(cells, inner_train_cells)
            scalers = fit_a1_scalers(train_raw[0], train_raw[1], train_raw[2])
            train_data = _scaled_cell_data(cells, inner_train_cells, scalers)
            val_data = _scaled_cell_data(cells, (validation_cell,), scalers)
            train_loader = make_a1_loader(
                *train_data[:4], args.batch_size, shuffle=True
            )
            val_loader = make_a1_loader(
                *val_data[:4], args.batch_size, shuffle=False
            )

            set_seed(args.seed)
            model = create_a1_model()
            _, history = train_a1_model(
                model,
                train_loader,
                device,
                epochs=args.epochs,
                lr=args.lr,
                val_loader=val_loader,
                restore_best=False,
            )
            inner_histories[validation_cell] = history
            write_csv(inner_dir / "selection_history.csv", history)
            best_row = min(history, key=lambda row: row["val_loss"])
            print(
                f"{fold.name} val_{validation_cell} complete: "
                f"best_epoch={int(best_row['epoch'])}, "
                f"SOH_MSE={float(best_row['val_loss']):.8f}"
            )

        macro_history = aggregate_inner_histories(inner_histories)
        write_csv(outer_dir / "inner_macro_validation_history.csv", macro_history)
        selected_row = min(macro_history, key=lambda row: row["macro_val_loss"])
        selected_epoch = int(selected_row["epoch"])

        _write_split_manifest(
            outer_dir / "final_split_manifest.csv",
            cells,
            {"train": fold.source_cells, "test": (fold.test_cell,)},
        )
        final_raw = stack_cells(cells, fold.source_cells)
        final_scalers = fit_a1_scalers(
            final_raw[0], final_raw[1], final_raw[2]
        )
        final_data = _scaled_cell_data(cells, fold.source_cells, final_scalers)
        final_loader = make_a1_loader(
            *final_data[:4], args.batch_size, shuffle=True
        )
        set_seed(args.seed)
        final_model = create_a1_model()
        final_model, final_history = train_a1_model(
            final_model,
            final_loader,
            device,
            epochs=selected_epoch,
            lr=args.lr,
        )
        metrics, predictions = evaluate_a1_model(
            final_model,
            cells,
            fold.test_cell,
            final_scalers,
            args.batch_size,
            device,
        )
        by_record = metrics_by_record(predictions)
        save_training_outputs(
            outer_dir,
            f"{MODEL_NAME}_{fold.test_cell}",
            final_model,
            final_history,
            metrics,
            by_record,
            predictions,
            final_scalers,
            {
                "experiment": "A1_CALCE_8cell_condition_stratified_nested_LOCO",
                "outer_fold": fold.name,
                "condition_group": fold.condition_group,
                "inner_validation_cells": list(fold.inner_validation_cells),
                "inner_selection": {
                    "maximum_epochs": args.epochs,
                    "criterion": (
                        "equal-cell macro mean of validation MSE in physical "
                        "nominal-SOH space at each epoch"
                    ),
                    "selected_epoch": selected_epoch,
                    "selected_macro_validation_loss": float(
                        selected_row["macro_val_loss"]
                    ),
                },
                "final_training": {
                    "records": list(fold.source_cells),
                    "epochs": selected_epoch,
                    "from_scratch": True,
                },
                "test_record": fold.test_cell,
                "model": {
                    "name": MODEL_NAME,
                    "use_duration_fusion": True,
                    "ablation": False,
                },
                "inputs": {
                    "voltage": "charge_voltage_v [B,12,256]",
                    "current": "charge_current_a [B,12,256]",
                    "duration": "charge_duration_s [B,12,1]",
                },
                "label": (
                    "window_target_soh_nominal: discharge-integrated capacity "
                    "Ah divided by CALCE nominal capacity 1.1 Ah; not clipped"
                ),
                "split": (
                    "outer leave-one-cell-out within one charge-rate group; "
                    "three source cells each serve once as complete inner "
                    "validation while the other two train"
                ),
                "normalization": (
                    "voltage/current/duration scalers fit on training cells "
                    "only in each inner/final fit; target is identity-scaled"
                ),
                "raw_cycle_overlap": (
                    "none across train/validation or train/test because all "
                    "partitions are disjoint physical cells"
                ),
                "purge_gap": "not applicable to complete-cell-disjoint splits",
                "data_root": str(data_root),
                "dataset_manifest_sha256": _dataset_manifest_sha256(data_root),
                "device": str(device),
                "lr": args.lr,
                "batch_size": args.batch_size,
                "window": args.window,
                "points": args.points,
                "seed": args.seed,
            },
        )
        save_json(
            outer_dir / "nested_selection_summary.json",
            {
                "outer_fold": fold.name,
                "condition_group": fold.condition_group,
                "test_record": fold.test_cell,
                "inner_validation_cells": list(fold.inner_validation_cells),
                "selected_epoch": selected_epoch,
                "selected_macro_validation_loss": float(
                    selected_row["macro_val_loss"]
                ),
            },
        )
        row = {
            "fold": fold.name,
            "condition_group": fold.condition_group,
            "test_record": fold.test_cell,
            "selected_epoch": selected_epoch,
            "macro_val_loss": float(selected_row["macro_val_loss"]),
            **metrics,
        }
        summary_rows.append(row)
        print(
            f"{fold.name} final complete: selected_epoch={selected_epoch}, "
            f"metrics={metrics}"
        )

    result_rows = list(summary_rows)
    for condition_group in sorted({row["condition_group"] for row in summary_rows}):
        group_rows = [
            row for row in summary_rows if row["condition_group"] == condition_group
        ]
        aggregate = {
            "fold": "condition_cell_macro_mean",
            "condition_group": condition_group,
            "test_record": "",
        }
        for key in ("selected_epoch", "macro_val_loss", *metrics.keys()):
            aggregate[key] = float(np.mean([row[key] for row in group_rows]))
        result_rows.append(aggregate)
    is_formal_full_run = set(folds) == set(all_folds)
    overall = {
        "fold": (
            "all_8_cells_macro_mean"
            if is_formal_full_run
            else "selected_smoke_folds_macro_mean"
        ),
        "condition_group": (
            "stratified_0.5C_and_1C"
            if is_formal_full_run
            else "partial_fold_selection"
        ),
        "test_record": "",
    }
    for key in ("selected_epoch", "macro_val_loss", *metrics.keys()):
        overall[key] = float(np.mean([row[key] for row in summary_rows]))
    result_rows.append(overall)
    write_csv(out_dir / "A1_nested_LOCO_summary.csv", result_rows)
    print(f"A1 nested LOCO complete. Cell-macro mean: {overall}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
