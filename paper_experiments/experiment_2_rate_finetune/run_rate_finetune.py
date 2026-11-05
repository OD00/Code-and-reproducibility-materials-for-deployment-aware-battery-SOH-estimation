"""Leakage-safe TJU 0.25C -> 0.5C full-parameter fine-tuning.

The 0.5C test cell is fixed and never used for scaling, model selection, or
fine-tuning. Target-cell subsets are nested within each subset seed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from paper_experiments.common.data import (
    StandardScalers,
    apply_scalers,
    fit_scalers,
    load_tju_cells,
    stack_cells,
    validate_roles,
)
from paper_experiments.common.models import ALL_MODEL_NAMES, MODEL_NAMES, create_model, unwrap_prediction

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data/model_input_v1/TJU"
DEFAULT_SOURCE_VALIDATION = "CY25-025_1-#7"
DEFAULT_TARGET_VALIDATION = "CY25-05_1-#1"
DEFAULT_TARGET_TEST = "CY25-05_1-#14"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_loader(arrays, batch_size: int, shuffle: bool, cell_balanced: bool = False):
    voltage, current, duration, target, cell_labels = arrays
    tensors = TensorDataset(
        torch.from_numpy(voltage),
        torch.from_numpy(current),
        torch.from_numpy(duration),
        torch.from_numpy(target),
    )
    sampler = None
    if cell_balanced:
        unique, counts = np.unique(cell_labels, return_counts=True)
        count_by_cell = dict(zip(unique.tolist(), counts.tolist()))
        weights = torch.as_tensor([1.0 / count_by_cell[cell] for cell in cell_labels], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=0)


def scaled_stack(cells, cell_ids, scalers: StandardScalers):
    voltage, current, duration, target, labels = stack_cells(cells, cell_ids)
    voltage, current, duration = apply_scalers(voltage, current, duration, scalers)
    return voltage, current, duration, target.astype(np.float32), labels


@torch.no_grad()
def predict(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets = [], []
    for voltage, current, duration, target in loader:
        output = unwrap_prediction(
            model(voltage.to(device), current.to(device), duration.to(device))
        )
        predictions.append(output.detach().cpu().numpy().reshape(-1))
        targets.append(target.numpy().reshape(-1))
    return np.concatenate(targets), np.concatenate(predictions)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    mse = float(np.mean(np.square(error), dtype=np.float64))
    denominator = float(np.sum(np.square(target - np.mean(target)), dtype=np.float64))
    r2 = float(1.0 - np.sum(np.square(error), dtype=np.float64) / denominator) if denominator > 1e-12 else 0.0
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(error), dtype=np.float64)),
        "Bias": float(np.mean(error, dtype=np.float64)),
        "R2": r2,
        "MaxError": float(np.max(np.abs(error))),
    }


def validation_loss_by_cell(model, cells, cell_ids, scalers, batch_size, device) -> tuple[float, dict]:
    values = {}
    for cell_id in cell_ids:
        loader = make_loader(scaled_stack(cells, [cell_id], scalers), batch_size, False)
        target, prediction = predict(model, loader, device)
        values[cell_id] = float(np.mean(np.square(prediction - target), dtype=np.float64))
    return float(np.mean(list(values.values()))), values


@dataclass
class TrainResult:
    state_dict: dict
    selected_epoch: int
    history: list[dict]


def train(
    model,
    train_loader,
    validation_cells,
    validation_ids,
    scalers,
    device,
    epochs,
    lr,
    patience,
) -> TrainResult:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history, stale = [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for voltage, current, duration, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = unwrap_prediction(
                model(voltage.to(device), current.to(device), duration.to(device))
            )
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_loss, per_cell = validation_loss_by_cell(
            model, validation_cells, validation_ids, scalers, train_loader.batch_size, device
        )
        history.append(
            {"epoch": epoch, "train_mse": float(np.mean(losses)), "macro_validation_mse": val_loss, **{f"val_mse__{k}": v for k, v in per_cell.items()}}
        )
        if val_loss < best_loss - 1e-12:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if patience > 0 and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    return TrainResult(best_state, best_epoch, history)


def evaluate_cells(model, cells, cell_ids, scalers, batch_size, device, context: dict):
    metric_rows, prediction_rows = [], []
    all_target, all_prediction = [], []
    for cell_id in cell_ids:
        loader = make_loader(scaled_stack(cells, [cell_id], scalers), batch_size, False)
        target, prediction = predict(model, loader, device)
        metric_rows.append({**context, "cell_id": cell_id, **regression_metrics(target, prediction)})
        all_target.append(target)
        all_prediction.append(prediction)
        prediction_rows.extend(
            {**context, "cell_id": cell_id, "sample_index": index, "target_soh": float(y), "prediction_soh": float(p), "error": float(p - y)}
            for index, (y, p) in enumerate(zip(target, prediction))
        )
    target = np.concatenate(all_target)
    prediction = np.concatenate(all_prediction)
    aggregate = regression_metrics(target, prediction)
    aggregate["MacroCellRMSE"] = float(np.mean([row["RMSE"] for row in metric_rows]))
    aggregate["MacroCellMAE"] = float(np.mean([row["MAE"] for row in metric_rows]))
    aggregate["MacroAbsCellBias"] = float(np.mean([abs(row["Bias"]) for row in metric_rows]))
    return {**context, "test_cells": ";".join(cell_ids), **aggregate}, metric_rows, prediction_rows


def parse_sizes(values: list[str], pool_size: int) -> list[int]:
    sizes = []
    for value in values:
        size = pool_size if value.lower() == "all" else int(value)
        if size < 1 or size > pool_size:
            raise ValueError(f"Invalid target subset size {value}; pool has {pool_size} cells")
        if size not in sizes:
            sizes.append(size)
    return sizes


def save_checkpoint(path: Path, model_name: str, result: TrainResult, scalers, metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": result.state_dict,
            "selected_epoch": result.selected_epoch,
            "scalers": scalers.to_dict(),
            "metadata": metadata,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", choices=ALL_MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--target-sizes", nargs="+", default=["1", "3", "6", "all"])
    parser.add_argument("--source-validation-cell", default=DEFAULT_SOURCE_VALIDATION)
    parser.add_argument("--target-validation-cell", default=DEFAULT_TARGET_VALIDATION)
    parser.add_argument("--target-test-cells", nargs="+", default=[DEFAULT_TARGET_TEST])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--source-epochs", type=int, default=80)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--source-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--experiment-label",
        default="baseline_four_models",
        help="Human-readable variant name recorded in the run manifest.",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    source_cells = load_tju_cells(args.data_root, "0.25C")
    target_cells = load_tju_cells(args.data_root, "0.5C")
    source_train = sorted(set(source_cells) - {args.source_validation_cell})
    target_excluded = {args.target_validation_cell, *args.target_test_cells}
    target_pool = sorted(set(target_cells) - target_excluded)
    validate_roles(source_cells, source_train=source_train, source_validation=[args.source_validation_cell])
    validate_roles(
        target_cells,
        target_pool=target_pool,
        target_validation=[args.target_validation_cell],
        target_test=args.target_test_cells,
    )
    target_sizes = parse_sizes(args.target_sizes, len(target_pool))
    model_rows = []
    for name in args.models:
        model = create_model(name)
        model_rows.append({"model": name, "parameters": sum(parameter.numel() for parameter in model.parameters())})
    print(f"TJU source train={source_train}; source validation={args.source_validation_cell}")
    print(f"TJU target pool={len(target_pool)} cells; validation={args.target_validation_cell}; test={args.target_test_cells}")
    print(f"Nested target sizes={target_sizes}; models={model_rows}")
    if args.validate_only:
        print("Validation-only mode complete; no training output was created.")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    if any("Mamba" in name for name in args.models) and device.type != "cuda":
        raise RuntimeError("Mamba model training requires CUDA; use --validate-only locally")
    output = (args.output_dir or Path(__file__).resolve().parent / "runs" / f"seed_{args.seed}").resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_raw = stack_cells(source_cells, source_train)
    scalers = fit_scalers(*source_raw[:3])
    source_train_loader = make_loader(
        scaled_stack(source_cells, source_train, scalers), args.batch_size, True, cell_balanced=True
    )
    manifest = {
        "experiment": "Experiment 2: TJU 0.25C to 0.5C full-parameter fine-tuning",
        "experiment_label": args.experiment_label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root.resolve()),
        "data_files": {cell.cell_id: {"path": str(cell.path), "sha256": file_sha256(cell.path), "windows": cell.size} for cell in [*source_cells.values(), *target_cells.values()]},
        "source_train_cells": source_train,
        "source_validation_cell": args.source_validation_cell,
        "target_candidate_pool": target_pool,
        "target_validation_cell": args.target_validation_cell,
        "target_test_cells": args.target_test_cells,
        "target_sizes": target_sizes,
        "subset_seeds": args.subset_seeds,
        "training_seed": args.seed,
        "models": model_rows,
        "scalers": scalers.to_dict(),
        "invariants": {
            "all_parameters_finetuned": True,
            "source_scalers_reused_for_all_target_levels": True,
            "test_cells_excluded_from_scaling_selection_and_training": True,
            "cell_balanced_training_sampler": True,
            "target_subsets_nested_within_each_subset_seed": True,
            "physical_cell_roles_are_disjoint": True,
        },
        "metrics": {"primary": ["RMSE", "MAE", "Bias", "MacroAbsCellBias"], "auxiliary": ["R2", "MaxError"]},
        "arguments": vars(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir) if args.output_dir else None},
    }
    write_json(output / "run_manifest.json", manifest)

    aggregate_rows, per_cell_rows, prediction_rows = [], [], []
    for model_name in args.models:
        set_seed(args.seed)
        source_model = create_model(model_name)
        source_result = train(
            source_model,
            source_train_loader,
            source_cells,
            [args.source_validation_cell],
            scalers,
            device,
            args.source_epochs,
            args.source_lr,
            args.patience,
        )
        source_model.load_state_dict(source_result.state_dict)
        source_model.to(device)
        model_dir = output / model_name
        write_csv(model_dir / "source_history.csv", source_result.history)
        save_checkpoint(model_dir / "source_025C_best.pt", model_name, source_result, scalers, {"source_train_cells": source_train, "source_validation_cell": args.source_validation_cell})

        context = {"model": model_name, "training_seed": args.seed, "stage": "direct", "subset_seed": "", "target_cell_count": 0, "target_cells": ""}
        aggregate, per_cell, predictions = evaluate_cells(source_model, target_cells, args.target_test_cells, scalers, args.batch_size, device, context)
        aggregate_rows.append(aggregate)
        per_cell_rows.extend(per_cell)
        prediction_rows.extend(predictions)

        for subset_seed in args.subset_seeds:
            permutation = np.random.default_rng(subset_seed).permutation(target_pool).tolist()
            for size in target_sizes:
                if size == len(target_pool) and subset_seed != args.subset_seeds[0]:
                    continue
                selected_cells = permutation[:size]
                tune_loader = make_loader(
                    scaled_stack(target_cells, selected_cells, scalers),
                    args.batch_size,
                    True,
                    cell_balanced=True,
                )
                set_seed(args.seed + subset_seed + size)
                tuned_model = copy.deepcopy(source_model)
                for parameter in tuned_model.parameters():
                    parameter.requires_grad = True
                tune_result = train(
                    tuned_model,
                    tune_loader,
                    target_cells,
                    [args.target_validation_cell],
                    scalers,
                    device,
                    args.finetune_epochs,
                    args.finetune_lr,
                    args.patience,
                )
                tuned_model.load_state_dict(tune_result.state_dict)
                tuned_model.to(device)
                level = f"subset_seed_{subset_seed}/cells_{size}"
                write_csv(model_dir / level / "finetune_history.csv", tune_result.history)
                save_checkpoint(
                    model_dir / level / "finetuned_best.pt",
                    model_name,
                    tune_result,
                    scalers,
                    {"target_cells": selected_cells, "target_validation_cell": args.target_validation_cell, "target_test_cells": args.target_test_cells, "full_parameter_finetune": True},
                )
                context = {
                    "model": model_name,
                    "training_seed": args.seed,
                    "stage": "finetuned",
                    "subset_seed": subset_seed,
                    "target_cell_count": size,
                    "target_cells": ";".join(selected_cells),
                }
                aggregate, per_cell, predictions = evaluate_cells(tuned_model, target_cells, args.target_test_cells, scalers, args.batch_size, device, context)
                aggregate_rows.append(aggregate)
                per_cell_rows.extend(per_cell)
                prediction_rows.extend(predictions)
                write_json(model_dir / level / "selected_cells.json", {"nested_permutation": permutation, "selected_cells": selected_cells})
        write_csv(output / "metrics_summary.csv", aggregate_rows)
        write_csv(output / "metrics_by_test_cell.csv", per_cell_rows)
        write_csv(output / "predictions.csv", prediction_rows)
    print(f"Experiment 2 complete: {output}")


if __name__ == "__main__":
    main()
