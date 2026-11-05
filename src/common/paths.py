"""Project paths and collision-free experiment output directories."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def create_run_dir(
    experiment: str,
    label: str,
    seed: int,
    output_dir: str | Path | None = None,
) -> Path:
    """Create a new run directory without overwriting an earlier run."""
    if output_dir is not None:
        path = Path(output_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUTPUT_DIR / experiment / "runs"
    candidate = base / f"{timestamp}_{safe_label}_seed{seed}"
    suffix = 1
    while candidate.exists():
        candidate = base / f"{timestamp}_{safe_label}_seed{seed}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def legacy_result_dir(experiment: str, name: str) -> Path:
    """Resolve an existing migrated result without copying it."""
    return OUTPUT_DIR / experiment / "legacy_runs" / name

