"""Shared data, split, training and evaluation utilities.

The implementation is intentionally kept in ``experiment.py`` during the
first migration so numerical behavior remains traceable to the old
``exp_common.py``. Public experiment code should import from ``src.common``.
"""

from .experiment import *  # noqa: F401,F403
from .paths import (  # noqa: F401
    DATA_DIR,
    OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    SPLIT_DIR,
    create_run_dir,
    legacy_result_dir,
)

