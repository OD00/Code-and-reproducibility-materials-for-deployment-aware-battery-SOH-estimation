"""Exact four-model registry used by A2 and the new experiments."""

from __future__ import annotations

from src.models import CnnLSTMControl, CnnMambaImproved, CnnTransformer, ResNet1d

MODEL_NAMES = (
    "CnnTransformer",
    "CnnLSTMControl",
    "ResNet1d",
    "CnnMambaImproved",
)

# Experimental variants used for the follow-up frontend study.  They are kept
# outside MODEL_NAMES so the historical four-model paper experiment retains
# its original default configuration.
EXPERIMENTAL_MODEL_NAMES = (
    "CnnMambaSimpleCNN",
    "CnnLSTMSimpleCNN",
)

ALL_MODEL_NAMES = MODEL_NAMES + EXPERIMENTAL_MODEL_NAMES


def create_model(name: str):
    if name == "CnnTransformer":
        return CnnTransformer(d_model=64, nhead=4, num_layers=2, output_dim=1, use_duration_fusion=True)
    if name == "CnnLSTMControl":
        return CnnLSTMControl(d_model=64, hidden_size=128, output_dim=1, use_duration_fusion=True)
    if name == "ResNet1d":
        return ResNet1d(output_dim=1, use_duration_fusion=True)
    if name == "CnnMambaImproved":
        return CnnMambaImproved(
            d_model=64, d_state=16, d_conv=4, expand=2, output_dim=1, use_duration_fusion=True
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
    raise ValueError(f"Unknown model: {name}")


def unwrap_prediction(value):
    return value[0] if isinstance(value, tuple) else value
