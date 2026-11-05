"""TensorFlow mapping that preserves the two PyTorch LSTM layers as fused layers."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import tf_keras

LegacyLSTM = tf_keras.layers.LSTM

from paper_experiments.experiment_3_mcu_deployment.manual_tensorflow import (
    ManualTensorFlowModel,
)


class FusedLSTMModel(ManualTensorFlowModel):
    """Exact LSTM mapping with trackable Keras sequence LSTMs.

    Both CnnLSTMControl and CnnLSTMSimpleCNN share the same two-layer LSTM
    state layout.  The model name only selects the matching cycle frontend in
    ManualTensorFlowModel; the recurrent weights are mapped identically.
    """

    def __init__(self, state: dict, model_name: str = "CnnLSTMControl"):
        if model_name not in ("CnnLSTMControl", "CnnLSTMSimpleCNN"):
            raise ValueError(f"Unsupported fused LSTM model: {model_name}")
        super().__init__(model_name, state)
        # TensorFlow 2.21 exposes Keras 3 through tf.keras. Keras 3 traces an
        # explicit WHILE/READ_VARIABLE graph here, which the TFLite RNN fusion
        # pass does not recognize. The separately installed tf_keras v2 runtime
        # carries the composite-function metadata expected by the official
        # TFLite RNN conversion pass.
        self.fused_lstm_0 = LegacyLSTM(
            128,
            return_sequences=True,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            unit_forget_bias=False,
            unroll=False,
            name="fused_lstm_0",
        )
        self.fused_lstm_1 = LegacyLSTM(
            128,
            return_sequences=True,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            unit_forget_bias=False,
            unroll=False,
            name="fused_lstm_1",
        )
        first = self.fused_lstm_0(tf.zeros([1, 12, 64], tf.float32))
        self.fused_lstm_1(first)
        self._load_lstm_weights(state, 0, self.fused_lstm_0)
        self._load_lstm_weights(state, 1, self.fused_lstm_1)

    @staticmethod
    def _array(state: dict, key: str) -> np.ndarray:
        return state[key].detach().cpu().numpy().astype(np.float32)

    @classmethod
    def _load_lstm_weights(
        cls,
        state: dict,
        layer_index: int,
        layer: LegacyLSTM,
    ) -> None:
        # PyTorch and Keras both use the i, f, candidate, o gate order. Keras
        # stores input/recurrent matrices transposed and has one combined bias.
        kernel = cls._array(state, f"lstm.weight_ih_l{layer_index}").T
        recurrent = cls._array(state, f"lstm.weight_hh_l{layer_index}").T
        bias = (
            cls._array(state, f"lstm.bias_ih_l{layer_index}")
            + cls._array(state, f"lstm.bias_hh_l{layer_index}")
        )
        layer.set_weights([kernel, recurrent, bias])

    def lstm(
        self,
        voltage: tf.Tensor,
        current: tf.Tensor,
        duration: tf.Tensor,
    ) -> tf.Tensor:
        x = self.feature_extractor(self.cycle_input(voltage, current))
        x = self.dense(x, self.w("proj_to_seq.weight"), self.w("proj_to_seq.bias"))
        x = tf.reshape(x, [1, 12, 64])
        x = self.duration_fusion(x, duration, "duration_encoder")
        x = self.fused_lstm_0(x, training=False)
        x = self.fused_lstm_1(x, training=False)
        return self.dense(
            x[:, -1, :],
            self.w("fc_out.0.weight"),
            self.w("fc_out.0.bias"),
        )
