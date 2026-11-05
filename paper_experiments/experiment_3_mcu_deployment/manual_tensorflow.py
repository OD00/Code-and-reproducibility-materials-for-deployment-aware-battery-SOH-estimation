"""TensorFlow implementations matching the fixed PyTorch deployment graphs.

This module intentionally does not use ONNX. PyTorch weights are copied into
TensorFlow variables and each operation is expressed directly with TensorFlow.
The deployment signature is fixed to batch=1, 12 cycles, and 256 points.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


class ManualTensorFlowModel(tf.Module):
    def __init__(self, model_name: str, state: dict):
        super().__init__(name=f"manual_{model_name.lower()}")
        self.model_name = model_name
        self._key_map = {key: key.replace(".", "__") for key in state}
        self.weights = {
            self._key_map[key]: tf.Variable(
                value.detach().cpu().numpy().astype(np.float32),
                trainable=False,
                name=self._key_map[key],
            )
            for key, value in state.items()
            if "num_batches_tracked" not in key
        }

    def w(self, key: str) -> tf.Tensor:
        return self.weights[self._key_map[key]]

    @staticmethod
    def dense(x: tf.Tensor, weight: tf.Tensor, bias: tf.Tensor | None = None) -> tf.Tensor:
        output = tf.linalg.matmul(x, weight, transpose_b=True)
        return output if bias is None else tf.nn.bias_add(output, bias)

    @staticmethod
    def conv1d(
        x: tf.Tensor,
        weight: tf.Tensor,
        bias: tf.Tensor | None,
        stride: int = 1,
        padding: int = 0,
    ) -> tf.Tensor:
        if padding:
            x = tf.pad(x, [[0, 0], [padding, padding], [0, 0]])
        kernel = tf.transpose(weight, [2, 1, 0])
        output = tf.nn.conv1d(x, kernel, stride=stride, padding="VALID")
        return output if bias is None else tf.nn.bias_add(output, bias)

    def batch_norm(self, x: tf.Tensor, prefix: str) -> tf.Tensor:
        return tf.nn.batch_normalization(
            x,
            self.w(f"{prefix}.running_mean"),
            self.w(f"{prefix}.running_var"),
            self.w(f"{prefix}.bias"),
            self.w(f"{prefix}.weight"),
            1e-5,
        )

    @staticmethod
    def layer_norm(x: tf.Tensor, weight: tf.Tensor, bias: tf.Tensor) -> tf.Tensor:
        mean, variance = tf.nn.moments(x, axes=[-1], keepdims=True)
        return (x - mean) * tf.math.rsqrt(variance + 1e-5) * weight + bias

    def multiscale(self, x: tf.Tensor, prefix: str) -> tf.Tensor:
        branches = []
        for name, padding in (("conv1", 1), ("conv2", 2), ("conv3", 3)):
            branches.append(
                self.conv1d(
                    x,
                    self.w(f"{prefix}.{name}.weight"),
                    self.w(f"{prefix}.{name}.bias"),
                    padding=padding,
                )
            )
        output = tf.concat(branches, axis=-1)
        output = tf.nn.relu(self.batch_norm(output, f"{prefix}.bn"))
        pooled = tf.reduce_mean(output, axis=1)
        excitation = self.dense(pooled, self.w(f"{prefix}.se.fc.0.weight"))
        excitation = tf.nn.relu(excitation)
        excitation = self.dense(excitation, self.w(f"{prefix}.se.fc.2.weight"))
        excitation = tf.nn.sigmoid(excitation)
        return output * excitation[:, None, :]

    def multiscale_feature_extractor(self, x: tf.Tensor) -> tf.Tensor:
        x = self.multiscale(x, "feature_extractor.0")
        x = tf.nn.max_pool1d(x, ksize=2, strides=2, padding="VALID")
        x = self.multiscale(x, "feature_extractor.2")
        x = tf.nn.max_pool1d(x, ksize=2, strides=2, padding="VALID")
        x = self.multiscale(x, "feature_extractor.4")
        return tf.reduce_mean(x, axis=1)

    def simple_feature_extractor(self, x: tf.Tensor) -> tf.Tensor:
        """Match Conv1d(2, 64, kernel=3, padding=1) + ReLU + global mean."""
        x = self.conv1d(
            x,
            self.w("feature_extractor.0.weight"),
            self.w("feature_extractor.0.bias"),
            padding=1,
        )
        return tf.reduce_mean(tf.nn.relu(x), axis=1)

    def feature_extractor(self, x: tf.Tensor) -> tf.Tensor:
        if self.model_name in ("CnnMambaSimpleCNN", "CnnLSTMSimpleCNN"):
            return self.simple_feature_extractor(x)
        return self.multiscale_feature_extractor(x)

    @staticmethod
    def cycle_input(voltage: tf.Tensor, current: tf.Tensor) -> tf.Tensor:
        x = tf.stack([voltage, current], axis=2)
        x = tf.reshape(x, [12, 2, 256])
        return tf.transpose(x, [0, 2, 1])

    def duration_fusion(self, x: tf.Tensor, duration: tf.Tensor, prefix: str) -> tf.Tensor:
        encoded = self.dense(
            duration,
            self.w(f"{prefix}.weight"),
            self.w(f"{prefix}.bias"),
        )
        return x + self.w("duration_gate") * encoded

    def multihead_attention(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        prefix: str,
    ) -> tf.Tensor:
        weight = self.w(f"{prefix}.in_proj_weight")
        bias = self.w(f"{prefix}.in_proj_bias")
        q = self.dense(query, weight[:64], bias[:64])
        k = self.dense(key, weight[64:128], bias[64:128])
        v = self.dense(value, weight[128:], bias[128:])

        def split_heads(tensor: tf.Tensor) -> tf.Tensor:
            shape = tf.shape(tensor)
            tensor = tf.reshape(tensor, [shape[0], shape[1], 4, 16])
            return tf.transpose(tensor, [0, 2, 1, 3])

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        scores = tf.linalg.matmul(q, k, transpose_b=True) * 0.25
        attention = tf.nn.softmax(scores, axis=-1)
        context = tf.linalg.matmul(attention, v)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [tf.shape(context)[0], tf.shape(context)[1], 64])
        return self.dense(
            context,
            self.w(f"{prefix}.out_proj.weight"),
            self.w(f"{prefix}.out_proj.bias"),
        )

    def transformer(self, voltage: tf.Tensor, current: tf.Tensor, duration: tf.Tensor) -> tf.Tensor:
        x = self.cycle_input(voltage, current)
        x = self.conv1d(
            x,
            self.w("feature_extractor.0.weight"),
            self.w("feature_extractor.0.bias"),
            padding=1,
        )
        x = tf.reduce_mean(tf.nn.relu(x), axis=1)
        x = self.dense(x, self.w("proj.weight"), self.w("proj.bias"))
        x = tf.reshape(x, [1, 12, 64])
        x = self.duration_fusion(x, duration, "duration_encoder")
        x = x + self.w("pos_embedding")[:, :12, :]
        for index in range(2):
            prefix = f"transformer.layers.{index}"
            attention = self.multihead_attention(x, x, x, f"{prefix}.self_attn")
            x = self.layer_norm(
                x + attention,
                self.w(f"{prefix}.norm1.weight"),
                self.w(f"{prefix}.norm1.bias"),
            )
            feedforward = self.dense(
                x,
                self.w(f"{prefix}.linear1.weight"),
                self.w(f"{prefix}.linear1.bias"),
            )
            feedforward = tf.nn.relu(feedforward)
            feedforward = self.dense(
                feedforward,
                self.w(f"{prefix}.linear2.weight"),
                self.w(f"{prefix}.linear2.bias"),
            )
            x = self.layer_norm(
                x + feedforward,
                self.w(f"{prefix}.norm2.weight"),
                self.w(f"{prefix}.norm2.bias"),
            )
        return self.dense(x[:, -1, :], self.w("fc_out.weight"), self.w("fc_out.bias"))

    def manual_lstm_layer(self, sequence: tf.Tensor, layer: int) -> tf.Tensor:
        hidden = tf.zeros([1, 128], tf.float32)
        cell = tf.zeros([1, 128], tf.float32)
        outputs = []
        for value in tf.unstack(sequence, num=12, axis=1):
            gates = self.dense(value, self.w(f"lstm.weight_ih_l{layer}"), self.w(f"lstm.bias_ih_l{layer}"))
            gates += self.dense(hidden, self.w(f"lstm.weight_hh_l{layer}"), self.w(f"lstm.bias_hh_l{layer}"))
            input_gate, forget_gate, candidate, output_gate = tf.split(gates, 4, axis=-1)
            input_gate = tf.nn.sigmoid(input_gate)
            forget_gate = tf.nn.sigmoid(forget_gate)
            candidate = tf.nn.tanh(candidate)
            output_gate = tf.nn.sigmoid(output_gate)
            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * tf.nn.tanh(cell)
            outputs.append(hidden)
        return tf.stack(outputs, axis=1)

    def lstm(self, voltage: tf.Tensor, current: tf.Tensor, duration: tf.Tensor) -> tf.Tensor:
        x = self.feature_extractor(self.cycle_input(voltage, current))
        x = self.dense(x, self.w("proj_to_seq.weight"), self.w("proj_to_seq.bias"))
        x = tf.reshape(x, [1, 12, 64])
        x = self.duration_fusion(x, duration, "duration_encoder")
        x = self.manual_lstm_layer(x, 0)
        x = self.manual_lstm_layer(x, 1)
        return self.dense(x[:, -1, :], self.w("fc_out.0.weight"), self.w("fc_out.0.bias"))

    def res_block(self, x: tf.Tensor, prefix: str) -> tf.Tensor:
        main = self.conv1d(x, self.w(f"{prefix}.conv1.weight"), self.w(f"{prefix}.conv1.bias"), padding=1)
        main = tf.nn.relu(self.batch_norm(main, f"{prefix}.bn1"))
        main = self.conv1d(main, self.w(f"{prefix}.conv2.weight"), self.w(f"{prefix}.conv2.bias"), padding=1)
        main = self.batch_norm(main, f"{prefix}.bn2")
        shortcut = self.conv1d(x, self.w(f"{prefix}.shortcut.0.weight"), self.w(f"{prefix}.shortcut.0.bias"))
        shortcut = self.batch_norm(shortcut, f"{prefix}.shortcut.1")
        return tf.nn.relu(main + shortcut)

    def resnet(self, voltage: tf.Tensor, current: tf.Tensor, duration: tf.Tensor) -> tf.Tensor:
        x = self.cycle_input(voltage, current)
        x = self.conv1d(x, self.w("cycle_feature.0.weight"), self.w("cycle_feature.0.bias"), stride=2, padding=3)
        x = tf.reduce_mean(tf.nn.relu(x), axis=1)
        x = tf.reshape(x, [1, 12, 32])
        x = self.duration_fusion(x, duration, "duration_encoder")
        x = self.res_block(x, "layer1")
        x = self.res_block(x, "layer2")
        x = tf.reduce_mean(x, axis=1)
        return self.dense(x, self.w("fc.weight"), self.w("fc.bias"))

    def mamba_block(self, hidden: tf.Tensor) -> tf.Tensor:
        projected = self.dense(hidden, self.w("mamba.in_proj.weight"))
        x, z = tf.split(projected, 2, axis=-1)
        padded = tf.pad(x, [[0, 0], [3, 0], [0, 0]])
        image = padded[:, None, :, :]
        kernel = tf.transpose(self.w("mamba.conv1d.weight"), [2, 0, 1])
        kernel = kernel[None, :, :, :]
        x = tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding="VALID")
        x = tf.squeeze(x, axis=1)
        x = tf.nn.bias_add(x, self.w("mamba.conv1d.bias"))
        x = tf.nn.silu(x)
        projected_state = self.dense(x, self.w("mamba.x_proj.weight"))
        delta_low_rank, b_term, c_term = tf.split(projected_state, [4, 16, 16], axis=-1)
        delta = self.dense(delta_low_rank, self.w("mamba.dt_proj.weight"), self.w("mamba.dt_proj.bias"))
        delta = tf.nn.softplus(delta)
        a_matrix = -tf.exp(self.w("mamba.A_log"))
        state = tf.zeros([1, 128, 16], tf.float32)
        outputs = []
        for index in range(12):
            delta_t = delta[:, index, :]
            x_t = x[:, index, :]
            a_discrete = tf.exp(delta_t[:, :, None] * a_matrix[None, :, :])
            b_discrete = delta_t[:, :, None] * b_term[:, index, None, :]
            state = state * a_discrete + x_t[:, :, None] * b_discrete
            y_t = tf.reduce_sum(state * c_term[:, index, None, :], axis=-1)
            y_t += self.w("mamba.D")[None, :] * x_t
            y_t *= tf.nn.silu(z[:, index, :])
            outputs.append(y_t)
        sequence = tf.stack(outputs, axis=1)
        return self.dense(sequence, self.w("mamba.out_proj.weight"))

    def mamba(self, voltage: tf.Tensor, current: tf.Tensor, duration: tf.Tensor) -> tf.Tensor:
        x = self.feature_extractor(self.cycle_input(voltage, current))
        x = self.dense(x, self.w("proj_to_mamba.0.weight"), self.w("proj_to_mamba.0.bias"))
        x = self.layer_norm(x, self.w("proj_to_mamba.1.weight"), self.w("proj_to_mamba.1.bias"))
        x = tf.nn.silu(x)
        x = tf.reshape(x, [1, 12, 64])
        x = self.duration_fusion(x, duration, "duration_encoder")
        x = self.mamba_block(x)
        x = self.layer_norm(x, self.w("norm_out.weight"), self.w("norm_out.bias"))
        query = self.w("decoder.query")
        context = self.multihead_attention(query, x, x, "decoder.attn")[:, 0, :]
        gate = tf.nn.silu(self.dense(context, self.w("decoder.swiglu_linear1.weight"), self.w("decoder.swiglu_linear1.bias")))
        up = self.dense(context, self.w("decoder.swiglu_linear2.weight"), self.w("decoder.swiglu_linear2.bias"))
        return self.dense(gate * up, self.w("decoder.output_proj.weight"), self.w("decoder.output_proj.bias"))

    @tf.function(
        input_signature=[
            tf.TensorSpec([1, 12, 256], tf.float32, name="voltage"),
            tf.TensorSpec([1, 12, 256], tf.float32, name="current"),
            tf.TensorSpec([1, 12, 1], tf.float32, name="duration"),
        ]
    )
    def __call__(self, voltage: tf.Tensor, current: tf.Tensor, duration: tf.Tensor) -> dict[str, tf.Tensor]:
        if self.model_name == "CnnTransformer":
            output = self.transformer(voltage, current, duration)
        elif self.model_name in ("CnnLSTMControl", "CnnLSTMSimpleCNN"):
            output = self.lstm(voltage, current, duration)
        elif self.model_name == "ResNet1d":
            output = self.resnet(voltage, current, duration)
        elif self.model_name in ("CnnMambaImproved", "CnnMambaSimpleCNN"):
            output = self.mamba(voltage, current, duration)
        else:
            raise ValueError(self.model_name)
        return {"soh": output}
