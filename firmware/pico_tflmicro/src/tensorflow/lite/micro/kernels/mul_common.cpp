/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cmath>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mul.h"
#include "tensorflow/lite/kernels/internal/reference/mul.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/mul.h"
#include "tensorflow/lite/micro/memory_helpers.h"

namespace tflite {

const int kMulInput1Tensor = 0;
const int kMulInput2Tensor = 1;
const int kMulOutputTensor = 0;

namespace {

constexpr int kNoSiluLutInput = -1;
constexpr int kSiluLutInput1 = 0;
constexpr int kSiluLutInput2 = 1;
constexpr int kLutSize = 256;
constexpr float kSigmoidInt8Scale = 1.0f / 256.0f;
constexpr float kFloatLutMin = -8.0f;
constexpr float kFloatLutMax = 8.0f;

bool NearlyEqual(float a, float b) {
  return std::fabs(a - b) <= 1e-6f;
}

bool SameShape(const TfLiteTensor* input1, const TfLiteTensor* input2,
               const TfLiteTensor* output) {
  if (input1->dims->size != input2->dims->size ||
      input1->dims->size != output->dims->size) {
    return false;
  }
  for (int i = 0; i < input1->dims->size; ++i) {
    if (input1->dims->data[i] != input2->dims->data[i] ||
        input1->dims->data[i] != output->dims->data[i]) {
      return false;
    }
  }
  return true;
}

bool IsSigmoidInt8Tensor(const TfLiteTensor* tensor) {
  return tensor->type == kTfLiteInt8 &&
         NearlyEqual(tensor->params.scale, kSigmoidInt8Scale) &&
         tensor->params.zero_point == -128;
}

int32_t ClampInt32(int32_t value, int32_t min_value, int32_t max_value) {
  if (value < min_value) return min_value;
  if (value > max_value) return max_value;
  return value;
}

float Silu(float x) {
  return x / (1.0f + std::exp(-x));
}

void FillInt8SiluLut(const TfLiteTensor* input, const TfLiteTensor* output,
                     const OpDataMul* data, int8_t* lut) {
  for (int i = 0; i < kLutSize; ++i) {
    const int32_t q = i - 128;
    const float x =
        (static_cast<float>(q) - input->params.zero_point) * input->params.scale;
    const float y = Silu(x);
    const int32_t q_out = static_cast<int32_t>(
        std::nearbyint(y / output->params.scale + output->params.zero_point));
    lut[i] = static_cast<int8_t>(
        ClampInt32(q_out, data->output_activation_min,
                   data->output_activation_max));
  }
}

void FillFloatSiluLut(float* lut) {
  const float step = (kFloatLutMax - kFloatLutMin) / (kLutSize - 1);
  for (int i = 0; i < kLutSize; ++i) {
    const float x = kFloatLutMin + step * i;
    lut[i] = Silu(x);
  }
}

bool MaybePrepareSiluLut(TfLiteContext* context, const TfLiteTensor* input1,
                         const TfLiteTensor* input2,
                         const TfLiteTensor* output, OpDataMul* data) {
  data->silu_lut_input_index = kNoSiluLutInput;
  data->silu_lut_int8 = nullptr;
  data->silu_lut_float = nullptr;

  if (!SameShape(input1, input2, output)) {
    return false;
  }

  if (output->type == kTfLiteInt8) {
    const bool input1_is_sigmoid = IsSigmoidInt8Tensor(input1);
    const bool input2_is_sigmoid = IsSigmoidInt8Tensor(input2);
    if (input1_is_sigmoid == input2_is_sigmoid) {
      return false;
    }

    const TfLiteTensor* silu_input = input1_is_sigmoid ? input2 : input1;
    int8_t* lut = static_cast<int8_t*>(
        context->AllocatePersistentBuffer(context, kLutSize));
    if (lut == nullptr) {
      return false;
    }

    FillInt8SiluLut(silu_input, output, data, lut);
    data->silu_lut_input_index =
        input1_is_sigmoid ? kSiluLutInput2 : kSiluLutInput1;
    data->silu_lut_int8 = lut;
    return true;
  }

  if (input1->type == kTfLiteFloat32 && input2->type == kTfLiteFloat32 &&
      output->type == kTfLiteFloat32) {
    float* lut = static_cast<float*>(
        context->AllocatePersistentBuffer(context, kLutSize * sizeof(float)));
    if (lut == nullptr) {
      return false;
    }

    FillFloatSiluLut(lut);
    data->silu_lut_input_index = kSiluLutInput1;
    data->silu_lut_float = lut;
    return true;
  }

  return false;
}

float EvalFloatSiluLut(float x, const float* lut) {
  if (x <= kFloatLutMin) {
    return 0.0f;
  }
  if (x >= kFloatLutMax) {
    return x;
  }
  const float step = (kFloatLutMax - kFloatLutMin) / (kLutSize - 1);
  const float idx_f = (x - kFloatLutMin) / step;
  const int idx = static_cast<int>(idx_f);
  const float frac = idx_f - idx;
  return lut[idx] + frac * (lut[idx + 1] - lut[idx]);
}

}  // namespace

void* MulInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataMul));
}

TfLiteStatus CalculateOpDataMul(TfLiteContext* context, TfLiteNode* node,
                                TfLiteMulParams* params, OpDataMul* data) {
  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input1 =
      micro_context->AllocateTempInputTensor(node, kMulInput1Tensor);
  TF_LITE_ENSURE(context, input1 != nullptr);
  TfLiteTensor* input2 =
      micro_context->AllocateTempInputTensor(node, kMulInput2Tensor);
  TF_LITE_ENSURE(context, input2 != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kMulOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);

  if (output->type == kTfLiteInt8 || output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));

    double real_multiplier = static_cast<double>(input1->params.scale) *
                             static_cast<double>(input2->params.scale) /
                             static_cast<double>(output->params.scale);
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);

    data->input1_zero_point = input1->params.zero_point;
    data->input2_zero_point = input2->params.zero_point;
    data->output_zero_point = output->params.zero_point;

    if (input1->type == kTfLiteInt16) {
      TF_LITE_ENSURE_EQ(context, data->input1_zero_point, 0);
      TF_LITE_ENSURE_EQ(context, data->input2_zero_point, 0);
      TF_LITE_ENSURE_EQ(context, data->output_zero_point, 0);
    }
  } else if (output->type == kTfLiteInt32) {
    CalculateActivationRange(params->activation, &data->output_activation_min,
                             &data->output_activation_max);
  } else {
    CalculateActivationRange(params->activation,
                             &data->output_activation_min_f32,
                             &data->output_activation_max_f32);
  }

  MaybePrepareSiluLut(context, input1, input2, output, data);

  micro_context->DeallocateTempTfLiteTensor(input1);
  micro_context->DeallocateTempTfLiteTensor(input2);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus MulPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLiteMulParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  OpDataMul* data = static_cast<OpDataMul*>(node->user_data);

  return CalculateOpDataMul(context, node, params, data);
}

TfLiteStatus EvalMulQuantizedReference(TfLiteContext* context, TfLiteNode* node,
                                       const OpDataMul* data,
                                       const TfLiteEvalTensor* input1,
                                       const TfLiteEvalTensor* input2,
                                       TfLiteEvalTensor* output) {
  tflite::ArithmeticParams op_params = {};
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.float_activation_max = data->output_activation_max_f32;
  op_params.input1_offset = -data->input1_zero_point;
  op_params.input2_offset = -data->input2_zero_point;
  op_params.output_offset = data->output_zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;

  bool need_broadcast = reference_ops::ProcessBroadcastShapes(
      tflite::micro::GetTensorShape(input1),
      tflite::micro::GetTensorShape(input2), &op_params);

  if (input1->type == kTfLiteInt8) {
    if (need_broadcast) {
      reference_integer_ops::BroadcastMul4DSlow(
          op_params, tflite::micro::GetTensorShape(input1),
          tflite::micro::GetTensorData<int8_t>(input1),
          tflite::micro::GetTensorShape(input2),
          tflite::micro::GetTensorData<int8_t>(input2),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
    } else {
      reference_integer_ops::Mul(op_params,
                                 tflite::micro::GetTensorShape(input1),
                                 tflite::micro::GetTensorData<int8_t>(input1),
                                 tflite::micro::GetTensorShape(input2),
                                 tflite::micro::GetTensorData<int8_t>(input2),
                                 tflite::micro::GetTensorShape(output),
                                 tflite::micro::GetTensorData<int8_t>(output));
    }
  } else if (input1->type == kTfLiteInt32) {
    if (need_broadcast) {
      reference_ops::BroadcastMul4DSlow(
          op_params, tflite::micro::GetTensorShape(input1),
          tflite::micro::GetTensorData<int32_t>(input1),
          tflite::micro::GetTensorShape(input2),
          tflite::micro::GetTensorData<int32_t>(input2),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int32_t>(output));
    } else {
      reference_ops::Mul(op_params, tflite::micro::GetTensorShape(input1),
                         tflite::micro::GetTensorData<int32_t>(input1),
                         tflite::micro::GetTensorShape(input2),
                         tflite::micro::GetTensorData<int32_t>(input2),
                         tflite::micro::GetTensorShape(output),
                         tflite::micro::GetTensorData<int32_t>(output));
    }
  } else if (input1->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, op_params.input1_offset, 0);
    TF_LITE_ENSURE_EQ(context, op_params.input2_offset, 0);
    TF_LITE_ENSURE_EQ(context, op_params.output_offset, 0);

    if (need_broadcast) {
      reference_integer_ops::BroadcastMul4DSlow(
          op_params, tflite::micro::GetTensorShape(input1),
          tflite::micro::GetTensorData<int16_t>(input1),
          tflite::micro::GetTensorShape(input2),
          tflite::micro::GetTensorData<int16_t>(input2),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(output));
    } else {
      reference_integer_ops::Mul(op_params,
                                 tflite::micro::GetTensorShape(input1),
                                 tflite::micro::GetTensorData<int16_t>(input1),
                                 tflite::micro::GetTensorShape(input2),
                                 tflite::micro::GetTensorData<int16_t>(input2),
                                 tflite::micro::GetTensorShape(output),
                                 tflite::micro::GetTensorData<int16_t>(output));
    }
  }
  return kTfLiteOk;
}

void EvalMulFloatReference(TfLiteContext* context, TfLiteNode* node,
                           TfLiteMulParams* params, const OpDataMul* data,
                           const TfLiteEvalTensor* input1,
                           const TfLiteEvalTensor* input2,
                           TfLiteEvalTensor* output) {
  tflite::ArithmeticParams op_params = {};
  op_params.float_activation_min = data->output_activation_min_f32;
  op_params.float_activation_max = data->output_activation_max_f32;

  bool need_broadcast = reference_ops::ProcessBroadcastShapes(
      tflite::micro::GetTensorShape(input1),
      tflite::micro::GetTensorShape(input2), &op_params);

  if (need_broadcast) {
    reference_ops::BroadcastMul4DSlow(
        op_params, tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorData<float>(input1),
        tflite::micro::GetTensorShape(input2),
        tflite::micro::GetTensorData<float>(input2),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<float>(output));
  } else {
    reference_ops::Mul(op_params, tflite::micro::GetTensorShape(input1),
                       tflite::micro::GetTensorData<float>(input1),
                       tflite::micro::GetTensorShape(input2),
                       tflite::micro::GetTensorData<float>(input2),
                       tflite::micro::GetTensorShape(output),
                       tflite::micro::GetTensorData<float>(output));
  }
}

bool EvalMulSiluLut(const OpDataMul* data, const TfLiteEvalTensor* input1,
                    const TfLiteEvalTensor* input2, TfLiteEvalTensor* output) {
  if (data->silu_lut_input_index == kNoSiluLutInput) {
    return false;
  }

  const TfLiteEvalTensor* silu_input =
      data->silu_lut_input_index == kSiluLutInput1 ? input1 : input2;
  const int flat_size = NumElements(output->dims);

  if (output->type == kTfLiteInt8 && data->silu_lut_int8 != nullptr) {
    const int8_t* input_data =
        tflite::micro::GetTensorData<int8_t>(silu_input);
    int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);
    for (int i = 0; i < flat_size; ++i) {
      output_data[i] = data->silu_lut_int8[input_data[i] + 128];
    }
    return true;
  }

  if (output->type == kTfLiteFloat32 && data->silu_lut_float != nullptr) {
    const float* input_data = tflite::micro::GetTensorData<float>(silu_input);
    float* output_data = tflite::micro::GetTensorData<float>(output);
    for (int i = 0; i < flat_size; ++i) {
      output_data[i] = EvalFloatSiluLut(input_data[i], data->silu_lut_float);
    }
    return true;
  }

  return false;
}

}  // namespace tflite
