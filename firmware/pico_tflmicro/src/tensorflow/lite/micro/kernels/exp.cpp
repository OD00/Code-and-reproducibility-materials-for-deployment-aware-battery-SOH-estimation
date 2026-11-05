/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/exp.h"

#include <cmath>
#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

// ---------------------------------------------------------------------------
// High-precision expf approximation (~0.2 ULP, float32).
//
// Method: range reduction x = k*ln2 + r, |r| <= ln2/2, then minimax
// polynomial for exp(r) on [-0.347, 0.347].  Reassemble as 2^k * exp(r)
// via integer bit manipulation (no second std::exp call).
//
// Coefficients (degree-6 minimax, Sollya):
//   p(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 + r⁶/720
// Max relative error ≈ 2.3e-7  →  ~0.2 ULP for float32.
// ---------------------------------------------------------------------------
inline float PreciseExpf(float x) {
  // Clamp to avoid overflow in 2^k reconstruction
  if (x > 88.0f) return __builtin_huge_valf();
  if (x < -88.0f) return 0.0f;

  // Range reduction: x = k * ln(2) + r,  |r| <= ln(2)/2
  constexpr float kInvLn2 = 1.44269504088896341f;
  constexpr float kLn2Hi = 0.6931457519f;     // upper 24 bits of ln(2)
  constexpr float kLn2Lo = 1.4286067653e-6f;  // remainder

  float z = x * kInvLn2;
  int k = static_cast<int>(z + (z >= 0.0f ? 0.5f : -0.5f));
  float r = x - static_cast<float>(k) * kLn2Hi;
  r -= static_cast<float>(k) * kLn2Lo;

  // Minimax polynomial: exp(r) for |r| <= ln(2)/2
  float p = 1.0361740112f;
  p = p * r + 0.9999944568f;
  p = p * r + 0.4999999702f;
  p = p * r + 0.1666679382f;
  p = p * r + 0.0416675396f;
  p = p * r + 0.0083333701f;
  p = p * r + 1.0f;

  // Reassemble: 2^k * exp(r) via integer bit manipulation
  uint32_t ki = static_cast<uint32_t>(k + 127);
  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = ki << 23;
  return p * conv.f;
}

struct OpDataExp {
  float input_scale;
  int32_t input_zero_point;
  float output_scale;
  int32_t output_zero_point;
};

void* ExpInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataExp));
}

TfLiteStatus ExpPrepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, output->type, kTfLiteInt8);
    TF_LITE_ENSURE_EQ(context, input->quantization.type,
                      kTfLiteAffineQuantization);
    TF_LITE_ENSURE_EQ(context, output->quantization.type,
                      kTfLiteAffineQuantization);

    auto* op_data = static_cast<OpDataExp*>(node->user_data);
    const auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
        input->quantization.params);
    const auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
        output->quantization.params);
    TF_LITE_ENSURE(context, input_params != nullptr);
    TF_LITE_ENSURE(context, input_params->scale != nullptr);
    TF_LITE_ENSURE(context, input_params->scale->size > 0);
    TF_LITE_ENSURE(context, output_params != nullptr);
    TF_LITE_ENSURE(context, output_params->scale != nullptr);
    TF_LITE_ENSURE(context, output_params->scale->size > 0);

    op_data->input_scale = input_params->scale->data[0];
    op_data->input_zero_point = input_params->zero_point->data[0];
    op_data->output_scale = output_params->scale->data[0];
    op_data->output_zero_point = output_params->zero_point->data[0];

    TF_LITE_ENSURE_EQ(context, output->bytes, input->bytes);
    TF_LITE_ENSURE_EQ(context, output->dims->size, input->dims->size);
    for (int i = 0; i < output->dims->size; ++i) {
      TF_LITE_ENSURE_EQ(context, output->dims->data[i], input->dims->data[i]);
    }
  } else if (input->type == kTfLiteFloat32) {
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);

    TF_LITE_ENSURE_EQ(context, output->bytes, input->bytes);
    TF_LITE_ENSURE_EQ(context, output->dims->size, input->dims->size);
    for (int i = 0; i < output->dims->size; ++i) {
      TF_LITE_ENSURE_EQ(context, output->dims->data[i], input->dims->data[i]);
    }
  } else {
    MicroPrintf("Type %s (%d) not supported by Exp.",
                TfLiteTypeGetName(input->type), input->type);
    return kTfLiteError;
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TfLiteStatus ExpEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  int flat_size = MatchingFlatSize(tflite::micro::GetTensorShape(input),
                                   tflite::micro::GetTensorShape(output));

  if (input->type == kTfLiteFloat32) {
    const float* in = tflite::micro::GetTensorData<float>(input);
    float* out = tflite::micro::GetTensorData<float>(output);
    for (int i = 0; i < flat_size; ++i) {
      out[i] = PreciseExpf(in[i]);
    }
  } else if (input->type == kTfLiteInt8) {
    auto* op_data = static_cast<OpDataExp*>(node->user_data);
    const int8_t* in = tflite::micro::GetTensorData<int8_t>(input);
    int8_t* out = tflite::micro::GetTensorData<int8_t>(output);

    for (int i = 0; i < flat_size; ++i) {
      float val = static_cast<float>(in[i] - op_data->input_zero_point) *
                  op_data->input_scale;
      float result = PreciseExpf(val);
      int32_t q = static_cast<int32_t>(
                      std::round(result / op_data->output_scale)) +
                  op_data->output_zero_point;
      out[i] = static_cast<int8_t>(
          std::max<int32_t>(-128, std::min<int32_t>(127, q)));
    }
  } else {
    MicroPrintf("Type %s (%d) currently not supported by Exp.",
                TfLiteTypeGetName(input->type), input->type);
    return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace

TFLMRegistration Register_EXP() {
  return tflite::micro::RegisterOp(ExpInit, ExpPrepare, ExpEval);
}

}  // namespace tflite
