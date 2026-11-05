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

#include "tensorflow/lite/micro/kernels/gelu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

const int kGeluInputTensor = 0;
const int kGeluOutputTensor = 0;

namespace {

// GELU activation: x * P(X <= x) where P ~ N(0, 1)
// Exact:  0.5 * x * (1 + erf(x / sqrt(2)))
// Approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// We use the tanh approximation for both paths since erf() is typically
// unavailable on embedded targets. The error is < 0.1% for all x.

inline float GeluFloat(float x, bool approximate) {
  if (approximate) {
    static constexpr float kAlpha = 0.7978845608f;  // sqrt(2/pi)
    static constexpr float kBeta = 0.044715f;
    const float x3 = x * x * x;
    const float inner = kAlpha * (x + kBeta * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
  } else {
    // exact: 0.5 * x * (1 + erf(x/sqrt(2)))
    // Use closer approximation via tanh with adjusted parameters.
    // erf(x/sqrt(2)) ≈ tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2))
    static constexpr float kAlpha = 0.7978845608f;
    return 0.5f * x * (1.0f + tanhf(kAlpha * x * (1.0f + 0.044715f * x * x)));
  }
}

void GeluEvalFloat(const TfLiteEvalTensor* input, TfLiteEvalTensor* output,
                   bool approximate) {
  const float* input_data = tflite::micro::GetTensorData<float>(input);
  float* output_data = tflite::micro::GetTensorData<float>(output);
  const int flat_size =
      MatchingFlatSize(tflite::micro::GetTensorShape(input),
                       tflite::micro::GetTensorShape(output));
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = GeluFloat(input_data[i], approximate);
  }
}

void PopulateLookupTableInt8(const TfLiteTensor* input, const TfLiteTensor* output,
                             bool approximate, OpDataGelu* data) {
  const float inverse_scale = 1.0f / output->params.scale;
  const int32_t maxval = std::numeric_limits<int8_t>::max();
  const int32_t minval = std::numeric_limits<int8_t>::min();
  for (int32_t val = minval; val <= maxval; ++val) {
    const float dequantized =
        input->params.scale * (val - input->params.zero_point);
    const float transformed = GeluFloat(dequantized, approximate);
    const float rescaled = nearbyintf(transformed * inverse_scale);
    const int32_t quantized =
        static_cast<int32_t>(rescaled + output->params.zero_point);
    data->table[static_cast<uint8_t>(static_cast<int8_t>(val))] =
        static_cast<int8_t>(std::max(std::min(maxval, quantized), minval));
  }
}

void EvalInt8UsingLookupTable(const OpDataGelu* data,
                              const TfLiteEvalTensor* input,
                              TfLiteEvalTensor* output) {
  const int flat_size =
      MatchingFlatSize(tflite::micro::GetTensorShape(input),
                       tflite::micro::GetTensorShape(output));
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = data->table[static_cast<uint8_t>(input_data[i])];
  }
}

void* GeluInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataGelu));
}

TfLiteStatus GeluPrepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kGeluInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kGeluOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  // Read the approximate flag from the builtin data (parsed by ParseGelu).
  bool approximate = false;
  if (node->builtin_data != nullptr) {
    TfLiteGeluParams* params = static_cast<TfLiteGeluParams*>(node->builtin_data);
    approximate = params->approximate;
  }

  if (input->type == kTfLiteInt8) {
    OpDataGelu* data = static_cast<OpDataGelu*>(node->user_data);
    PopulateLookupTableInt8(input, output, approximate, data);
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus GeluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kGeluInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kGeluOutputTensor);

  bool approximate = false;
  if (node->builtin_data != nullptr) {
    TfLiteGeluParams* params =
        static_cast<TfLiteGeluParams*>(node->builtin_data);
    approximate = params->approximate;
  }

  switch (input->type) {
    case kTfLiteFloat32: {
      GeluEvalFloat(input, output, approximate);
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      const OpDataGelu* data = static_cast<OpDataGelu*>(node->user_data);
      EvalInt8UsingLookupTable(data, input, output);
      return kTfLiteOk;
    }
    default: {
      MicroPrintf("GELU only supports float32 and int8, got %s.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

}  // namespace

TFLMRegistration Register_GELU() {
  return tflite::micro::RegisterOp(GeluInit, GeluPrepare, GeluEval);
}

}  // namespace tflite
