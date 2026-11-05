#pragma once
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
constexpr int kModelOperatorCapacity = 10;
inline TfLiteStatus RegisterModelOps(
    tflite::MicroMutableOpResolver<kModelOperatorCapacity>& resolver) {
  if (resolver.AddFullyConnected() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddPack() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddReshape() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddTranspose() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddPad() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddConv2D() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddMean() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddAdd() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddUnidirectionalSequenceLSTM() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddStridedSlice() != kTfLiteOk) return kTfLiteError;
  return kTfLiteOk;
}
