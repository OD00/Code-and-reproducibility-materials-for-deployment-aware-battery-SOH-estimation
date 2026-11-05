#pragma once
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
constexpr int kModelOperatorCapacity = 22;
inline TfLiteStatus RegisterModelOps(
    tflite::MicroMutableOpResolver<kModelOperatorCapacity>& resolver) {
  if (resolver.AddFullyConnected() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddPack() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddReshape() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddTranspose() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddPad() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddConv2D() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddMean() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddSquaredDifference() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddAdd() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddRsqrt() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddSub() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddMul() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddLogistic() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddSplit() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddStridedSlice() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddDepthwiseConv2D() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddSplitV() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddExp() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddLog() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddSum() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddBatchMatMul() != kTfLiteOk) return kTfLiteError;
  if (resolver.AddSoftmax() != kTfLiteOk) return kTfLiteError;
  return kTfLiteOk;
}
