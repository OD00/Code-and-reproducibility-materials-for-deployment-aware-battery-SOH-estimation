#include "main_functions.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "hardware/clocks.h"
#include "hardware/timer.h"
#include "pico/stdlib.h"
#include "model_data.h"
#include "model_ops_resolver.h"
#include "test_fixture.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
constexpr int kTensorArenaBytes = 448 * 1024;
constexpr int kWarmupRuns = 10;
constexpr int kMeasuredRuns = 100;
alignas(16) uint8_t tensor_arena[kTensorArenaBytes];
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* voltage = nullptr;
TfLiteTensor* current = nullptr;
TfLiteTensor* duration = nullptr;
TfLiteTensor* output = nullptr;
bool ready = false;
uint64_t timings[kMeasuredRuns];

bool LoadFixture(int sample) {
  if (voltage->bytes != kVoltageElementsPerSample ||
      current->bytes != kCurrentElementsPerSample ||
      duration->bytes != kDurationElementsPerSample) {
    MicroPrintf("fixture length mismatch voltage=%d/%d current=%d/%d duration=%d/%d",
                voltage->bytes, kVoltageElementsPerSample,
                current->bytes, kCurrentElementsPerSample,
                duration->bytes, kDurationElementsPerSample);
    return false;
  }
  std::memcpy(voltage->data.int8,
              kVoltageInput + sample * kVoltageElementsPerSample,
              kVoltageElementsPerSample);
  std::memcpy(current->data.int8,
              kCurrentInput + sample * kCurrentElementsPerSample,
              kCurrentElementsPerSample);
  std::memcpy(duration->data.int8,
              kDurationInput + sample * kDurationElementsPerSample,
              kDurationElementsPerSample);
  return true;
}
}  // namespace

void setup() {
  tflite::InitializeTarget();
  // Keep the early allocator and readiness messages observable after the UF2
  // reboot and USB CDC enumeration.
  sleep_ms(3000);
  MicroPrintf("firmware build=%s sys_clock_hz=%u", PICO_CMAKE_BUILD_TYPE,
              clock_get_hz(clk_sys));
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("schema mismatch");
    return;
  }
  // Resolver must be generated/checked against the final TFLite operator list.
  static tflite::MicroMutableOpResolver<kModelOperatorCapacity> resolver;
  if (RegisterModelOps(resolver) != kTfLiteOk) {
    MicroPrintf("operator registration failed");
    return;
  }
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaBytes);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors failed arena=%d", kTensorArenaBytes);
    return;
  }
  // The final generated full-INT8 models have a verified, fixed input order:
  // voltage=0, duration=1, current=2. TfLiteTensor in this TFLM revision does
  // not expose input names, so bind by the audited model signature.
  voltage = interpreter->input(0);
  duration = interpreter->input(1);
  current = interpreter->input(2);
  output = interpreter->output(0);
  if (!voltage || !current || !duration || !output) {
    MicroPrintf("tensor binding failed");
    return;
  }
  if (!LoadFixture(0)) return;
  ready = true;
  MicroPrintf("model ready samples=%d arena_used=%d",
              kTestSampleCount, interpreter->arena_used_bytes());
}

void loop() {
  if (!ready) return;
  for (int sample = 0; sample < kTestSampleCount; ++sample) {
    if (!LoadFixture(sample) || interpreter->Invoke() != kTfLiteOk) {
      MicroPrintf("accuracy Invoke failed sample=%d", sample);
      return;
    }
    const int8_t yq = output->data.int8[0];
    const float y =
        (static_cast<int>(yq) - output->params.zero_point) * output->params.scale;
    MicroPrintf("sample=%d yq=%d y=%f target=%f", sample, yq, y,
                kTargetSoh[sample]);
  }
  if (!LoadFixture(0)) return;
  const uint64_t cold_start = time_us_64();
  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("cold Invoke failed");
    return;
  }
  const uint64_t cold_us = time_us_64() - cold_start;
  for (int index = 0; index < kWarmupRuns; ++index) interpreter->Invoke();
  uint64_t total = 0;
  for (int index = 0; index < kMeasuredRuns; ++index) {
    const uint64_t start = time_us_64();
    if (interpreter->Invoke() != kTfLiteOk) {
      MicroPrintf("Invoke failed");
      return;
    }
    timings[index] = time_us_64() - start;
    total += timings[index];
  }
  std::sort(timings, timings + kMeasuredRuns);
  const double mean = static_cast<double>(total) / kMeasuredRuns;
  double squared_difference_sum = 0.0;
  for (int index = 0; index < kMeasuredRuns; ++index) {
    const double difference = static_cast<double>(timings[index]) - mean;
    squared_difference_sum += difference * difference;
  }
  const double standard_deviation =
      std::sqrt(squared_difference_sum / (kMeasuredRuns - 1));
  const int8_t yq = output->data.int8[0];
  const float y = (static_cast<int>(yq) - output->params.zero_point) * output->params.scale;
  MicroPrintf("timing runs=%d cold_us=%llu mean_us=%f sd_us=%f p50_us=%llu p95_us=%llu max_us=%llu yq=%d y=%f",
              kMeasuredRuns, cold_us, mean, standard_deviation,
              timings[49], timings[94], timings[kMeasuredRuns - 1], yq, y);
  ready = false;
}
