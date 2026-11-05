# Pico 2 firmware assets

Copy `model_data.cc/.h` into a new Pico TFLM target. The JSON fixture contains the fixed, already-quantized three-input test samples and FP32 SOH labels. Register only the operators reported by the TFLite analyzer, then record AllocateTensors status, arena usage, repeated latency, and serial output. Do not report MCU deployment success until the generated UF2 runs on the board.
