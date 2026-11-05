# Battery SOH experiments

CALCE/TJU data preparation, model training, full-integer INT8 quantization, and Raspberry Pi Pico 2 inference.

## Files

```text
preprocessing/       Raw records, cycle pairing, quality control and resampling
src/                 PyTorch models and CALCE training utilities
paper_experiments/   TJU training/fine-tuning, INT8 conversion and serial analysis
scripts/             Command-line entry points
firmware/            TFLM/CMSIS-NN sources and four Pico 2 examples
artifacts/           Deployment checkpoints, INT8 models and desktop predictions
data/                Dataset layout; download data separately
requirements-*.txt   Dependencies for each stage
```

Inputs use 12 charging cycles: voltage/current `[N,12,256]` and duration `[N,12,1]`. The target is the last cycle's SOH, `Q_discharge / Q_nominal`.

`src/models/` defines the four PyTorch model families; `src/common/` provides shared training and evaluation utilities, and `src/A1/` plus `src/A2/` implement CALCE loading, validation and training. `paper_experiments/experiment_2_rate_finetune/` handles TJU source training and supervised fine-tuning. `paper_experiments/experiment_3_mcu_deployment/` contains TensorFlow mapping, INT8 conversion, firmware asset generation and serial analysis.

`artifacts/checkpoints/` contains the four fixed deployment checkpoints and their configurations. `artifacts/int8/` contains the corresponding TFLite models and desktop predictions used for board comparison. Raw datasets must be downloaded separately; new processing, training and capture results are written to `outputs/`.

## Included TFLM runtime

TensorFlow Lite for Microcontrollers (TFLM) is included as C/C++ source in the Pico port, together with the runtime's bundled dependencies:

```text
firmware/pico_tflmicro/
  src/
    tensorflow/       TFLM interpreter, memory allocation and operator kernels
    signal/           Signal-processing sources
    third_party/
      cmsis/          ARM CMSIS headers
      cmsis_nn/       Optimized ARM neural-network kernels
      flatbuffers/    TFLite model parsing
      gemmlowp/       Quantized arithmetic helpers
      kissfft/        FFT dependency
      ruy/            Matrix-operation helpers
  examples/           Four model-specific Pico 2 firmware examples
  CMakeLists.txt      Runtime library and firmware build configuration
  pico_sdk_import.cmake
```

Each example includes the embedded INT8 model, operator resolver, quantized test inputs and firmware entry points. The bundled runtime is compiled into the firmware; supported operators use CMSIS-NN and other paths retain generic/reference kernels.

Pico SDK, the ARM cross-compiler, CMake, Ninja and picotool are external tools and must be installed separately. Desktop Python packages such as PyTorch, TensorFlow and Mamba are also installed separately using the stage-specific requirements below.

## Setup

Use Python 3.10 and install the requirements for the stage you need:

```bash
python -m pip install -r requirements-preprocessing.txt
python -m pip install -r requirements-training.txt
python -m pip install -r requirements-quantization.txt
python -m pip install -r requirements-serial.txt
```

Training requires a compatible PyTorch/CUDA/Mamba installation. Several conversion and analysis modules also import Mamba through the shared model registry. Preprocessing and serial capture can use separate environments.

Download CALCE and TJU Dataset 1 using [data/README.md](data/README.md). Run commands from the repository root. Use `--help` for arguments and `--dry-run` to inspect wrapper commands.

## Preprocess and train

```bash
python scripts/preprocess.py
python scripts/train.py calce --output-root outputs/calce_run1
python scripts/train.py tju --output-root outputs/tju_run1
```

CALCE uses four held-out-cell folds and seeds 42/123/2026. TJU uses 0.25C source cells, 0.5C target cells, fixed target validation cell #1 and test cell #14. Completed full runs are aggregated automatically. Existing training output directories are protected from overwrite.

## Quantize

```bash
python scripts/quantize.py --prepared-dir outputs/deployment_run1
```

The wrapper uses the bundled seed-42/CS2_38 CALCE checkpoints and processed CALCE data. It prepares 384 source-only calibration windows, converts four models to INT8, and generates 64-window firmware fixtures. CNN-LSTM uses the fused conversion in `optimization_tests/`; other models use the manual TensorFlow mapping.

## Build Pico 2 firmware

Install CMake, Ninja, an ARM cross-compiler, Pico SDK and picotool, then supply the local paths:

```bash
python scripts/build_firmware.py --sdk /path/to/pico-sdk --toolchain /path/to/arm-toolchain/bin
```

The four examples include model arrays and test inputs, so building does not require retraining. For an offline picotool package, add `--picotool-dir /path/to/directory-containing-picotoolConfig.cmake`. UF2 files are written under the printed build directory's `examples/<target>/`. On Windows with a non-ASCII repository path, the wrapper uses a temporary build directory; `--build-dir` overrides it.

| Model | Firmware target | Configured Arena |
|---|---|---:|
| CNN-LSTM | `paper_exp3_cnnlstmsimplecnn_fused` | 448 KiB |
| CNN-Mamba | `paper_exp3_cnnmambasimplecnn` | 448 KiB |
| CNN-Transformer | `paper_exp3_cnntransformer` | 448 KiB |
| ResNet | `paper_exp3_resnet1d` | 360 KiB |

## Capture and analyze USB serial output

Flash the selected UF2 using BOOTSEL. Start capture and reset the board:

```bash
python scripts/serial_capture.py --port COM6 --output outputs/serial/resnet.txt
```

Use the actual port, such as `COM6` or `/dev/ttyACM0`. The firmware runs embedded inputs and sends results through USB CDC; the PC does not stream input samples. Capture ends at `timing runs=`.

```bash
python -m paper_experiments.experiment_3_mcu_deployment.analyze_pico2_serial --log outputs/serial/resnet.txt --fixture firmware/pico_tflmicro/examples/paper_exp3_resnet1d/test_fixture.json --predictions artifacts/int8/ResNet1d/int8_test_predictions.csv --configured-arena-bytes 368640 --output outputs/serial/resnet_result.json
```

Change the fixture, prediction file and Arena for other models. Timing order is 64 accuracy calls, the call named `cold_us`, 10 warmups and 100 measured calls. The `cold_us` field is not the first invocation after startup.

Third-party source headers and license files remain in the firmware runtime and its `src/third_party/` directories. Raw/intermediate datasets and generated outputs are ignored by Git.
