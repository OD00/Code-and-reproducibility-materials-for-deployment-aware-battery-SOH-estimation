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

#include "tensorflow/lite/micro/micro_interpreter_graph.h"

#include "third_party/flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include <cstdio>
#include "tensorflow/lite/schema/schema_generated.h"

#ifdef USE_TFLM_COMPRESSION

#include "tensorflow/lite/micro/micro_context.h"

#endif  // USE_TFLM_COMPRESSION

namespace tflite {
namespace {

const char* OpNameFromRegistration(const TFLMRegistration* registration) {
  if (registration->builtin_code == BuiltinOperator_CUSTOM) {
    return registration->custom_name;
  } else {
    return EnumNameBuiltinOperator(BuiltinOperator(registration->builtin_code));
  }
}

#ifdef TFLM_ENABLE_TENSOR_DUMP
bool ShouldDumpFullTensor(int tensor_index) {
  switch (tensor_index) {
    case 150:  // input to tensor 151 Mean
    case 137:  // first SE output before MaxPool1d
    case 138:  // reshape before MaxPool1d
    case 139:  // MaxPool1d output
    case 140:  // FE second stage conv branch 0
    case 141:  // FE second stage branch 0 reshape
    case 142:  // FE second stage conv branch 1
    case 143:  // FE second stage branch 1 reshape
    case 144:  // FE second stage conv branch 2
    case 145:  // FE second stage branch 2 reshape
    case 146:  // FE second stage concat
    case 147:  // FE second stage sub
    case 148:  // FE second stage mul
    case 149:  // FE second stage mul_1
    case 169:  // input to tensor 170 Mean
    case 156:  // second SE output before MaxPool1d_1
    case 157:  // reshape before MaxPool1d_1
    case 158:  // MaxPool1d_1 output
    case 159:  // FE third stage conv branch 0
    case 160:  // FE third stage branch 0 reshape
    case 161:  // FE third stage conv branch 1
    case 162:  // FE third stage branch 1 reshape
    case 163:  // FE third stage conv branch 2
    case 164:  // FE third stage branch 2 reshape
    case 165:  // FE third stage concat
    case 166:  // FE third stage sub
    case 167:  // FE third stage mul
    case 168:  // FE third stage mul_1
    case 175:  // source for FE slice/Mean tensors
    case 176:  // input to tensor 177 Mean
    case 178:  // input to tensor 179 Mean
    case 180:  // input to tensor 181 Mean
    case 182:  // input to tensor 183 Mean
    case 184:  // concat before final FE transpose
    case 185:  // transpose before final FE reshape
    case 186:  // FC input to tensor 187
      return true;
    default:
      return false;
  }
}

size_t TensorDumpLimit(int tensor_index, size_t buffer_size) {
  if (ShouldDumpFullTensor(tensor_index)) {
    return buffer_size;
  }
  return buffer_size < 1800 ? buffer_size : 1800;
}
#endif  // TFLM_ENABLE_TENSOR_DUMP

}  // namespace

extern "C" void C3MaybeDumpEvalTensor(int op_index, int tensor_index,
                                      const TfLiteEvalTensor* tensor)
    __attribute__((weak));

MicroInterpreterGraph::MicroInterpreterGraph(
    TfLiteContext* context, const Model* model, MicroAllocator* allocator,
    MicroResourceVariables* resource_variables)
    : context_(context),
      model_(model),
      allocator_(allocator),
      current_subgraph_index_(0),
      current_operator_index_(0),
      resource_variables_(resource_variables) {
  if (model != nullptr) {
    subgraphs_ = model->subgraphs();
  }
}

MicroInterpreterGraph::~MicroInterpreterGraph() {}

TfLiteStatus MicroInterpreterGraph::InitSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;
  uint32_t previous_operator_idx = current_operator_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (current_operator_index_ = 0; current_operator_index_ < operators_size;
         ++current_operator_index_) {
      TfLiteNode* node = &(subgraph_allocations_[subgraph_idx]
                               .node_and_registrations[current_operator_index_]
                               .node);
      const TFLMRegistration* registration =
          subgraph_allocations_[subgraph_idx]
              .node_and_registrations[current_operator_index_]
              .registration;
      size_t init_data_size;
      const char* init_data;
      if (registration->builtin_code == BuiltinOperator_CUSTOM) {
        init_data = reinterpret_cast<const char*>(node->custom_initial_data);
        init_data_size = node->custom_initial_data_size;
      } else {
        init_data = reinterpret_cast<const char*>(node->builtin_data);
        init_data_size = 0;
      }
      if (registration->init) {
        node->user_data =
            registration->init(context_, init_data, init_data_size);
      }
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;
  current_operator_index_ = previous_operator_idx;

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::PrepareSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;
  uint32_t previous_operator_idx = current_operator_index_;
  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (current_operator_index_ = 0; current_operator_index_ < operators_size;
         ++current_operator_index_) {
      TfLiteNode* node = &(subgraph_allocations_[subgraph_idx]
                               .node_and_registrations[current_operator_index_]
                               .node);
      const TFLMRegistration* registration =
          subgraph_allocations_[subgraph_idx]
              .node_and_registrations[current_operator_index_]
              .registration;
      if (registration->prepare != nullptr) {
        TfLiteStatus prepare_status = registration->prepare(context_, node);
        if (prepare_status != kTfLiteOk) {
          MicroPrintf("Node %s (number %df) failed to prepare with status %d",
                      OpNameFromRegistration(registration),
                      current_operator_index_, prepare_status);
          return kTfLiteError;
        }
#ifdef USE_TFLM_COMPRESSION
        GetMicroContext(context_)->ResetDecompressionMemoryAllocations();
#endif  // USE_TFLM_COMPRESSION
      }
      allocator_->FinishPrepareNodeAllocations(
          /*node_id=*/current_operator_index_);
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;
  current_operator_index_ = previous_operator_idx;
  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::ResetSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;
  uint32_t previous_operator_idx = current_operator_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (current_operator_index_ = 0; current_operator_index_ < operators_size;
         ++current_operator_index_) {
      TfLiteNode* node = &(subgraph_allocations_[subgraph_idx]
                               .node_and_registrations[current_operator_index_]
                               .node);
      const TFLMRegistration* registration =
          subgraph_allocations_[subgraph_idx]
              .node_and_registrations[current_operator_index_]
              .registration;
      // registration is allocated outside the interpreter, so double check to
      // make sure it's not nullptr;
      if (registration != nullptr && registration->reset != nullptr) {
        registration->reset(context_, node->user_data);
      }
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;
  current_operator_index_ = previous_operator_idx;

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::FreeSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;
  uint32_t previous_operator_idx = current_operator_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (current_operator_index_ = 0; current_operator_index_ < operators_size;
         ++current_operator_index_) {
      TfLiteNode* node = &(subgraph_allocations_[subgraph_idx]
                               .node_and_registrations[current_operator_index_]
                               .node);
      const TFLMRegistration* registration =
          subgraph_allocations_[subgraph_idx]
              .node_and_registrations[current_operator_index_]
              .registration;
      // registration is allocated outside the interpreter, so double check to
      // make sure it's not nullptr;
      if (registration != nullptr && registration->free != nullptr) {
        registration->free(context_, node->user_data);
      }
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;
  current_operator_index_ = previous_operator_idx;

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::InvokeSubgraph(int subgraph_idx) {
  int previous_subgraph_idx = current_subgraph_index_;
  uint32_t previous_operator_idx = current_operator_index_;
  current_subgraph_index_ = subgraph_idx;

  if (static_cast<size_t>(subgraph_idx) >= subgraphs_->size()) {
    MicroPrintf("Accessing subgraph %d but only %d subgraphs found",
                subgraph_idx, subgraphs_->size());
    return kTfLiteError;
  }
  uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
  for (current_operator_index_ = 0; current_operator_index_ < operators_size;
       ++current_operator_index_) {
    TfLiteNode* node = &(subgraph_allocations_[subgraph_idx]
                             .node_and_registrations[current_operator_index_]
                             .node);
    const TFLMRegistration* registration =
        subgraph_allocations_[subgraph_idx]
            .node_and_registrations[current_operator_index_]
            .registration;

// This ifdef is needed (even though ScopedMicroProfiler itself is a no-op with
// -DTF_LITE_STRIP_ERROR_STRINGS) because the function OpNameFromRegistration is
// only defined for builds with the error strings.
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
    ScopedMicroProfiler scoped_profiler(
        OpNameFromRegistration(registration),
        reinterpret_cast<MicroProfilerInterface*>(context_->profiler));
#endif

#ifdef TFLM_ENABLE_TENSOR_DUMP
    // --- INPUT TENSOR DUMP ---
    static bool dumped_tensors[532] = {};
    for (int k = 0; k < node->inputs->size; ++k) {
      int t_idx = node->inputs->data[k];
      if (t_idx >= 0 && t_idx < 532 && !dumped_tensors[t_idx]) {
        TfLiteEvalTensor* t = &subgraph_allocations_[subgraph_idx].tensors[t_idx];
        if (t != nullptr && t->type == kTfLiteInt8 && t->data.int8 != nullptr) {
          size_t buf_size = 0;
          TfLiteEvalTensorByteLength(t, &buf_size);
          printf("I%zu T%d %zu ", static_cast<size_t>(current_operator_index_), t_idx, buf_size);
          uint8_t* p = reinterpret_cast<uint8_t*>(t->data.int8);
          size_t limit = TensorDumpLimit(t_idx, buf_size);
          for (size_t j = 0; j < limit; ++j) printf("%02x", p[j]);
          printf("\n");
        }
        dumped_tensors[t_idx] = true;
      }
    }
    // ---------------------------
#endif  // TFLM_ENABLE_TENSOR_DUMP
  TFLITE_DCHECK(registration->invoke);
    TfLiteStatus invoke_status = registration->invoke(context_, node);

    if (C3MaybeDumpEvalTensor != nullptr && invoke_status == kTfLiteOk) {
      for (int k = 0; k < node->outputs->size; ++k) {
        const int tensor_index = node->outputs->data[k];
        if (tensor_index >= 0) {
          C3MaybeDumpEvalTensor(
              current_operator_index_, tensor_index,
              &subgraph_allocations_[subgraph_idx].tensors[tensor_index]);
        }
      }
    }

#ifdef TFLM_ENABLE_TENSOR_DUMP
    // FORCE CAPTURE: tensor 186 (FC input to tensor 187)
    for (int k = 0; k < node->outputs->size; ++k) {
      if (node->outputs->data[k] == 186) {
        TfLiteEvalTensor* t = &subgraph_allocations_[subgraph_idx].tensors[186];
        size_t bs = 0;
        TfLiteEvalTensorByteLength(t, &bs);
        printf("F186 T186 %zu ", bs);
        uint8_t* p = (uint8_t*)t->data.raw;
        size_t lim = TensorDumpLimit(186, bs);
        for (size_t j = 0; j < lim; ++j) printf("%02x", p[j]);
        printf("\n");
        break;
      }
    }    // --- TENSOR DUMP after each operator ---
    for (int k = 0; k < node->outputs->size; ++k) {
      int t_idx = node->outputs->data[k];
      TfLiteEvalTensor* t = &subgraph_allocations_[subgraph_idx].tensors[t_idx];
      if (t != nullptr && t->type == kTfLiteInt8 && t->data.int8 != nullptr) {
        size_t buf_size = 0;
        TfLiteEvalTensorByteLength(t, &buf_size);
        printf("O%zu T%d %zu ", static_cast<size_t>(current_operator_index_), t_idx, buf_size);
        uint8_t* p = reinterpret_cast<uint8_t*>(t->data.int8);
        size_t limit = TensorDumpLimit(t_idx, buf_size);
        for (size_t j = 0; j < limit; ++j) printf("%02x", p[j]);
        printf("\n");
      }
    }
    fflush(stdout);
    // --------------------------------
#endif  // TFLM_ENABLE_TENSOR_DUMP

#ifdef USE_TFLM_COMPRESSION
    GetMicroContext(context_)->ResetDecompressionMemoryAllocations();
#endif  // USE_TFLM_COMPRESSION

    // All TfLiteTensor structs used in the kernel are allocated from temp
    // memory in the allocator. This creates a chain of allocations in the
    // temp section. The call below resets the chain of allocations to
    // prepare for the next call.
    allocator_->ResetTempAllocations();

    if (invoke_status == kTfLiteError) {
      MicroPrintf("Node %s (number %d) failed to invoke with status %d",
                  OpNameFromRegistration(registration), current_operator_index_,
                  invoke_status);
      return kTfLiteError;
    } else if (invoke_status != kTfLiteOk) {
      return invoke_status;
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;
  current_operator_index_ = previous_operator_idx;
  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::ResetVariableTensors() {
  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    const SubGraph* subgraph = (*subgraphs_)[subgraph_idx];
    for (size_t i = 0; i < subgraph->tensors()->size(); ++i) {
      auto* tensor = subgraph->tensors()->Get(i);
      if (tensor->is_variable()) {
        size_t buffer_size;
        TF_LITE_ENSURE_STATUS(TfLiteEvalTensorByteLength(
            &subgraph_allocations_[subgraph_idx].tensors[i], &buffer_size));

        int value = 0;
        if (tensor->type() == tflite::TensorType_INT8) {
          value = tensor->quantization()->zero_point()->Get(0);
        }
        memset(subgraph_allocations_[subgraph_idx].tensors[i].data.raw, value,
               buffer_size);
      }
    }
  }
  if (resource_variables_ != nullptr) {
    resource_variables_->ResetAll();
  }

  return kTfLiteOk;
}

int MicroInterpreterGraph::NumSubgraphs() {
  return model_->subgraphs()->size();
}

void MicroInterpreterGraph::SetSubgraphAllocations(
    SubgraphAllocations* subgraph_allocations) {
  subgraph_allocations_ = subgraph_allocations;
}

size_t MicroInterpreterGraph::NumSubgraphInputs(int subgraph_idx) {
  return model_->subgraphs()->Get(subgraph_idx)->inputs()->size();
}

TfLiteEvalTensor* MicroInterpreterGraph::GetSubgraphInput(int subgraph_idx,
                                                          int input_idx) {
  int tensor_idx =
      model_->subgraphs()->Get(subgraph_idx)->inputs()->Get(input_idx);
  return &subgraph_allocations_[subgraph_idx].tensors[tensor_idx];
}

size_t MicroInterpreterGraph::NumSubgraphOutputs(int subgraph_idx) {
  return model_->subgraphs()->Get(subgraph_idx)->outputs() == nullptr
             ? 0
             : model_->subgraphs()->Get(subgraph_idx)->outputs()->size();
}

TfLiteEvalTensor* MicroInterpreterGraph::GetSubgraphOutput(int subgraph_idx,
                                                           int output_idx) {
  int tensor_idx =
      model_->subgraphs()->Get(subgraph_idx)->outputs()->Get(output_idx);
  return &subgraph_allocations_[subgraph_idx].tensors[tensor_idx];
}

}  // namespace tflite





