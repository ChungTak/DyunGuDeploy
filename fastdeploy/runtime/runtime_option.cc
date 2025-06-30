// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fastdeploy/runtime/runtime.h"
#include "fastdeploy/utils/unique_ptr.h"
#include "fastdeploy/utils/utils.h"

namespace fastdeploy {

void RuntimeOption::SetModelPath(const std::string& model_path,
                                 const std::string& params_path,
                                 const ModelFormat& format) {
  model_file = model_path;
  params_file = params_path;
  model_format = format;
  model_from_memory_ = false;
}

void RuntimeOption::SetModelBuffer(const std::string& model_buffer,
                                   const std::string& params_buffer,
                                   const ModelFormat& format) {
  model_file = model_buffer;
  params_file = params_buffer;
  model_format = format;
  model_from_memory_ = true;
}

void RuntimeOption::SetEncryptionKey(const std::string& encryption_key) {
#ifdef ENABLE_ENCRYPTION
  encryption_key_ = encryption_key;
#else
  FDERROR << "The FastDeploy didn't compile with encryption function."
          << std::endl;
#endif
}

void RuntimeOption::UseGpu(int gpu_id) {
#ifdef WITH_GPU
  device = Device::GPU;
  device_id = gpu_id;

#else
  FDWARNING << "The FastDeploy didn't compile with GPU, will force to use CPU."
            << std::endl;
  device = Device::CPU;
#endif
}

void RuntimeOption::UseCpu() { device = Device::CPU; }

void RuntimeOption::UseRKNPU2(fastdeploy::rknpu2::CpuName rknpu2_name,
                              fastdeploy::rknpu2::CoreMask rknpu2_core) {
  rknpu2_option.cpu_name = rknpu2_name;
  rknpu2_option.core_mask = rknpu2_core;
  device = Device::RKNPU;
}

void RuntimeOption::UseHorizon() { device = Device::SUNRISENPU; }

void RuntimeOption::UseIpu(int device_num, int micro_batch_size,
                           bool enable_pipelining, int batches_per_step) {
  FDWARNING << "IPU device support has been removed from FastDeploy, will force to use CPU."
            << std::endl;
  device = Device::CPU;
}

void RuntimeOption::UseSophgo() {
  device = Device::SOPHGOTPUD;
  UseSophgoBackend();
}

void RuntimeOption::SetExternalStream(void* external_stream) {
  external_stream_ = external_stream;
}

void RuntimeOption::SetCpuThreadNum(int thread_num) {
  FDASSERT(thread_num > 0, "The thread_num must be greater than 0.");
  cpu_thread_num = thread_num;
  ort_option.intra_op_num_threads = thread_num;
  openvino_option.cpu_thread_num = thread_num;
}

void RuntimeOption::SetOrtGraphOptLevel(int level) {
  FDWARNING << "`RuntimeOption::SetOrtGraphOptLevel` will be removed in "
               "v1.2.0, please modify its member variables directly, e.g "
               "`runtime_option.ort_option.graph_optimization_level = 99`."
            << std::endl;
  std::vector<int> supported_level{-1, 0, 1, 2};
  auto valid_level = std::find(supported_level.begin(), supported_level.end(),
                               level) != supported_level.end();
  FDASSERT(valid_level, "The level must be -1, 0, 1, 2.");
  ort_option.graph_optimization_level = level;
}

// use onnxruntime backend
void RuntimeOption::UseOrtBackend() {
#ifdef ENABLE_ORT_BACKEND
  backend = Backend::ORT;
#else
  FDASSERT(false, "The FastDeploy didn't compile with OrtBackend.");
#endif
}

// use sophgoruntime backend
void RuntimeOption::UseSophgoBackend() {
#ifdef ENABLE_SOPHGO_BACKEND
  backend = Backend::SOPHGOTPU;
#else
  FDASSERT(false, "The FastDeploy didn't compile with SophgoBackend.");
#endif
}

void RuntimeOption::UseTrtBackend() {
#ifdef ENABLE_TRT_BACKEND
  backend = Backend::TRT;
#else
  FDASSERT(false, "The FastDeploy didn't compile with TrtBackend.");
#endif
}

void RuntimeOption::UseOpenVINOBackend() {
#ifdef ENABLE_OPENVINO_BACKEND
  backend = Backend::OPENVINO;
#else
  FDASSERT(false, "The FastDeploy didn't compile with OpenVINO.");
#endif
}

void RuntimeOption::UseHorizonNPUBackend() {
#ifdef ENABLE_HORIZON_BACKEND
  backend = Backend::HORIZONNPU;
#else
  FDASSERT(false, "The FastDeploy didn't compile with horizon");
#endif
}

void RuntimeOption::SetOpenVINODevice(const std::string& name) {
  FDWARNING << "`RuntimeOption::SetOpenVINODevice` will be removed in v1.2.0, "
               "please use `RuntimeOption.openvino_option.SetDeivce(const "
               "std::string&)` instead."
            << std::endl;
  openvino_option.SetDevice(name);
}

void RuntimeOption::SetTrtInputShape(const std::string& input_name,
                                     const std::vector<int32_t>& min_shape,
                                     const std::vector<int32_t>& opt_shape,
                                     const std::vector<int32_t>& max_shape) {
  FDWARNING << "`RuntimeOption::SetTrtInputShape` will be removed in v1.2.0, "
               "please use `RuntimeOption.trt_option.SetShape()` instead."
            << std::endl;
  trt_option.SetShape(input_name, min_shape, opt_shape, max_shape);
}

void RuntimeOption::SetTrtInputData(const std::string& input_name,
                                    const std::vector<float>& min_shape_data,
                                    const std::vector<float>& opt_shape_data,
                                    const std::vector<float>& max_shape_data) {
  FDWARNING << "`RuntimeOption::SetTrtInputData` will be removed in v1.2.0, "
               "please use `RuntimeOption.trt_option.SetInputData()` instead."
            << std::endl;
  trt_option.SetInputData(input_name, min_shape_data, opt_shape_data,
                          max_shape_data);
}

void RuntimeOption::SetTrtMaxWorkspaceSize(size_t max_workspace_size) {
  FDWARNING << "`RuntimeOption::SetTrtMaxWorkspaceSize` will be removed in "
               "v1.2.0, please modify its member variable directly, e.g "
               "`RuntimeOption.trt_option.max_workspace_size = "
            << max_workspace_size << "`." << std::endl;
  trt_option.max_workspace_size = max_workspace_size;
}
void RuntimeOption::SetTrtMaxBatchSize(size_t max_batch_size) {
  FDWARNING << "`RuntimeOption::SetTrtMaxBatchSize` will be removed in v1.2.0, "
               "please modify its member variable directly, e.g "
               "`RuntimeOption.trt_option.max_batch_size = "
            << max_batch_size << "`." << std::endl;
  trt_option.max_batch_size = max_batch_size;
}

void RuntimeOption::EnableTrtFP16() {
  FDWARNING << "`RuntimeOption::EnableTrtFP16` will be removed in v1.2.0, "
               "please modify its member variable directly, e.g "
               "`runtime_option.trt_option.enable_fp16 = true;`"
            << std::endl;
  trt_option.enable_fp16 = true;
}

void RuntimeOption::DisableTrtFP16() {
  FDWARNING << "`RuntimeOption::DisableTrtFP16` will be removed in v1.2.0, "
               "please modify its member variable directly, e.g "
               "`runtime_option.trt_option.enable_fp16 = false;`"
            << std::endl;
  trt_option.enable_fp16 = false;
}

void RuntimeOption::EnablePinnedMemory() { enable_pinned_memory = true; }

void RuntimeOption::DisablePinnedMemory() { enable_pinned_memory = false; }

void RuntimeOption::SetTrtCacheFile(const std::string& cache_file_path) {
  FDWARNING << "`RuntimeOption::SetTrtCacheFile` will be removed in v1.2.0, "
               "please modify its member variable directly, e.g "
               "`runtime_option.trt_option.serialize_file = \""
            << cache_file_path << "\"." << std::endl;
  trt_option.serialize_file = cache_file_path;
}

void RuntimeOption::SetOpenVINOStreams(int num_streams) {
  FDWARNING << "`RuntimeOption::SetOpenVINOStreams` will be removed in v1.2.0, "
               "please modify its member variable directly, e.g "
               "`runtime_option.openvino_option.num_streams = "
            << num_streams << "`." << std::endl;
  openvino_option.num_streams = num_streams;
}

}  // namespace fastdeploy
