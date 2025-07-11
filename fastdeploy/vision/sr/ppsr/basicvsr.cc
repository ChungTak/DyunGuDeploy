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

#include "fastdeploy/vision/sr/ppsr/basicvsr.h"

namespace fastdeploy {
namespace vision {
namespace sr {

BasicVSR::BasicVSR(const std::string& model_file,
                   const std::string& params_file,
                   const RuntimeOption& custom_option,
                   const ModelFormat& model_format) {
  // unsupported ORT backend
  valid_cpu_backends = {Backend::ORT, Backend::OPENVINO};
  valid_gpu_backends = {Backend::TRT, Backend::ORT};

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;

  initialized = Initialize();
}
}  // namespace sr
}  // namespace vision
}  // namespace fastdeploy