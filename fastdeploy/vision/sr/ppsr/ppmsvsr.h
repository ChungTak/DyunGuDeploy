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
#pragma once
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/processors/transform.h"

namespace fastdeploy {
namespace vision {
namespace sr {

class FASTDEPLOY_DECL PPMSVSR : public FastDeployModel {
 public:
  /**
   * Set path of model file and configuration file, and the configuration of runtime
   * @param[in] model_file Path of model file, e.g PPMSVSR/model.pdmodel
   * @param[in] params_file Path of parameter file, e.g PPMSVSR/model.pdiparams
   * @param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
   * @param[in] model_format Model format of the loaded model, default is Paddle format
   */
  PPMSVSR(const std::string& model_file, const std::string& params_file,
          const RuntimeOption& custom_option = RuntimeOption(),
          const ModelFormat& model_format = ModelFormat::ONNX);
  /// model name contained PP-MSVSR。
  std::string ModelName() const override { return "PPMSVSR"; }
  /**
   * get super resolution frame sequence
   * @param[in] imgs origin frame sequences
   * @param[in] results super resolution frame sequence
   * @return true if the prediction successed, otherwise false
   */
  virtual bool Predict(std::vector<cv::Mat>& imgs,
                       std::vector<cv::Mat>& results);

 protected:
  PPMSVSR(){};

  virtual bool Initialize();

  virtual bool Preprocess(Mat* mat, std::vector<float>& output);

  virtual bool Postprocess(std::vector<FDTensor>& infer_results,
                           std::vector<cv::Mat>& results);

  std::vector<float> mean_;
  std::vector<float> scale_;
};
}  // namespace sr
}  // namespace vision
}  // namespace fastdeploy
