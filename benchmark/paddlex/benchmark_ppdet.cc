// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "flags.h"
#include "macros.h"
#include "option.h"

namespace vision = fastdeploy::vision;
namespace benchmark = fastdeploy::benchmark;

DEFINE_bool(no_nms, false, "Whether the model contains nms.");
DEFINE_string(trt_shape, "1,3,640,640:1,3,640,640:1,3,640,640",
              "Set min/opt/max shape for trt/paddle_trt backend."
              "eg:--trt_shape 1,3,640,640:1,3,640,640:1,3,640,640");
DEFINE_string(input_name, "image",
              "Set input name for trt/paddle_trt backend."
              "eg:--input_names x");

int main(int argc, char* argv[]) {
#if defined(ENABLE_BENCHMARK) && defined(ENABLE_VISION)
  // Initialization
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option, argc, argv, true)) {
    return -1;
  }
  auto im = cv::imread(FLAGS_image);
  std::unordered_map<std::string, std::string> config_info;
  benchmark::ResultManager::LoadBenchmarkConfig(FLAGS_config_path,
                                                &config_info);
  std::string model_name, params_name, config_name;
  auto model_format = fastdeploy::ModelFormat::ONNX;
  if (!UpdateModelResourceName(&model_name, &params_name, &config_name,
                               &model_format, config_info)) {
    return -1;
  }
  auto model_file = FLAGS_model + sep + model_name;
  auto params_file = FLAGS_model + sep + params_name;
  auto config_file = FLAGS_model + sep + config_name;
  if (config_info["backend"] == "paddle_trt") {
    option// Note: paddle_infer_option removed, using TRT backend
  }
  if (config_info["backend"] == "paddle_trt" ||
      config_info["backend"] == "trt") {
    std::vector<std::vector<int32_t>> trt_shapes =
        benchmark::ResultManager::GetInputShapes(FLAGS_trt_shape);
    option.trt_option.SetShape(FLAGS_input_name, trt_shapes[0], trt_shapes[1],
                               trt_shapes[2]);
    option.trt_option.SetShape("scale_factor", {1, 2}, {1, 2}, {1, 2});
    option.trt_option.SetShape("im_shape", {1, 2}, {1, 2}, {1, 2});
  }
  auto model_ppdet = vision::detection::PaddleDetectionModel(
      model_file, params_file, config_file, option, model_format);
  vision::DetectionResult res;
  // Run profiling
  if (FLAGS_no_nms) {
    model_ppdet.GetPostprocessor().ApplyNMS();
  }
  BENCHMARK_MODEL(model_ppdet, model_ppdet.Predict(im, &res))
  // std::cout << res.Str() << std::endl;
  auto vis_im = vision::VisDetection(im, res, 0.3);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
#endif

  return 0;
}
