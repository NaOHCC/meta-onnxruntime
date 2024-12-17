#include <assert.h>

#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "onnxruntime/core/session/experimental_onnxruntime_cxx_api.h"
// #include "onnxruntime_cxx_api.h"
using namespace Ort::Experimental;

int ArgMaxRow(const std::vector<float> &row) {
  return std::distance(row.begin(), std::max_element(row.begin(), row.end()));
}

// ArgMax for a 2D Tensor
std::vector<int> ArgMax2D(Ort::Value tensor) {
  auto typeAndShape = tensor.GetTensorTypeAndShapeInfo();

  auto shape = typeAndShape.GetShape();

  assert(shape.size() == 2 && "ArgMax2D");

  auto rows = shape[0];
  auto cols = shape[1];
  auto tensor_data = tensor.GetTensorData<float>();
  std::vector<int> result(rows);

  for (int i = 0; i < rows; ++i) {
    auto row_begin = tensor_data + i * cols;
    auto row_end = row_begin + cols;
    result[i] = ArgMaxRow(std::vector<float>(row_begin, row_end));
  }

  return result;
}

template <class T> auto vectorToString(const std::vector<T> &v) -> std::string {
  std::ostringstream oss;
  oss << "(";
  for (size_t i = 0; i < v.size(); ++i) {
    oss << v[i];
    if (i != v.size() - 1) {
      oss << ", ";
    }
  }
  oss << ")";
  return oss.str();
};

int main(int argc, char *argv[]) {
  // 记录程序运行时间
  auto start = std::chrono::high_resolution_clock::now();
  // 初始化环境，每个进程一个环境
  // 环境保留了线程池和其他状态信息
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
  // 初始化Session选项
  Ort::SessionOptions session_options;
  // 创建Session并把模型加载到内存中

  std::string model_path;
  if (argc < 2) {
    std::cout << "model path is not provided, use default path: "
                 "/data/tmp/model.fp32.onnx"
              << std::endl;
    model_path = "/data/tmp/model.fp32.onnx";
  } else {
    std::cout << "model path: " << argv[1] << std::endl;
    model_path = argv[1];
  }
  // std::string model_path =
  // "/workspaces/test/pytorch/model/model.torch.qat.onnx";

  Ort::Experimental::Session session(env, model_path, session_options);

  // 打印模型的输入层(node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // 输出模型输入节点的数量
  size_t num_output_nodes = session.GetOutputCount();
  std::vector<std::string> input_node_names = session.GetInputNames();
  std::vector<std::string> output_node_names = session.GetOutputNames();
  std::vector<int64_t> input_node_dims;
  auto inputShapes = session.GetInputShapes();
  auto outputShapes = session.GetOutputShapes();

  // 迭代所有的输入节点

  for (int i = 0, num = session.GetInputCount(); i < num; i++) {
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::cout << i << " name: " << input_node_names[i]
              << " shape:" << vectorToString(inputShapes[i])
              << " type: " << type << "\n";
  }

  // 打印输出节点信息，方法类似
  for (size_t i = 0; i < num_output_nodes; i++) {
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::cout << i << " name: " << output_node_names[i]
              << " shape:" << vectorToString(outputShapes[i])
              << " type: " << type << "\n";
  }

  // // 使用样本数据对模型进行评分，并检验出入值的合法性
  size_t input_tensor_size = 4 * 3 * 224 * 224;

  std::vector<float> input_tensor_values(input_tensor_size);

  // // 初始化一个数据（演示用,这里实际应该传入归一化的数据）
  // for (unsigned int i = 0; i < input_tensor_size; i++) input_tensor_values[i]
  // = (float)i / (input_tensor_size + 1);

  // 为输入数据创建一个Tensor对象
  try {
    std::vector<int64_t> input_shape = {4, 3, 224, 224};

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
    // input_tensor_values.data(),
    //                                                           input_tensor_size,
    //                                                           input_shape.data(),
    //                                                           4);
    // assert(input_tensor.IsTensor());
    std::vector<Ort::Value> inputValue;
    for (auto &s : inputShapes) {
      // auto v = Value::CreateTensor<float>(s);
      int product = std::accumulate(s.begin(), s.end(), 1, std::multiplies<>());

      auto p = new float[product];
      std::fill(p, p + product, 100.0f);

      inputValue.push_back(Value::CreateTensor(p, product, s));
    }

    std::vector<Ort::Value> outputValue;
    for (auto &s : outputShapes) {
      // auto v = Value::CreateTensor<float>(s);
      outputValue.push_back(Value::CreateTensor<float>(s));
    }

    // // 推理得到结果
    session.Run(input_node_names, inputValue, output_node_names, outputValue);

    auto max = ArgMax2D(std::move(outputValue[0]));

    std::cout << "result: " << vectorToString(max) << std::endl;
    // auto out = &outputValue[0];

    // // 另一种形式
    // // Ort::IoBinding io_binding{session};
    // // io_binding.BindInput("img", input_tensor);
    // // Ort::MemoryInfo output_mem_info =
    // Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // // io_binding.BindOutput("mask", output_mem_info);
    // // session.Run(Ort::RunOptions{nullptr}, io_binding);

    // printf("Number of outputs = %d\n", outputValue.size());

  } catch (Ort::Exception &e) {
    std::cout << e.what() << std::endl;
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
  printf("Done!\n");
  return 0;
}