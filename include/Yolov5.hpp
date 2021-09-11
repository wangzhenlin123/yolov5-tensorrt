#pragma once

#include "TRTinfer.hpp"
namespace Yolov5{

template <typename T>
using TensorRTUniquePtr = std::unique_ptr<T, TensorRTCommon::InferDeleter>;

class Yolov5 : public trtinfer::TRTInfer
{
public:
    // Yolov5Params mParams;
    Yolov5(const trtinfer::TRTParams& params):TRTInfer(params){};
    ~Yolov5(){};
    bool init();
    bool constructNetwork(TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
        TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network, TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TensorRTUniquePtr<nvonnxparser::IParser>& parser);
    bool preProcess(const TensorRTCommon::BufferManager& mbuffers);
    bool postProcess(const TensorRTCommon::BufferManager& mbuffers);
};

} // namespace Yolov5
