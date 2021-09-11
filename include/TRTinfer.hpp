/*
TensorRT inference Class
 */

#pragma once
// yaml config
#include "yaml-cpp/yaml.h"
#include "EntropyCalibrator.h"
#include "argsParser.h"
#include "buffers.h"
#include "logger.h"

#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace trtinfer{


//!
//! \brief The Params structure groups the additional parameters required by
//!         the SSD sample.
//!
struct TRTParams
{
    int outputClsSize = 1;              //!< The number of output classes
    int topK = 512;
    int keepTopK = 100;                   //!< The maximum number of detection post-NMS
    int nbCalBatches = 100;               //!< The number of batches for calibration
    std::vector<std::string> dataDirs;
    int dlaCore = -1;
    // nums of keypoints, Keypoint Detection only
    int keypoints;
    int int8 = 0;
    int fp16 = 0;
    int explicitBatchSize = 1;
    int batchSize = 1;
    int n_batch = 1;
    int workspace = 2048;
    std::vector<int> inputShape;
    std::vector<std::vector<int>> outputShapes;
    std::string gLoggerName;
    std::string onnxFileName;
    std::string inputImageName;
    std::string outputImageName;
    std::string calibrationBatches; //!< The path to calibration batches
    std::string engingFileName;
    std::string cocoClassNamesFileName;
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
    // dynamic转换生成trt时用到
    int ifdynamic = 1;
    std::vector<int> min_shape;
    std::vector<int> opt_shape;
    std::vector<int> max_shape;
};

struct BoundingBox
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int cls;
};

struct DagcppBox
{
    int cl;
    float x, y, w, h;
    float prob;
};

struct SpeedInfo
{
    long long preProcess;
    long long model;
    long long postProcess;

    SpeedInfo() :
        preProcess {0},
        model {0},
        postProcess {0}
    {}

    void printTimeConsmued()
    {
        std::cout << "Time consumed in preProcess: " << this->preProcess << std::endl;
        std::cout << "Time consumed in model: " << this->model << std::endl;
        std::cout << "Time consumed in postProcess: " << this->postProcess << std::endl;
    }
};

//! \brief  The TRTInfer class implements the SSD sample
//!
//! \details It creates the network using a caffe model
//!
template <typename T>
using TensorRTUniquePtr = std::unique_ptr<T, TensorRTCommon::InferDeleter>;
class TRTInfer
{
    // template <typename initmParams>
    // using initParams = initmParams;

public:
    template <typename initParams>
    TRTInfer(initParams params)
    {
        char str[100];
        mParams = params;
        std::ifstream coco_names(this->mParams.cocoClassNamesFileName);  
        while(coco_names.getline(str, 100) )
        {
            std::string cls_name {str};
            this->mClasses.push_back(cls_name.substr(0, cls_name.size()));
        }
        coco_names.close();
    }
    ~TRTInfer(){}
    // TRTInfer();
    //!
    //! \brief Function builds the network engine
    //!
    // bool build();
    //!
    //! \brief Creates the network, configures the builder and creates the network engine
    //!
    //! \details This function creates the YOLO network by parsing the ONNX model and builds
    //!          the engine that will be used to run YOLO (this->mEngine)
    //!
    //! \return Returns true if the engine was created successfully and false otherwise
    //!
    bool build()
    {
        initLibNvInferPlugins(&Logging::gLogger.getTRTLogger(), "");
        initLibNvInferPlugins(&Logging::gLogger.getTRTLogger(), "ONNXTRT_NAMESPACE");

        if (this->fileExists(mParams.engingFileName))
        {
            std::vector<char> trtModelStream;
            size_t size{0};
            std::ifstream file(mParams.engingFileName, std::ios::binary);
            if (file.good())
            {
                file.seekg(0, file.end);
                size = file.tellg();
                file.seekg(0, file.beg);
                trtModelStream.resize(size);
                file.read(trtModelStream.data(), size);
                file.close();
            }

            IRuntime* infer = nvinfer1::createInferRuntime(Logging::gLogger);
            if (mParams.dlaCore >= 0)
            {
                infer->setDLACore(mParams.dlaCore);
            }
            this->mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
                infer->deserializeCudaEngine(trtModelStream.data(), size, nullptr), TensorRTCommon::InferDeleter());

            infer->destroy();

            Logging::gLogInfo << "TRT Engine loaded from: " << mParams.engingFileName << std::endl;
            if (!this->mEngine)
            {
                return false;
            }
            else
            {
                this->mInputDims.nbDims = this->mParams.inputShape.size();
                this->mInputDims.d[0] = this->mParams.inputShape[0];
                this->mInputDims.d[1] = this->mParams.inputShape[1];
                this->mInputDims.d[2] = this->mParams.inputShape[2];
                this->mInputDims.d[3] = this->mParams.inputShape[3];

                return true;
            }
        }
        else
        {
            auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(Logging::gLogger.getTRTLogger()));
            if (!builder)
            {
                return false;
            }

            const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
            auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
            if (!network)
            {
                return false;
            }

            auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
            if (!config)
            {
                return false;
            }

            auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, Logging::gLogger.getTRTLogger()));
            if (!parser)
            {
                return false;
            }

            auto constructed = constructNetwork(builder, network, config, parser);
            if (!constructed)
            {
                return false;
            }

            assert(network->getNbInputs() == 1);
            this->mInputDims = network->getInput(0)->getDimensions();
            std::cout << this->mInputDims.nbDims << std::endl;
            assert(this->mInputDims.nbDims == 4);
        }

        return true;
    }

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    // bool infer(cv::Mat &inputImage);
    bool infer(std::vector<cv::Mat> &inputImages)
    {
        // this->inferImage = inputImage;
        this->resetTime();
        this->inferImages = inputImages;
        nvinfer1::IExecutionContext* mcontext = this->mEngine->createExecutionContext();
        if (!mcontext)
        {
            Logging::gLogInfo << "Context init failed" << std::endl;
        }
        // dynamic 输入设置
        auto bindingInputDimentions = this->mEngine->getBindingDimensions(0);
        if (bindingInputDimentions.d[0] == -1 || bindingInputDimentions.d[1] == -1 || 
            bindingInputDimentions.d[2] == -1 || bindingInputDimentions.d[3] == -1)
        {
            /* TensorRT 7中，没有setOptimizationProfileAsync这个函数，但在TensorRT最新的官方API文档中，推荐使用新版函数，
            但目前无法使用，暂时使用旧版函数，不知道会不会造成可能的GPU内存相关的问题
            */
            /*官方说明：
            setOptimizationProfileAsync() function replaces the now deprecated version of the API setOptimizationProfile(). 
            Using setOptimizationProfile() to switch between optimization profiles can cause GPU memory copy operations 
            in the subsequent enqueue() or enqueueV2() operations operation. 
            To avoid these calls during enqueue, use setOptimizationProfileAsync() API instead. 
            */
            // context->setOptimizationProfileAsync(0, stream);

            // 旧版函数（TensorRT 7）
            mcontext->setOptimizationProfile(0);
            mcontext->setBindingDimensions(0, this->mInputDims);
            if(!mcontext->allInputDimensionsSpecified()){
                return false;
            }
            nvinfer1::Dims out_dim = mcontext->getBindingDimensions(1);
            if (out_dim.nbDims == -1) {
            std::cout << "Invalid network output, this might be caused by inconsistent input shapes." << std::endl;
                return false;
            }
        }
        TensorRTCommon::BufferManager mbuffers(this->mEngine, mParams.batchSize, mcontext);
        // 配置dynamic输入维度
        this->infer_iteration(mcontext, mbuffers);
        this->mSpeedInfo.printTimeConsmued();
        return true;
    }
    std::vector<cv::Mat> inferImages;
    std::vector<std::vector<DagcppBox> > batchDetected;
    TRTParams mParams; //!< The parameters for the TRTInfer

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    SpeedInfo mSpeedInfo;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    std::vector<std::string> mClasses;

    std::vector<std::vector<BoundingBox>> get_bboxes(int batch_size, int keep_topk,
        int32_t *num_detections, float *nmsed_boxes, float *nmsed_scores, float *nmsed_classes)
    {
        int n_detect_pos = 0;
        int box_pos = 0;
        int score_pos = 0;
        int cls_pos = 0;

        std::vector<std::vector<BoundingBox>> bboxes {static_cast<size_t>(batch_size)};

        for (int b = 0; b < batch_size; ++b)
        {
            for (int t = 0; t < keep_topk; ++t)
            {
                if (static_cast<int>(nmsed_classes[cls_pos + t]) < 0)
                {
                    break;
                }

                int box_coord_pos = box_pos + 4 * t;
                float x1 = nmsed_boxes[box_coord_pos];
                float y1 = nmsed_boxes[box_coord_pos + 1];
                float x2 = nmsed_boxes[box_coord_pos + 2];
                float y2 = nmsed_boxes[box_coord_pos + 3];

                bboxes[b].push_back(BoundingBox {
                    std::min(x1, x2),
                    std::min(y1, y2),
                    std::max(x1, x2),
                    std::max(y1, y2),
                    nmsed_scores[score_pos + t],
                    static_cast<int>(nmsed_classes[cls_pos + t]) });
            }

            n_detect_pos += 1;
            box_pos += 4 * keep_topk;
            score_pos += keep_topk;
            cls_pos += keep_topk;
        }

        return bboxes;
    }

    // void draw_bboxes(const std::vector<BoundingBox> &bboxes, cv::Mat &img);
    void draw_bboxes(const std::vector<BoundingBox> &bboxes, cv::Mat &testImg)
    {
        // std::cout << "Writing detection to image ..." << std::endl;
        int H = testImg.rows;
        int W = testImg.cols;
        const int inputH = this->mInputDims.d[2];
        const int inputW = this->mInputDims.d[3];
        float ratio_w = inputW / (W * 1.0);
        float ratio_h = inputH / (H * 1.0);
        float ratio_total = 0.0f;
        float pad_x = 0, pad_y = 0;
        if (ratio_h > ratio_w)
        {
            ratio_total = ratio_w;
            pad_y = (inputH - H * ratio_total) / 2;
        }
        else
        {
            ratio_total = ratio_h;
            pad_x = (inputW - W * ratio_total) / 2;
        }
        std::cout << "Writing detection to image ..." << std::endl;

            for(size_t k = 0; k < bboxes.size(); k++)
            {
                if (bboxes[k].cls == -1)
                {
                    break;
                }

                int x1 = (bboxes[k].x1 - pad_x) / ratio_total;// * W/ inputW;
                int y1 = (bboxes[k].y1 - pad_y) / ratio_total;// * H / inputH;
                int x2 = (bboxes[k].x2 - pad_x) / ratio_total;// * W / inputW;
                int y2 = (bboxes[k].y2 - pad_y) / ratio_total;// * H / inputH;

                cv::rectangle(testImg, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 1);

                cv::putText(testImg, //target image
                    this->mClasses[bboxes[k].cls], //text
                    cv::Point(x1, y1), //top-left position
                    cv::FONT_HERSHEY_DUPLEX,
                    0.8,
                    CV_RGB(118, 185, 0), //font color
                    1);
            }

            cv::imwrite(this->mParams.outputImageName, testImg);
    }

    // long long now_in_milliseconds();
    long long now_in_milliseconds()
    {
        return std::chrono::duration_cast <std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();  
    }
    virtual bool init() = 0;
    //!
    //! \brief Parses an ONNX model for YOLO and creates a TensorRT network
    //!
    virtual bool constructNetwork(TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
        TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network, TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TensorRTUniquePtr<nvonnxparser::IParser>& parser) = 0;

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    virtual bool preProcess(const TensorRTCommon::BufferManager& mbuffers) = 0;

    // 使用cuda 做后处理，继承此类时，若不需要cuda后处理，无需重写该函数
    virtual bool postProcess_cuda(const TensorRTCommon::BufferManager& mbuffers){
        return true;
    }
    //!
    //! \brief Filters output detections and verify results
    //!
    virtual bool postProcess(const TensorRTCommon::BufferManager& mbuffers) = 0;
private:
    // 清空计算时间的结构体变量
    bool resetTime()
    {
        // long long preProcess;
        // long long model;
        // long long postProcess;
        this->mSpeedInfo.preProcess = 0;
        this->mSpeedInfo.model = 0;
        this->mSpeedInfo.postProcess = 0;
        return true;
    }
    //!
    //! \brief To check if certain file exists given the path
    //!
    bool fileExists(const std::string& name)
    {
        std::ifstream f(name.c_str());
        return f.good();
    }

    // bool infer_iteration(nvinfer1::IExecutionContext* context, TensorRTCommon::BufferManager& mbuffers);
    bool infer_iteration(nvinfer1::IExecutionContext* context, TensorRTCommon::BufferManager& mbuffers)
    {
        auto time1 = this->now_in_milliseconds();
        // Read the input data into the managed buffers
        assert(mParams.inputTensorNames.size() == 1);
        if (!preProcess(mbuffers))
        {
            return false;
        }

        auto time2 = this->now_in_milliseconds();

        // Memcpy from host input buffers to device input buffers
        mbuffers.copyInputToDevice();

        bool status = context->executeV2(mbuffers.getDeviceBindings().data());

        if (!status)
        {
            return false;
        }

        auto time3 = this->now_in_milliseconds();

        // 使用cuda做后处理
        if(!postProcess_cuda(mbuffers))
        {
            return false;
        }

        // Memcpy from device output buffers to host output buffers
        mbuffers.copyOutputToHost();

        // Post-process detections and verify results
        if (!postProcess(mbuffers))
        {
            return false;
        }

        auto time4 = this->now_in_milliseconds();

        this->mSpeedInfo.preProcess += time2 - time1;
        this->mSpeedInfo.model += time3 - time2;
        this->mSpeedInfo.postProcess += time4 - time3;

        return true;
    }

    
};

} // namespace trtinfer
