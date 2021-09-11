#include "Yolov5.hpp"
#include <chrono>
namespace Yolov5{

bool Yolov5::init(){
    auto mLoggingName = Logging::gLogger.defineTest(this->mParams.gLoggerName, 0, NULL);
    Logging::gLogger.reportTestStart(mLoggingName);
    if (!build())
    {
        Logging::gLogger.reportFail(mLoggingName);
    }
    Logging::gLogInfo << "Building and running a GPU inference engine for Yolov5" << std::endl;
    // parse the dimensions of inputs and outputs. 
    const int nb_bindings = this->mEngine->getNbBindings();
    for(int i = 0; i < nb_bindings; ++i) {
        std::string name = this->mEngine->getBindingName(i);
        auto bindingDimentions = this->mEngine->getBindingDimensions(i);
        if (this->mEngine->bindingIsInput(i)){
            std::cout << "Intput BindingName " << i << ": " << name << std::endl;
            this->mParams.inputTensorNames.push_back(name);
        }
        else{
            std::cout << "output BindingName " << i << ": " << name << std::endl;
            this->mParams.outputTensorNames.push_back(name);
        }
        std::cout << "Dimentions: ";
        for(int j = 0; j < bindingDimentions.nbDims; j++)
        {
            std::cout << bindingDimentions.d[j] << ", ";
        }
        std::cout << std::endl;
    }
    return true;
}

//!
//! \brief Uses an onnx parser to create the YOLO Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the YOLO network
//!
//! \param builder Pointer to the engine builder
//!
bool Yolov5::constructNetwork(TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
    TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network, TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
    TensorRTUniquePtr<nvonnxparser::IParser>& parser)
{
    // Parse ONNX model file to populate TensorRT INetwork
    int verbosity = (int) nvinfer1::ILogger::Severity::kERROR;

    Logging::gLogInfo << "Parsing ONNX file: " << mParams.onnxFileName << std::endl;

    if (!parser->parseFromFile(mParams.onnxFileName.c_str(), verbosity))
    {
        Logging::gLogError << "Unable to parse ONNX model file: " << mParams.onnxFileName << std::endl;
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);

    // config->setMaxWorkspaceSize(2048_MiB);
    config->setMaxWorkspaceSize(mParams.workspace*1024_MiB);

    builder->allowGPUFallback(true);

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    nvinfer1::Dims mindims, optdims, maxdims;
    mindims.nbDims = this->mParams.min_shape.size();
    optdims.nbDims = this->mParams.opt_shape.size();
    maxdims.nbDims = this->mParams.max_shape.size();
    for(int i = 0; i < this->mParams.min_shape.size(); i++){
        mindims.d[i] = this->mParams.min_shape[i];
        optdims.d[i] = this->mParams.opt_shape[i];
        maxdims.d[i] = this->mParams.max_shape[i];
    }
    std::string inputname = "input";
    profile->setDimensions(inputname.c_str(), OptProfileSelector::kMIN, Dims4(mindims.d[0], mindims.d[1], mindims.d[2], mindims.d[3]));
    profile->setDimensions(inputname.c_str(), OptProfileSelector::kOPT, Dims4(optdims.d[0], optdims.d[1], optdims.d[2], optdims.d[3]));
    profile->setDimensions(inputname.c_str(), OptProfileSelector::kMAX, Dims4(maxdims.d[0], maxdims.d[1], maxdims.d[2], maxdims.d[3]));
    config->addOptimizationProfile(profile);

    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    
    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

    // issue for int8 mode
    if (mParams.int8)
    {
        BatchStream calibrationStream(
            mParams.explicitBatchSize, mParams.nbCalBatches, mParams.calibrationBatches, mParams.dataDirs);
        calibrator.reset(
            new Int8EntropyCalibrator2<BatchStream>(calibrationStream, 0, "Yolo", mParams.inputTensorNames[0].c_str()));
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }

    // add BatchedNMS_TRT Plugin
    auto pluginCreator = getPluginRegistry()->getPluginCreator("BatchedNMSDynamic_TRT", "1");
    auto tensor1 = (*network).getOutput(0);
    auto tensor2 = (*network).getOutput(1);
    ITensor *tensors[] = {tensor1, tensor2};
    std::vector<nvinfer1::PluginField> f;
    bool shareLocation = true;
    int backgroundLabelId = -1;
    int numClasses = mParams.outputClsSize;
    int topK = mParams.topK;
    int clipBoxes = 0;
    float iouThreshold = 0.25;
    float scoreThreshold = 0.45;
    int keepTopK = mParams.keepTopK;
    std::cout << "iouThreshold: " << iouThreshold << ", scoreThreshold: " << scoreThreshold << ", keepTopK: " << keepTopK <<std::endl;
    bool isNormalized = false;
    f.emplace_back("shareLocation", &shareLocation, nvinfer1::PluginFieldType::kUNKNOWN, 1);
    f.emplace_back("backgroundLabelId", &backgroundLabelId, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("numClasses", &numClasses, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("keepTopK", &keepTopK, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("topK", &topK, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("iouThreshold", &iouThreshold, nvinfer1::PluginFieldType::kFLOAT32, 1);
    f.emplace_back("scoreThreshold", &scoreThreshold, nvinfer1::PluginFieldType::kFLOAT32, 1);
    f.emplace_back("isNormalized", &isNormalized, nvinfer1::PluginFieldType::kUNKNOWN, 1);
    f.emplace_back("clipBoxes", &clipBoxes, nvinfer1::PluginFieldType::kINT32, 1);
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = f.size();
    fc.fields = f.data();
    auto plugin = pluginCreator->createPlugin("BatchedNMS_TRT", &fc);
    auto nms = network->addPluginV2(tensors, 2, *plugin);
    nms->getOutput(0)->setName("num_detections");
    network->markOutput(*nms->getOutput(0));

    nms->getOutput(1)->setName("nmsed_boxes");
    network->markOutput(*nms->getOutput(1));

    nms->getOutput(2)->setName("nmsed_scores");
    network->markOutput(*nms->getOutput(2));

    nms->getOutput(3)->setName("nmsed_classes");
    network->markOutput(*nms->getOutput(3));
    network->unmarkOutput(*tensor1);
    network->unmarkOutput(*tensor2);

    // Enable DLA if mParams.dlaCore is true
    TensorRTCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    Logging::gLogInfo << "Building TensorRT engine" << mParams.engingFileName << std::endl;

    this->mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), TensorRTCommon::InferDeleter());
    
    if (!this->mEngine)
    {
        return false;
    }

    if (mParams.engingFileName.size() > 0)
    {
        std::ofstream p(mParams.engingFileName, std::ios::binary);
        if (!p)
        {
            return false;
        }
        nvinfer1::IHostMemory* ptr = this->mEngine->serialize();
        assert(ptr);
        p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
        ptr->destroy();
        p.close();
        Logging::gLogInfo << "TRT Engine file saved to: " << mParams.engingFileName << std::endl;
    }

    return true;
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool Yolov5::preProcess(const TensorRTCommon::BufferManager& mbuffers)
{
    const int inputB = this->mInputDims.d[0];
    const int inputC = this->mInputDims.d[1];
    const int inputH = this->mInputDims.d[2];
    const int inputW = this->mInputDims.d[3];

    float* hostInputBuffer = static_cast<float*>(mbuffers.getHostBuffer(this->mParams.inputTensorNames[0]));
    std::vector<std::vector<cv::Mat>> input_channels;
    int d_batch_pos = 0;
    for (int b = 0; b < inputB; ++b)
    {
        input_channels.push_back(std::vector<cv::Mat> {static_cast<size_t>(inputC)});

        // cv::Mat img = *this->inferImage;
        cv::Mat img = this->inferImages[b];//.data();
        auto IMAGE_WIDTH = inputW;
        auto IMAGE_HEIGHT = inputH;
        int w, h, tx1, tx2, ty1, ty2;
        float r_w = IMAGE_WIDTH / (img.cols * 1.0);
        float r_h = IMAGE_HEIGHT / (img.rows * 1.0);
        if (r_h > r_w)
        {
            w = IMAGE_WIDTH;
            h = r_w * img.rows;
            tx1 = 0;
            tx2 = 0;
            ty1 = (IMAGE_HEIGHT - h) / 2;
            ty2 = IMAGE_HEIGHT - h - ty1;
        }
        else
        {
            w = r_h * img.cols;
            h = IMAGE_HEIGHT;
            tx1 = (IMAGE_WIDTH - w) / 2;
            tx2 = IMAGE_WIDTH - w - tx1;
            ty1 = 0;
            ty2 = 0;
        }
        cv::Mat re;
        auto scaleSize = cv::Size(w, h);
        cv::Mat pad_dst;
        cv::Scalar value(114, 114, 114);
        cv::resize(img, re, scaleSize, 0, 0, cv::INTER_LINEAR);
        copyMakeBorder(re, pad_dst, ty1, ty2, tx1, tx2, cv::BORDER_CONSTANT, value);
        // for (int bi = 0; bi < inputB; ++bi)
        // {
        cv::split(pad_dst, input_channels[b]);
        // }

        int volBatch = inputC * inputH * inputW;
        int volChannel = inputH * inputW;
        int volW = inputW;

    // int d_batch_pos = 0;
    // for (int b = 0; b < inputB; b++)
    // {  
        int d_c_pos = d_batch_pos;
        for (int c = 0; c < inputC; c++)
        {
            int s_h_pos = 0;
            int d_h_pos = d_c_pos;
            for (int h = 0; h < inputH; h++)
            {
                int s_pos = s_h_pos;
                int d_pos = d_h_pos;
                for (int w = 0; w < inputW; w++)
                {
                    hostInputBuffer[d_pos] = (float)input_channels[b][c].data[s_pos] / 255.0f;
                    ++s_pos;
                    ++d_pos;
                }
                s_h_pos += volW;
                d_h_pos += volW;
            }
            d_c_pos += volChannel;
        }
        d_batch_pos += volBatch;
    // }
    }
    return true;
}

//!
//! \brief Filters output detections and verify result
//!
//! \return whether the detection output matches expectations
//!
bool Yolov5::postProcess(const TensorRTCommon::BufferManager& mbuffers)
{
    const int keepTopK = mParams.keepTopK;

    int32_t *num_detections = static_cast<int32_t*>(mbuffers.getHostBuffer(this->mParams.outputTensorNames[0]));
    float *nmsed_boxes = static_cast<float*>(mbuffers.getHostBuffer(this->mParams.outputTensorNames[1]));
    float *nmsed_scores = static_cast<float*>(mbuffers.getHostBuffer(this->mParams.outputTensorNames[2]));
    float *nmsed_classes = static_cast<float*>(mbuffers.getHostBuffer(this->mParams.outputTensorNames[3]));

    if (!num_detections || !nmsed_boxes || !nmsed_scores || !nmsed_classes)
    {
        std::cout << "NULL value output detected!" << std::endl;
    }

    auto nms_bboxes = this->get_bboxes(this->mParams.outputShapes[0][0], keepTopK, num_detections, nmsed_boxes, nmsed_scores, nmsed_classes);

    std::cout << "batch size: " << nms_bboxes.size() << std::endl;

    for (int b = 0; b < this->mParams.explicitBatchSize; ++b)
    {
        auto &bboxes_image = nms_bboxes[b];
        std::cout << "------------ Next Image! --------------" << std::endl;
        std::cout << "Number of detections: " << num_detections[b] << std::endl;
        cv::Mat bgr_img_cpy = this->inferImages[b];
        int H = bgr_img_cpy.rows;
        int W = bgr_img_cpy.cols;
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

        std::vector<trtinfer::DagcppBox> detected;
        for(auto &bbox : bboxes_image)
        {
            trtinfer::DagcppBox result_box;
            result_box.x = (bbox.x1 - pad_x) / ratio_total;// * W / inputW;
            result_box.y = (bbox.y1 - pad_y) / ratio_total;//* H / inputH;
            result_box.w = (bbox.x2 - bbox.x1 - pad_x) / ratio_total; // * W / inputW;
            result_box.h = (bbox.y2 - bbox.y1 - pad_y) / ratio_total; //* H / inputH;
            result_box.cl = bbox.cls;
            result_box.prob = bbox.score;
            detected.push_back(result_box);
            std::cout << "[ " << bbox.x1 << " " << bbox.y1 << " " << bbox.x2 << " " << bbox.y2 << " ] score: " << bbox.score << " class: " << bbox.cls << std::endl;
        }
        this->batchDetected.push_back(detected);
        // Draw bboxes only for the first image in each batch
        this->draw_bboxes(nms_bboxes[b], bgr_img_cpy);
    }

    return true;
}

} // namespace Yolov5