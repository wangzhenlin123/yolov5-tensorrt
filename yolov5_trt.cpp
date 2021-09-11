#include "Yolov5.hpp"
#include "cmdline.h"

trtinfer::TRTParams specifyInputAndOutputNamesAndShapesYolov5(trtinfer::TRTParams &params, YAML::Node &config)
{
    int channels = config["args"]["channels"].as<int>();
    int inputH = config["args"]["height"].as<int>();
    int inputW = config["args"]["width"].as<int>();
    params.inputShape = std::vector<int> {params.explicitBatchSize, channels, inputH, inputW};

    // Output shapes when BatchedNMSPlugin is available
    params.outputShapes.push_back(std::vector<int>{params.explicitBatchSize, 1});
    params.outputShapes.push_back(std::vector<int>{params.explicitBatchSize, params.keepTopK, 4});
    params.outputShapes.push_back(std::vector<int>{params.explicitBatchSize, params.keepTopK});
    params.outputShapes.push_back(std::vector<int>{params.explicitBatchSize, params.keepTopK});

    return params;
}


trtinfer::TRTParams initializeYolov5Params(const std::string& tensor_path, const int n_classes, const int n_batches, const float conf_thresh, YAML::Node &config)
{
    trtinfer::TRTParams params;
    const std::string root_path = config["path"].as<std::string>();
    // This argument is for calibration of int8
    // Int8 calibration is not available until now
    // You have to prepare samples for int8 calibration by yourself 
    params.nbCalBatches = 100;

    params.engingFileName = tensor_path;

    // log name
    params.gLoggerName = "Yolov5.TensorRT";
    // The onnx file to load  input_args.get<int>("classes")
    params.onnxFileName = root_path + config["model"]["onnx"].as<std::string>();
    
    // Input tensor name of ONNX file & engine file
    // params.inputTensorNames.push_back("input");
    
    // Old batch configuration, it is zero if explicitBatch flag is true for the tensorrt engine
    // May be deprecated in the future
    params.batchSize = 0;
    
    // Number of classes (usually 80, but can be other values)
    params.outputClsSize = n_classes;
    
    // topK parameter of BatchedNMSPlugin
    params.topK = config["nms"]["topK"].as<int>();
    
    // keepTopK parameter of BatchedNMSPlugin
    params.keepTopK = config["nms"]["keepTopK"].as<int>();

    // Batch size, you can modify to other batch size values if needed
    params.explicitBatchSize = n_batches;

    params.n_batch = n_batches;

    // params.inputImageName = "../data/person.jpg";
    params.cocoClassNamesFileName = root_path + config["args"]["names"].as<std::string>();
    // params.cocoClassIDFileName = "../data/categories.txt";

    // Config number of DLA cores, -1 if there is no DLA core
    params.dlaCore = -1;

    params.int8 = config["build"]["int8"].as<int>();
    params.fp16 = config["build"]["fp16"].as<int>();
    params.workspace = config["build"]["workspace"].as<int>();
    params.outputImageName = root_path + config["image"]["output"].as<std::string>();

    // dynamic build trt
    params.min_shape = config["args"]["min"].as<std::vector<int>>();
    params.opt_shape = config["args"]["opt"].as<std::vector<int>>();
    params.max_shape = config["args"]["max"].as<std::vector<int>>();

    specifyInputAndOutputNamesAndShapesYolov5(params, config);

    return params;
}


int main(int argc, char** argv)
{
    cmdline::parser input_args;
    input_args.add<std::string>("config", 'c', "config file path", true, "../data/config.yaml");
    input_args.parse_check(argc, argv);
    YAML::Node config = YAML::LoadFile(input_args.get<std::string>("config"));
    const std::string root_path = config["path"].as<std::string>();
    const std::string tensor_path = root_path + config["model"]["tensorrt"].as<std::string>();
    const int n_classes = config["args"]["n_classes"].as<int>();
    const int n_batches = 1;
    const float conf_thresh = 0.45;
    const std::string image_path = root_path + config["image"]["input"].as<std::string>();
    cv::Mat InputImage = cv::imread(image_path, /*CV_LOAD_IMAGE_COLOR*/-1);
    std::vector<cv::Mat> InputImages{InputImage};

    // Yolov5::Yolov5 Yolov5Detector(initializeYolov5Params(tensor_path, n_classes, n_batches, conf_thresh, config));
    Yolov5::Yolov5* Yolov5Detector = new Yolov5::Yolov5(initializeYolov5Params(tensor_path, n_classes, n_batches, conf_thresh, config));
    Yolov5Detector->init();
    Yolov5Detector->infer(InputImages);
    // for (int i = 0; i < 10; i++){
    //     Yolov5Detector.infer(InputImages);
    // }

    return EXIT_SUCCESS;
}
