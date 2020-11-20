#include "detector.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <inference_engine.hpp>

using namespace cv;
using namespace InferenceEngine;

Detector::Detector() {
    Core ie;
    // Load deep learning network into memory
    auto net = ie.ReadNetwork(utils::fs::join(DATA_FOLDER, "face-detection-0104.xml"),
        utils::fs::join(DATA_FOLDER, "face-detection-0104.bin"));

    InputInfo::Ptr inputInfo = net.getInputsInfo()["image"];
    inputInfo->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
    inputInfo->setLayout(Layout::NHWC);
    inputInfo->setPrecision(Precision::U8);
    outputName = net.getOutputsInfo().begin()->first;

    // Initialize runnable object on CPU device
    ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");

    // Create a single processing thread
    req = execNet.CreateInferRequest();
}

Blob::Ptr wrapMatToBlob(const Mat& m) {
    CV_Assert(m.depth() == CV_8U);
    std::vector<size_t> dims = { 1, (size_t)m.channels(), (size_t)m.rows, (size_t)m.cols };
    return make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC),
        m.data);
}

void Detector::detect(const cv::Mat& image,
                      float nmsThreshold,
                      float probThreshold,
                      std::vector<cv::Rect>& boxes,
                      std::vector<float>& probabilities,
                      std::vector<unsigned>& classes) {
    Blob::Ptr input = wrapMatToBlob(image);

    req.SetBlob("image", input);

    // Launch network
    req.Infer();

    int x_min, y_min, x_max, y_max;

    std::vector<Rect> boxesCopy;
    std::vector<float> probabilitiesCopy;
    std::vector<unsigned> classesCopy;

    float* output = req.GetBlob(outputName)->buffer();
    int size = req.GetBlob(outputName)->size() / 7;
    
    for (int i = 0; i < size; ++i) {

        float probability = output[i * 7  + 2];
        
        if (probability > probThreshold) {
            
            int height = image.rows;
            int width = image.cols;

            x_min = output[i * 7 + 3] * width;
            y_min = output[i * 7 + 4] * height;
            x_max = output[i * 7 + 5] * width;
            y_max = output[i * 7 + 6] * height;

            Rect approvedRect(x_min, y_min, int(x_max - x_min + 1), int(y_max - y_min + 1));
            boxesCopy.push_back(approvedRect);

            unsigned classIndex = output[i * 7 + 1];
            classesCopy.push_back(classIndex);
            classes.push_back(classIndex);

            probabilitiesCopy.push_back(probability);
        }
    }

    std::vector<unsigned> indices;
    nms(boxesCopy, probabilitiesCopy, nmsThreshold, indices);

    for (auto indice : indices)
    {
        boxes.push_back(boxesCopy[indice]);
        classes.push_back(classesCopy[indice]);
        probabilities.push_back(probabilitiesCopy[indice]);
    }
}

void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
         float threshold, std::vector<unsigned>& indices) {

    std::map<float, int> probIdx;
    for (int i = 0; i < probabilities.size(); ++i){
        probIdx.insert(std::make_pair(probabilities[i], i));  
    }

    indices.resize(0);
    auto first_iter = probIdx.end();

    for (size_t i = 0; i < probIdx.size(); ++i){

        --first_iter;
        indices.push_back(first_iter->second);
        auto temp_iter = first_iter;                                

        for (size_t j = 0; j < probIdx.size() - i; ++j){

            --temp_iter;
            float iou_result = iou(boxes[first_iter->second], boxes[temp_iter->second]);

            if (iou_result > threshold) {
                probIdx.erase(temp_iter);
            }
        }
    }
}

float iou(const cv::Rect& a, const cv::Rect& b) {

    float inner_area = (a & b).area();
    float onner_area = a.area() + b.area();
    return inner_area / (onner_area - inner_area);
}