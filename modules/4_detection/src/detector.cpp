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
    InputInfo::Ptr input_info = net.getInputsInfo().begin()->second;
    std::string input_name = net.getInputsInfo().begin()->first;
    
    input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    input_info->setLayout(Layout::NHWC);
    input_info->setPrecision(Precision::U8);

    ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");


}

void Detector::detect(const cv::Mat& image,
                      float nmsThreshold,
                      float probThreshold,
                      std::vector<cv::Rect>& boxes,
                      std::vector<float>& probabilities,
                      std::vector<unsigned>& classes) {



    //CV_Error(Error::StsNotImplemented, "detect");
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
