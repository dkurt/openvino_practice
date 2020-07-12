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
}


void Detector::detect(const cv::Mat& image,
                      float nmsThreshold,
                      float probThreshold,
                      std::vector<cv::Rect>& boxes,
                      std::vector<float>& probabilities,
                      std::vector<unsigned>& classes) {
    CV_Error(Error::StsNotImplemented, "detect");
}


void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
         float threshold, std::vector<unsigned>& indices) {
    std::map<float, int> probIdx;
    for (int i = 0; i < probabilities.size(); ++i){
        if (probabilities[i] >= threshold) {
            probIdx.insert(std::make_pair(probabilities[i], i));
        }
    }
    indices.resize(probIdx.size());
    auto iter = probIdx.end();
    for (size_t i = 0; i < probIdx.size(); ++i){
        --iter;
        indices[i] = iter->second;
    }

    CV_Error(Error::StsNotImplemented, "nms");
}

float iou(const cv::Rect& a, const cv::Rect& b) {

    float inner_area = (a & b).area();
    float onner_area = a.area() + b.area();
    return inner_area / (onner_area - inner_area);
}
