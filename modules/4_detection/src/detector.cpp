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
    std::vector<Rect> boxesTmp = boxes;
    std::vector<float> probTmp = probabilities;
    std::vector<Rect> acceptBoxes;

    while (boxesTmp.size() > 0)
    {
        float tempprob = *std::max_element(probTmp.begin(), probTmp.end());
        auto iterMax = std::find(probTmp.begin(), probTmp.end(), tempprob);
        int iterMaxint = iterMax - probTmp.begin();

        Rect currentRect = boxesTmp[iterMaxint];

        boxesTmp.erase(boxesTmp.begin() + iterMaxint);
        probTmp.erase(probTmp.begin() + iterMaxint);

        acceptBoxes.push_back(currentRect);

        for (int i = 0; i < boxesTmp.size(); i++)
        {

            if (iou(currentRect, boxesTmp[i]) > threshold)
            {
                boxesTmp.erase(boxesTmp.begin() + i);
                probTmp.erase(probTmp.begin() + i);
            }
        }
    }

    for (int i = 0; i < acceptBoxes.size(); i++)
    {
        Rect tempRect = acceptBoxes[i];
        auto iterMax = std::find(boxes.begin(), boxes.end(), tempRect);
        int iterMaxint = iterMax - boxes.begin();
        indices.push_back(iterMaxint);
    }
}

float iou(const cv::Rect& a, const cv::Rect& b) {
    return (float)(a & b).area() / (float)(a.area() + b.area() - (a & b).area());
}
