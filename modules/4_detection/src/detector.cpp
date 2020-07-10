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
    std::vector<Rect> boxesCopy = boxes;
    std::vector<float> probsCopy = probabilities;
    std::vector<Rect> approvedBoxes;

    while (boxesCopy.size() > 0)
    {
        float tempprob = *std::max_element(probsCopy.begin(), probsCopy.end());
        auto iteratorFinding = std::find(probsCopy.begin(), probsCopy.end(), tempprob);
        int indexFinding = iteratorFinding - probsCopy.begin();

        Rect currentRect = boxesCopy[indexFinding];

        boxesCopy.erase(boxesCopy.begin() + indexFinding);
        probsCopy.erase(probsCopy.begin() + indexFinding);

        approvedBoxes.push_back(currentRect);

        for (int i = 0; i < boxesCopy.size(); i++)
        {
            float iou_metric = iou(currentRect, boxesCopy[i]);
            if (iou_metric > threshold)
            {
                boxesCopy.erase(boxesCopy.begin() + i);
                probsCopy.erase(probsCopy.begin() + i);
            }
        }
    }

    for (int i = 0; i < approvedBoxes.size(); i++)
    {
        Rect tempRect = approvedBoxes[i];
        auto iteratorFinding = std::find(boxes.begin(), boxes.end(), tempRect);
        int indexFinding = std::distance(boxes.begin(), iteratorFinding);
        indices.push_back(indexFinding);
    }
}

float iou(const cv::Rect& a, const cv::Rect& b) {
    float intersectionRects = (a & b).area();
    float unionRects = a.area() + b.area();
    return intersectionRects / (unionRects - intersectionRects);
}
