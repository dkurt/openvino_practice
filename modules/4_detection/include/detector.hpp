#pragma once
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

class Detector {
public:
    Detector();

    // Performs object detection
    // [inp] image
    // [inp] nmsThreshold  - threshold for nms
    // [inp] probThreshold - threshold for boxes probabilities
    // [out] boxes         - list of bounding boxes
    // [out] probabilities - list of probabilities corresponding to bounding boxes
    // [out] classes       - indices of classes
    void detect(const cv::Mat& image,
                float nmsThreshold,
                float probThreshold,
                std::vector<cv::Rect>& boxes,
                std::vector<float>& probabilities,
                std::vector<unsigned>& classes);
private:
    InferenceEngine::InferRequest req;
    std::string outputName;
};

// Compute Intersection over Union (IoU) metric between two rectangles.
// Returns a ratio of intersection area over union area.
float iou(const cv::Rect& a, const cv::Rect& b);

// Non-maximum suppression for detected bounding boxes.
// Method returns boxes with the highest probabilities among other boxes with
// intersection-over-union (IoU) more than specified threshold.
// [inp] boxes         - list of bounding boxes
// [inp] probabilities - list of probabilities corresponding to bounding boxes
// [inp] threshold     - IoU threshold. Overlapped boxes with IoU more than
//                       threshold but with lower probability should be suppressed.
// [out] indices       - output indices of bounding boxes from input list which are not suppressed
void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
         float threshold, std::vector<unsigned>& indices);
