#pragma once
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

class Detector {
public:
    Detector();

    // Performs object detection
    void detect(const cv::Mat& image, std::vector<cv::Rect>& boxes,
                std::vector<float>& probabilities, std::vector<unsigned>& classes);
};

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
