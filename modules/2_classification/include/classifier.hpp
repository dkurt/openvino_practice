#pragma once
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

class Classifier {
public:
    Classifier();

    // Performs image classification
    // [inp] image         - OpenCV image in BGR color namespace
    // [out] probabilities - vector of probabilities for every class
    void classify(const cv::Mat& image, std::vector<float>& probabilities);

private:
    InferenceEngine::InferRequest req;
    std::string outputName;
};

// Returns top K elements in descending order with corresponding indices.
// [inp] src     - vector of values
// [inp] k       - positive integer to indicate number of top values
// [out] dst     - output top k highest values in descending order
// [out] indices - output indices of top 4 highest values from <src>
void topK(const std::vector<float>& src, unsigned k,
          std::vector<float>& dst,
          std::vector<unsigned>& indices);

// Apply SoftMax function to set of values [x0, x1, ..., xN] by formula
// yi = exp(xi) / ( exp(x1) + exp(x2) + ... + exp(xN) )
void softmax(std::vector<float>& values);
