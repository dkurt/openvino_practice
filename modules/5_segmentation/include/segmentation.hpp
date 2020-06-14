#pragma once
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

// Compute Dice score over to binary masks
float Dice(const cv::Mat& a, const cv::Mat& b);

// Basic interface
class Segmenter {
public:
    // Performs semantic segmentation
    // [inp] image - color BGR image
    // [out] mask  - output binary mask of the same size as input image but with
    //               a single channel. Every pixel is a number which indicates
    //               predicted class.
    virtual void segment(const cv::Mat& image, cv::Mat& mask) = 0;
};

// Semantic segmentation for  Autonomous Driving Assistant System scenario.
// origin: https://github.com/opencv/open_model_zoo/blob/master/models/intel/semantic-segmentation-adas-0001/
class ADAS : public Segmenter {
public:
    ADAS();

    virtual void segment(const cv::Mat& image, cv::Mat& mask);
};

// Glands segmentation in colon histology images
// origin: https://github.com/NifTK/NiftyNetModelZoo/tree/5-reorganising-with-lfs/unet_histology
class UNetHistology : public Segmenter {
public:
    UNetHistology();

    virtual void segment(const cv::Mat& image, cv::Mat& mask);

    // Convert image from BGR to RGB color space.
    // [inp] src - color BGR image
    // [out] dst - color RGB image
    static void bgr2rgb(const cv::Mat& src, cv::Mat& dst);

    // Add extra pixels from an every side of image.
    static void padMinimum(const cv::Mat& src, int width, int height, cv::Mat& dst);

    // Perform mean-variance normalization.
    // [inp] src - input image
    // [out] dst - output image of type CV_32F with mean = 0 and std = 1.
    static void normalize(const cv::Mat& src, cv::Mat& dst);

    // Estimate number of glands by segmentation mask.
    // [inp] segm - binary input segmentation mask where non-zero pixels
    //              correspond to glands
    static int countGlands(const cv::Mat& segm);

private:
    InferenceEngine::InferRequest req;
};
