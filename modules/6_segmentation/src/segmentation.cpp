#include "segmentation.hpp"

#include <opencv2/core/utils/filesystem.hpp>

using namespace InferenceEngine;
using namespace cv;
using namespace cv::utils::fs;

float Dice(const Mat& a, const Mat& b) {
    CV_Error(Error::StsNotImplemented, "Dice score computation");
}

ADAS::ADAS() {
    Core ie;

    // Load deep learning network into memory
    auto net = ie.ReadNetwork(join(DATA_FOLDER, "semantic-segmentation-adas-0001.xml"),
                              join(DATA_FOLDER, "semantic-segmentation-adas-0001.bin"));
}

void ADAS::segment(const Mat& image, Mat& mask) {
    CV_Error(Error::StsNotImplemented, "ADAS semantic segmentation");
}
