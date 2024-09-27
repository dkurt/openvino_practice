#include "segmentation.hpp"

#include <opencv2/core/utils/filesystem.hpp>

using namespace InferenceEngine;
using namespace cv;
using namespace cv::utils::fs;

float Dice(const Mat& a, const Mat& b) {
    return 2.0f * countNonZero(a & b) / (countNonZero(a) + countNonZero(b));
}

ADAS::ADAS() {
    Core ie;

    // Load deep learning network into memory
    auto net = ie.ReadNetwork(join(DATA_FOLDER, "semantic-segmentation-adas-0001.xml"),
                              join(DATA_FOLDER, "semantic-segmentation-adas-0001.bin"));
    InputInfo::Ptr inputInfo = net.getInputsInfo()["data"];
    inputInfo->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
    inputInfo->setLayout(Layout::NHWC);
    inputInfo->setPrecision(Precision::U8);
    outputName = net.getOutputsInfo().begin()->first;

    // Initialize runnable object on CPU device
    ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");

    // Create a single processing thread
    req = execNet.CreateInferRequest();
}

void ADAS::segment(const Mat& image, Mat& mask) {
    CV_Assert(image.depth() == CV_8U);
    std::vector<size_t> dims = {1, (size_t)image.channels(), (size_t)image.rows, (size_t)image.cols};
    Blob::Ptr input = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC),
                                     image.data);
    req.SetBlob("data", input);
    req.Infer();

    int* output = req.GetBlob(outputName)->buffer();

    int rows = 1024, cols = 2048;
    mask = Mat::zeros(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mask.at<uint8_t>(i, j) = output[i * cols + j];
        }
    }
    resize(mask, mask, image.size());
}
