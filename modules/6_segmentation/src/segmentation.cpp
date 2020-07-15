#include "segmentation.hpp"

#include <opencv2/core/utils/filesystem.hpp>

using namespace InferenceEngine;
using namespace cv;
using namespace cv::utils::fs;

float Dice(const Mat& a, const Mat& b) {
    return  (float)countNonZero(a&b) * 2.0 / ((float)countNonZero(a) + (float)countNonZero(b));
}


Blob::Ptr wrapMatToBlob2(const Mat& m) {
    CV_Assert(m.depth() == CV_8U);
    std::vector<size_t> dims = { 1, (size_t)m.channels(), (size_t)m.rows, (size_t)m.cols };
    return make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC),
        m.data);
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
    Blob::Ptr input = wrapMatToBlob2(image);

    req.SetBlob("data", input);

    req.Infer();

    const int outHeight = 1024, outWidth = 2048;
    Mat res = Mat::zeros(outHeight, outWidth, CV_8UC1);
    int* output = req.GetBlob(outputName)->buffer();
    for (int i = 0; i < outHeight; i++) {
        for (int j = 0; j < outWidth; j++) {
            res.at<uint8_t>(i, j) = output[i * outWidth + j];
        }
    }
    resize(res, mask, image.size());
    imwrite("mask.jpg", mask);
}
