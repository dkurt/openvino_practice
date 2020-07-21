#include "segmentation.hpp"

#include <opencv2/core/utils/filesystem.hpp>

using namespace InferenceEngine;
using namespace cv;
using namespace cv::utils::fs;

float Dice(const Mat& a, const Mat& b) {
    //CV_Error(Error::StsNotImplemented, "Dice score computation");
    return (2.0f * countNonZero(a & b) / (countNonZero(a) + countNonZero(b)));
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

static Blob::Ptr wrapMatToBlob(const Mat& m) {
    CV_Assert(m.depth() == CV_8U);
    std::vector<size_t> dims = { 1, (size_t)m.channels(), (size_t)m.rows, (size_t)m.cols };
    return make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC),
        (uint8_t*)m.data);
}

void ADAS::segment(const Mat& image, Mat& mask) {
    //CV_Error(Error::StsNotImplemented, "ADAS semantic segmentation");
    Blob::Ptr input = wrapMatToBlob(image);
    // Pass blob as network's input. "data" is a name of input from .xml file
    req.SetBlob("data", input);
    // Launch network
    req.Infer();
    int* output = req.GetBlob(outputName)->buffer();
    auto width = image.cols;
    auto height = image.rows;
    mask = Mat(req.GetBlob(outputName)->size(), 1, CV_32SC1, output);
    mask = mask.reshape(1, 1024);
    mask.convertTo(mask, CV_32F);
    resize(mask, mask, Size(width, height));
    mask.convertTo(mask, CV_8UC1);
}