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

    // Initialize runnable object on CPU device
    ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");

    // Create a single processing thread
    req = execNet.CreateInferRequest();

    outputName = net.getOutputsInfo().begin()->first;
}


static Blob::Ptr wrapMatToBlob(const Mat& m) {
    CV_Assert(m.depth() == CV_8U);
    std::vector<size_t> dims = { 1, (size_t)m.channels(), (size_t)m.rows, (size_t)m.cols };
    return make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC),
        (uint8_t*)m.data);
}


void ADAS::segment(const Mat& image, Mat& mask) {
    Blob::Ptr input = wrapMatToBlob(image);

    req.SetBlob("data", input);

    // Launch network
    req.Infer();

    // Copy output. 
    int* output = req.GetBlob(outputName)->buffer();

    for (int i = 0; i < req.GetBlob(outputName)->size(); i++)
        mask.push_back(float(output[i]));

    int reshapingVal = 1024;

    mask = mask.reshape(1, reshapingVal);

    Size imageSize = Size(image.cols, image.rows);

    resize(mask, mask, imageSize);

    mask.convertTo(mask, CV_8UC1);
}