#include "classifier.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <inference_engine.hpp>
#include <algorithm>

using namespace InferenceEngine;
using namespace cv;
using namespace cv::utils::fs;

void topK(const std::vector<float>& src, unsigned k, std::vector<float>& dst, std::vector<unsigned>& indices) {

    std::vector<float> temp(src);
    std::sort(begin(temp), end(temp));
    std::reverse(begin(temp), end(temp));
    for (int i = 0; i < k; ++i) {
        dst.push_back(temp.at(i));

        auto it = std::find(begin(src), end(src), temp.at(i));
        indices.push_back(std::distance(begin(src), it));
    }
}
void softmax(std::vector<float>& values) {
    std::vector<float> tempValues(values);
    float total = 0;
    float maxValue = *std::max_element(begin(values), end(values));
    for (int i = 0; i < values.size(); ++i) {
        total += exp(tempValues.at(i) - maxValue);
    }
    for (int i = 0; i < values.size(); ++i) {
        values.at(i) = exp(tempValues.at(i) - maxValue) / total;
    }
}

Blob::Ptr wrapMatToBlob(const Mat& m) {
    CV_Assert(m.depth() == CV_8U);
    std::vector<size_t> dims = { 1, (size_t)m.channels(), (size_t)m.rows, (size_t)m.cols };
    return make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC),
        m.data);
}

Classifier::Classifier() {
    Core ie;
    CNNNetwork net = ie.ReadNetwork(join(DATA_FOLDER, "DenseNet_121.xml"), join(DATA_FOLDER, "DenseNet_121.bin"));

    InputInfo::Ptr inputInfo = net.getInputsInfo()["data"];
    inputInfo->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
    inputInfo->setLayout(Layout::NHWC);
    inputInfo->setPrecision(Precision::U8);
    outputName = net.getOutputsInfo().begin()->first;
    ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");
    req = execNet.CreateInferRequest();
}

void Classifier::classify(const cv::Mat& image, int k, std::vector<float>& probabilities, std::vector<unsigned>& indices) {
    Blob::Ptr input = wrapMatToBlob(image);
    req.SetBlob("data", input);

    req.Infer();

    float* output = req.GetBlob(outputName)->buffer();
    std::vector<float> temp;
    for (int i = 0; i < req.GetBlob(outputName)->size(); ++i) {
        temp.push_back(output[i]);
    }
    topK(temp, k, probabilities, indices);
    softmax(probabilities);
}