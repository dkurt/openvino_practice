#include "classifier.hpp"

#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <inference_engine.hpp>

using namespace InferenceEngine;
using namespace cv;
using namespace cv::utils::fs;

void topK(const std::vector<float>& src, unsigned k,
          std::vector<float>& dst,
          std::vector<unsigned>& indices) {
    size_t n = src.size();
    std::map<float, unsigned> srcidx;
    for (size_t i = 0; i < n; i++) {
        srcidx.insert(std::make_pair(src[i], i));
    }
    dst = std::vector<float>(k);
    indices = std::vector<unsigned>(k);
    auto it = srcidx.cend();
    for (size_t i = 0; i < k; i++) {
        it--;
        dst[i] = (*it).first;
        indices[i] = (*it).second;
    }
}

void softmax(std::vector<float>& values) {
    size_t n = values.size();

    // normalize vector
    float maxVal = *std::max_element(values.begin(), values.end());
    for (size_t i = 0; i < n; i++) {
        values[i] -= maxVal;
    }

    // calc denominator 
    float den = 0.0f;
    for (size_t i = 0; i < n; i++) {
        den += cv::exp(values[i]);
    }

    for (size_t i = 0; i < n; i++) {
        values[i] = cv::exp(values[i]) / den;
    }
}

Blob::Ptr wrapMatToBlob(const Mat& m) {
    CV_Assert(m.depth() == CV_8U);
    std::vector<size_t> dims = {1, (size_t)m.channels(), (size_t)m.rows, (size_t)m.cols};
    return make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC),
                                     m.data);
}

Classifier::Classifier() {
    Core ie;

    // Load deep learning network into memory
    CNNNetwork net = ie.ReadNetwork(join(DATA_FOLDER, "DenseNet_121.xml"),
                                    join(DATA_FOLDER, "DenseNet_121.bin"));

    // Specify preprocessing procedures
    // (NOTE: this part is different for different models!)
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

void Classifier::classify(const cv::Mat& image, int k, std::vector<float>& probabilities,
                          std::vector<unsigned>& indices) {
    // Create 4D blob from BGR image
    Blob::Ptr input = wrapMatToBlob(image);

    // Pass blob as network's input. "data" is a name of input from .xml file
    req.SetBlob("data", input);

    // Launch network
    req.Infer();

    // Copy output. "prob" is a name of output from .xml file
    float* output = req.GetBlob(outputName)->buffer();
    size_t n = req.GetBlob(outputName)->size();
    std::vector<float> src(n);
    for (int i = 0; i < n; i++) {
        src[i] = output[i];
    }
    probabilities = std::vector<float>(k);
    indices = std::vector<unsigned>(k);
    topK(src, k, probabilities, indices);
    softmax(probabilities);
}
