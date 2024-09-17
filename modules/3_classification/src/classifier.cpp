#include "classifier.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <inference_engine.hpp>

#include <cmath>
#include <algorithm>

using namespace InferenceEngine;
using namespace cv;
using namespace cv::utils::fs;

void topK(const std::vector<float>& src, unsigned k,
          std::vector<float>& dst,
          std::vector<unsigned>& indices) {
    dst.resize(5, 0);
    indices.resize(5, 0);
    for (int i = 0; i < src.size(); ++i) {
        for (int j = k - 1; j >= 0; --j) {
            if (src[i] > dst[j]) {
                if (j < k - 1) {
                    dst[j + 1] = dst[j];
                    indices[j + 1] = indices[j];
                }
                dst[j] = src[i];
                indices[j] = i;
            }
        }
    }
}

void softmax(std::vector<float>& values) {
    float sum = 0;
    float max = *std::max_element(values.begin(), values.end());
    for (int i = 0; i < values.size(); ++i) {
        sum += exp(values[i] - max);
    }
    for (int i = 0; i < values.size(); ++i) {
        values[i] = exp(values[i] - max) / sum;
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
    int size = req.GetBlob(outputName)->size();
    std::vector<float> src(size);
    for (int i = 0; i < size; ++i) {
        src[i] = output[i];
    }

    topK(src, k, probabilities, indices);
    softmax(probabilities);
}
