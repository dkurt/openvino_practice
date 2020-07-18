#include "classifier.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <inference_engine.hpp>

using namespace InferenceEngine;
using namespace cv;
using namespace cv::utils::fs;

void topK(const std::vector<float>& src, unsigned k,
          std::vector<float>& dst,
          std::vector<unsigned>& indices) {

    std::vector<std::pair<int, float*>> srcIndexed;

    for (int index = 0; index < src.size(); index++) {
        srcIndexed.push_back(std::pair<int, float*>(index, &const_cast<float &>(src[index])));
    }

    std::sort(
            srcIndexed.begin(),
            srcIndexed.end(),
            [](std::pair<int, float*> i, std::pair<int, float*> j) -> bool { return *i.second > *j.second; }
            );

    k = srcIndexed.size() < k ? srcIndexed.size() : k;

    for (int index = 0; index < k; index++) {
        indices.push_back(srcIndexed[index].first);
        dst.push_back(*srcIndexed[index].second);
    }
}

void softmax(std::vector<float>& values) {
    float sumOfExps = 0;
    float maxValue = *std::max_element(values.begin(), values.end());

    for (auto value: values) {
        sumOfExps += exp(value - maxValue);
    }

    for (auto& value: values) {
        value = exp(value - maxValue) / sumOfExps;
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

    // Initialize runnable object on CPU device
    // Throws Exception: EXC_BAD_ACCESS (code=1, address=0x80) on my PC
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
    int outputLength = req.GetBlob(outputName)->size();

    std::vector<float> outValues;

    for (int i = 0; i < outputLength; i++) {
        outValues.push_back(output[i]);
    }

    topK(outValues, (unsigned)k, probabilities, indices);
    softmax(probabilities);
}
