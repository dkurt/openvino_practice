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
    /*std::vector<std::pair<float, unsigned>> pairtop(src.size());
    for (int i = 0; i < src.size(); i++){
        pairtop[i]=std::make_pair(src[i], i);
    }
    std::sort(pairtop.begin(), pairtop.end(), std::greater<std::pair<float,unsigned>>());*/
    std::map<float, unsigned> srcidx;
    for (size_t i = 0; i < src.size(); i++) {
        srcidx.insert(std::make_pair(src[i], i));
    }
    dst = std::vector<float>(k);
    indices = std::vector<unsigned>(k);
    auto it = srcidx.cend();
    for (int i = 0; i < k; i++){
        //dst[i] = pairtop[i].first;
        //indices[i] = pairtop[i].second;
        it--;
        dst[i] = (*it).first;
        indices[i] = (*it).second;
    }
}

void softmax(std::vector<float>& values) {
    float buf = 0.0f;
    float max = *std::max_element(values.begin(),values.end());
    for (int i = 0; i < values.size(); i++) {
        values[i] -= max;
    }
    for (int i = 0; i < values.size(); i++){
        buf += std::exp(values[i]);
    }
    for (int i = 0; i < values.size(); i++) {
        values[i] = std::exp(values[i]) / buf;
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
    int n = req.GetBlob(outputName)->size();
    std::vector<float> buf(n);
    for (int i = 0; i < n; i++) {
        buf[i] = output[i];
    }
    probabilities = std::vector<float>(k);
    indices = std::vector<unsigned>(k);
    topK(buf, k, probabilities, indices);
    softmax(probabilities);
}
