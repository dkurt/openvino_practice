#include "detector.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <inference_engine.hpp>

#include <algorithm>

using namespace cv;
using namespace InferenceEngine;

Detector::Detector() {
    Core ie;

    // Load deep learning network into memory
    auto net = ie.ReadNetwork(utils::fs::join(DATA_FOLDER, "face-detection-0104.xml"),
                              utils::fs::join(DATA_FOLDER, "face-detection-0104.bin"));
    // Specify preprocessing procedures
    // (NOTE: this part is different for different models!)
    InputInfo::Ptr inputInfo = net.getInputsInfo()["image"];
    inputInfo->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
    inputInfo->setLayout(Layout::NHWC);
    inputInfo->setPrecision(Precision::U8);
    outputName = net.getOutputsInfo().begin()->first;

    // Initialize runnable object on CPU device
    ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");

    // Create a single processing thread
    req = execNet.CreateInferRequest();
}


Blob::Ptr wrapMatToBlob(const Mat& m) {
    CV_Assert(m.depth() == CV_8U);
    std::vector<size_t> dims = { 1, (size_t)m.channels(), (size_t)m.rows, (size_t)m.cols };
    return make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC),
        m.data);
}

void Detector::detect(const cv::Mat& image,
                      float nmsThreshold,
                      float probThreshold,
                      std::vector<cv::Rect>& boxes,
                      std::vector<float>& probabilities,
                      std::vector<unsigned>& classes) {
    // Create 4D blob from BGR image
    Blob::Ptr input = wrapMatToBlob(image);

    // Pass blob as network's input. "data" is a name of input from .xml file
    req.SetBlob("image", input);

    // Launch network
    req.Infer();

    // Copy output. "prob" is a name of output from .xml file
    float* output = req.GetBlob(outputName)->buffer();
    size_t numRect = req.GetBlob(outputName)->size();
    std::vector<cv::Rect> boxesVec;
    std::vector<float> probsVec;
    std::vector<unsigned> classVec;

    for (int i = 0; i < numRect / 7; i++) {
        if (output[i * 7 + 2] > probThreshold) {
            int xmin = output[i * 7 + 3] * image.cols;
            int ymin = output[i * 7 + 4] * image.rows;
            int xmax = output[i * 7 + 5] * image.cols;
            int ymax = output[i * 7 + 6] * image.rows;
            boxesVec.push_back(Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
            probsVec.push_back(output[i * 7 + 2]);
            classVec.push_back(output[i * 7 + 1]);
        }
    }

    std::vector<unsigned> ind;
    nms(boxesVec, probsVec, nmsThreshold, ind);
    for (auto i : ind) {
        boxes.push_back(boxesVec.at(i));
        probabilities.push_back(probsVec.at(i));
        classes.push_back(classVec.at(i));
    }
}


void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
         float threshold, std::vector<unsigned>& indices) {

    std::vector<cv::Rect> boxesVec = boxes;
    std::vector<float> probsVec = probabilities;
    std::vector<cv::Rect> newBoxesVec;
    while (boxesVec.size())
    {
        float maxProb = *std::max_element(probsVec.begin(), probsVec.end());
        auto iterMaxProb = std::find(probsVec.begin(), probsVec.end(), maxProb);
        int indexMaxProb = std::distance(probsVec.begin(), iterMaxProb);

        cv::Rect maxBox = boxesVec.at(indexMaxProb);
        boxesVec.erase(boxesVec.begin() + indexMaxProb);
        probsVec.erase(probsVec.begin() + indexMaxProb);

        newBoxesVec.push_back(maxBox);

        for (int i(0); i < boxesVec.size(); i++) {
            if (iou(boxesVec.at(i), maxBox) > threshold)
            {
                boxesVec.erase(boxesVec.begin() + i);
                probsVec.erase(probsVec.begin() + i);
            }
        }
    }

    for (auto i : newBoxesVec) {
        Rect tempRect = i;
        auto tempIter = std::find(boxes.begin(), boxes.end(), tempRect);
        int tempIndex = std::distance(boxes.begin(), tempIter);
        indices.push_back(tempIndex);
    }    
}

float iou(const cv::Rect& a, const cv::Rect& b) {
    float intersectionR = (a & b).area();
    float unionR = a.area() + b.area()- (a & b).area();
    return intersectionR / (unionR);
}