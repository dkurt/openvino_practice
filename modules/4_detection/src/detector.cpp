#include "detector.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <inference_engine.hpp>

using namespace cv;
using namespace InferenceEngine;

Detector::Detector() {
    Core ie;

    // Load deep learning network into memory
    auto net = ie.ReadNetwork(utils::fs::join(DATA_FOLDER, "face-detection-0104.xml"),
                              utils::fs::join(DATA_FOLDER, "face-detection-0104.bin"));
    
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

    // Pass blob as network's input. "image" is a name of input from .xml file
    req.SetBlob("image", input);

    // Launch network
    req.Infer();
    float* output = req.GetBlob(outputName)->buffer();
    int n = req.GetBlob(outputName)->size()/7;
    std::vector<cv::Rect> boxesbuf;
    std::vector<float> probabilitiesbuf;
    std::vector<unsigned> classesbuf;
    int height = image.rows;
    int width = image.cols;
    for (int i = 0; i < n ; i++) {
        int Index = i * 7;
        if (output[Index + 2] > probThreshold) {

            int xmin = (int)(output[Index + 3] * width);
            int ymin = (int)(output[Index + 4] * height);
            int xmax = (int)(output[Index + 5] * width);
            int ymax = (int)(output[Index + 6] * height);
            Rect appRect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
            boxesbuf.push_back(appRect);
            probabilitiesbuf.push_back(output[Index + 2]);
            classesbuf.push_back(output[Index + 1]);
        }
    }

    std::vector<unsigned> indices;
    nms(boxesbuf, probabilitiesbuf, nmsThreshold, indices);
    boxes = std::vector<cv::Rect>(indices.size());
    probabilities = std::vector<float>(indices.size());
    classes = std::vector<unsigned>(indices.size());
    for (int i = 0; i < indices.size(); i++) {
        boxes[i] = boxesbuf[indices[i]];
        probabilities[i] = probabilitiesbuf[indices[i]];
        classes[i] = classesbuf[indices[i]];
    }
}


void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
         float threshold, std::vector<unsigned>& indices) {
    std::set<unsigned> index;
    indices = std::vector<unsigned>();
    std::vector<std::pair<cv::Rect, unsigned>> boxespair = std::vector<std::pair<cv::Rect, unsigned>>(boxes.size());
    for (int i = 0; i < boxes.size(); i++) {
        boxespair[i] = std::make_pair(boxes[i], i);
        index.insert(i);
    }
    while (index.size() != 0) {
        float max = 0.0f;
        int maxind = 0;
        for (auto i : index) {
            if (probabilities[i] > max) {
                max = probabilities[i];
                maxind = i;
            }
        }
        std::pair<cv::Rect, unsigned> better = std::make_pair(boxespair[maxind].first, boxespair[maxind].second);
        indices.push_back(maxind);
        index.erase(maxind);
        for (auto i:index) {
            float ioubuf = iou(better.first, boxespair[i].first);
            if (ioubuf > threshold) {
                index.erase(i);
            }
        }
    }
}

float iou(const cv::Rect& a, const cv::Rect& b) {
    float metric = (float)(a & b).area() / (float)(a.area() + b.area() - (a & b).area());
    return metric;
}
