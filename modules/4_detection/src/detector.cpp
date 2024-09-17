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
    //CV_Error(Error::StsNotImplemented, "detect");
    // Create 4D blob from BGR image
    Blob::Ptr input = wrapMatToBlob(image);
    // Pass blob as network's input. "data" is a name of input from .xml file
    req.SetBlob("image", input);
    // Launch network
    req.Infer();
    float* output = req.GetBlob(outputName)->buffer();
    int width = image.cols;
    int height = image.rows;
    std::vector<float>probs;
    std::vector<cv::Rect>box;
    std::vector<unsigned>cl;
    int xmin, xmax, ymin, ymax;
    auto numRect= req.GetBlob(outputName)->size()/7;
    for (int i = 0; i < numRect; ++i) {
        int id = i * 7;
        if (output[id+2] >= probThreshold) {
            xmin = (int)(output[id + 3] * width);
            ymin = (int)(output[id + 4] * height);
            xmax = (int)(output[id + 5] * width);
            ymax = (int)(output[id + 6] * height);
            probs.push_back(output[id + 2]);
            box.push_back(Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
            cl.push_back(output[id +1]);

        }
    }
    std::vector<uint> indices;
    nms(box, probs, nmsThreshold, indices);
    boxes = std::vector<cv::Rect>(indices.size());
    probabilities = std::vector<float>(indices.size());
    classes = std::vector<unsigned>(indices.size());
    for (auto ind: indices)
    {
        probabilities[ind]=probs[indices[ind]];
        boxes[ind]=box[indices[ind]];
        classes[ind]=cl[indices[ind]];
    }
}


void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
         float threshold, std::vector<unsigned>& indices) {
    //CV_Error(Error::StsNotImplemented, "nms");
    std::vector<cv::Rect> Copyboxes = boxes;
    std::vector<float> Copyprob = probabilities;
    std::vector<cv::Rect> box_check;
    
    for (int i = 0; i < Copyboxes.size(); ++i) {
        auto max_prob= *std::max_element(Copyprob.begin(), Copyprob.end());
        auto it = std::find(Copyprob.begin(), Copyprob.end(), max_prob);
        int index = std::distance(Copyprob.begin(), it);
        Rect box = Copyboxes[index];
        Copyboxes.erase(Copyboxes.begin() + index);
        Copyprob.erase(Copyprob.begin() + index);
        box_check.push_back(box);
        for (int j = 0; j < Copyboxes.size(); ++j) {
            if (threshold<iou(Copyboxes[j],box)){
                Copyboxes.erase(Copyboxes.begin() + j);
                Copyprob.erase(Copyprob.begin() + j);
            }
        }
    }
    for (auto b : box_check) {
        Rect Rect_b = b;
        auto it = std::find(boxes.begin(), boxes.end(), Rect_b);
        int index = std::distance(boxes.begin(), it);
        indices.push_back(index);
    }

}

float iou(const cv::Rect& a, const cv::Rect& b) {
    //CV_Error(Error::StsNotImplemented, "iou");
    float res = (float)(a & b).area() /(float)(a.area() + b.area() - (a & b).area());
    return res;
}
