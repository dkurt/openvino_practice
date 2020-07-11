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


void Detector::detect(const cv::Mat& image,
                      float nmsThreshold,
                      float probThreshold,
                      std::vector<cv::Rect>& boxes,
                      std::vector<float>& probabilities,
                      std::vector<unsigned>& classes) {
    std::vector<size_t> dims = {1, (size_t)image.channels(), (size_t)image.rows, (size_t)image.cols};
    Blob::Ptr input = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC), image.data);

    req.SetBlob("image", input);
    req.Infer();

    float* output = req.GetBlob(outputName)->buffer();
    int size = req.GetBlob(outputName)->size()/7;

    int indx;
    for (int i = 0; i < size; i = ++i) {
        indx = i * 7;
        if (output[indx + 2] > probThreshold) {
            int xmin = static_cast<int>(output[indx + 3] * image.cols);
            int ymin = static_cast<int>(output[indx + 4] * image.rows);
            int xmax = static_cast<int>(output[indx + 5] * image.cols);
            int ymax = static_cast<int>(output[indx + 6] * image.rows);
            Rect rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
            boxes.push_back(rect);

            probabilities.push_back(output[indx + 2]);
            classes.push_back(output[indx + 1]);
        }
    }
    std::vector<unsigned> indices;
    nms(boxes, probabilities, nmsThreshold, indices);
    size = boxes.size();

    int j, k = 0;
    int dist;
    for (int i = 0; i < size; ++i) {
        if (indices[k] != i) {
            dist = i - j;
            boxes.erase(boxes.begin() + dist);
            probabilities.erase(probabilities.begin() + dist);
            classes.erase(classes.begin() + dist);
            j++;
        } else {
            k++;
        }
    }
 }

void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
         float threshold, std::vector<unsigned>& indices){

    std::vector <std::pair<int,float> > spec;
    for (int i = 0; i < boxes.size(); ++i) {
        spec.push_back(std::make_pair(i,probabilities[i]));
    }

    for (int i = 0; i < boxes.size(); ++i) {
        for (int j = i + 1; j < boxes.size(); ++j) {
            if (iou(boxes[i], boxes[j]) > threshold) {
                if (spec[i].second > spec[j].second) {
                    spec[j].first = -1;
                } else {
                    spec[i].first = -1;
                    break;
                }
            }
        }
    } 

    std::sort(spec.begin(), spec.end(), [](const std::pair<int,float>& F, const std::pair<int,float>& S){
            if(S.second < F.second)
                return true;
            return false; 
            });
    for (int i = 0; i < boxes.size(); ++i){
        if(spec[i].first != -1)
            indices.push_back(spec[i].first);
    }
}	

float iou(const cv::Rect& a, const cv::Rect& b) {
    float Intersection = (a & b).area();
    float Union = a.area() + b.area();
    return Intersection / (Union - Intersection);
}
