#include "detector.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <inference_engine.hpp>

using namespace cv;
using namespace InferenceEngine;

Detector::Detector() {
    //Creates an executable network from a network object.
    Core ie;

    // Load deep learning network into memory
    auto net = ie.ReadNetwork(utils::fs::join(DATA_FOLDER, "face-detection-0104.xml"),
                              utils::fs::join(DATA_FOLDER, "face-detection-0104.bin"));
    // InputInfo - This class contains information about each input of the network.
    InputInfo::Ptr input_info = net.getInputsInfo()["image"];
    input_info->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
    input_info->setLayout(Layout::NHWC);
    input_info->setPrecision(Precision::U8);
    out_name = net.getOutputsInfo().begin()->first;

    // initialize runnable object on CPU device.
    ExecutableNetwork exec_net = ie.LoadNetwork(net, "CPU");
    // create a single processing thread.
    request = exec_net.CreateInferRequest();
}

void Detector::detect(const cv::Mat& image,
                      float nmsThreshold,
                      float probThreshold,
                      std::vector<cv::Rect>& boxes,
                      std::vector<float>& probabilities,
                      std::vector<unsigned>& classes) {

    std::vector<size_t> vec = { 1, (size_t)image.channels(), (size_t)image.rows, (size_t)image.cols };
    Blob::Ptr input = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, vec, Layout::NHWC), image.data);
    request.SetBlob("image", input);
    request.Infer();

    float* output = request.GetBlob(out_name)->buffer();
    int size = request.GetBlob(out_name)->size() / 7;

    for (int i = 0; i < size; i++) {
        int indx = i * 7;
        float Probability = output[indx + 2];
        if (Probability > probThreshold) {
            int HEIGHT = image.rows;
            int WIDTH = image.cols;
            int xmin = output[indx + 3] * WIDTH;
            int ymin = output[indx + 4] * HEIGHT;
            int xmax = output[indx + 5] * WIDTH;
            int ymax = output[indx + 6] * HEIGHT;
            Rect rectangle(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
            boxes.push_back(rectangle);

            probabilities.push_back(Probability);
            classes.push_back(output[indx + 1]);
        }
    }
    std::vector<unsigned> indices;
    nms(boxes, probabilities, nmsThreshold, indices);
    int Size = boxes.size();
    int k = 0,j=0;
    for (int i = 0; i < Size; i++) {
        if (indices[k] == i) {
            k++;
        }
        else {
            boxes.erase(boxes.begin() + i - j);
            probabilities.erase(probabilities.begin() + i - j);
            classes.erase(classes.begin() + i - j);
            j++;
        }
    }
}


bool comp(const std::pair<int, float>& a, const std::pair<int, float>& b) {
    return a.second > b.second;
}

//Non-maximum Suppression
void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
         float threshold, std::vector<unsigned>& indices) {

    std::vector <std::pair<int, float> > tmp;
    for (int i = 0; i < boxes.size(); i++) {
        tmp.push_back(std::make_pair(i, probabilities[i]));
    }
    for (int i = 0; i < boxes.size(); i++) {
        for (int j = i + 1; j < boxes.size(); j++) {
            if (iou(boxes[i], boxes[j]) > threshold) {
                if (tmp[i].second > tmp[j].second) {
                    tmp[j].first = -1;
                }
                else {
                    tmp[i].first = -1;
                    break;
                }
            }
        }
        std::sort(tmp.begin(), tmp.end(), comp);
    }
    for (int i = 0; i < boxes.size(); i++) {
        if (tmp[i].first != -1) {
            indices.push_back(tmp[i].first);
        }
    }
}

float iou(const cv::Rect& a, const cv::Rect& b) {
    float INTERSECTION = (a & b).area();
    float UNION = a.area() + b.area();
    return INTERSECTION / (UNION - INTERSECTION);
}
