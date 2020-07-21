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
    // InputInfo - This class contains information about each input of the network.
    InputInfo::Ptr input_info = net.getInputsInfo()["image"];
    input_info->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
    input_info->setLayout(Layout::NHWC);
    input_info->setPrecision(Precision::U8);
    out_name = net.getOutputsInfo().begin()->first;
    ExecutableNetwork exec_net = ie.LoadNetwork(net, "CPU");
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
    int k = 0, j = 0;
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



void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
    float threshold, std::vector<unsigned>& indices) {
    //CV_Error(Error::StsNotImplemented, "nms");

    std::vector<int> ind, nonind;
    for (int i = 0; i < boxes.size(); i++)
    {
        ind.push_back(i);
    }

    for (int i = 0; i < boxes.size() - 1; i++)
    {
        for (int j = i + 1; j < boxes.size(); j++)
        {

            if (iou(boxes[i], boxes[j]) > threshold) {
                if (probabilities[i] > probabilities[j]) {
                    ind[j] = -1;

                }
                else {
                    ind[i] = -1;
                    break;
                }
            }
        }
    }

    indices.clear();
    for (int i = 0; i < ind.size(); i++)
    {
        if (ind[i] >= 0) indices.push_back(ind[i]);
    }

    for (int i = 0; i < indices.size() - 1; i++)
    {
        
        for (int j = 0; j < indices.size() - i - 1; j++)
        {
            
            if (probabilities[indices[j]] < probabilities[indices[j + 1]])swap(indices[j], indices[j + 1]);
        }
    }
    
}

float iou(const cv::Rect& a, const cv::Rect& b) {
    //CV_Error(Error::StsNotImplemented, "iou");
    float s;

    s = (a & b).area();
 
    float SU = a.area() + b.area() - s;
    return s / SU;
}
