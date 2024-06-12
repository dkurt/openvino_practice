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
    inputInfo->setPrecision(Precision::U8);
    inputInfo->setLayout(Layout::NHWC);
   
    outputName = net.getOutputsInfo().begin()->first;

    // Initialize runnable object on CPU device
    ExecutableNetwork exnet = ie.LoadNetwork(net, "CPU");

    // Create a single processing thread
    request = exnet.CreateInferRequest();

}


void Detector::detect(const cv::Mat& image,
    float nmsThreshold,
    float probThreshold,
    std::vector<cv::Rect>& boxes,
    std::vector<float>& probabilities,
    std::vector<unsigned>& classes)
{
    CV_Assert(image.depth() == CV_8U);
    std::vector<size_t> dims = { 1, (size_t)image.channels(), (size_t)image.rows, (size_t)image.cols };
    Blob::Ptr input = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC), image.data);
    request.SetBlob("image", input);
    request.Infer();

    float* output = request.GetBlob(outputName)->buffer();
    int size = request.GetBlob(outputName)->size();
    for (int i = 0; i < size; i = i + 7) {
        if (output[i + 2] >= probThreshold) {
            probabilities.push_back(output[i + 2]);
            classes.push_back(output[i + 1]);
            boxes.push_back(Rect(static_cast<int>(output[i + 3] * image.cols),
                static_cast<int>(output[i + 4] * image.rows),
                static_cast<int>(output[i + 5] * image.cols)
                - static_cast<int>(output[i + 3] * image.cols) + 1,
                static_cast<int>(output[i + 6] * image.rows)
                - static_cast<int>(output[i + 4] * image.rows) + 1));
        }
    }

    std::vector<uint> indices;
    nms(boxes, probabilities, nmsThreshold, indices);
    size = boxes.size();
    int j = 0, supp = 0;
    for (int i = 0; i < size; ++i) {
        if (indices[j] != i)
        {
            boxes.erase(boxes.begin() + i - supp);
            probabilities.erase(probabilities.begin() + i - supp);
            classes.erase(classes.begin() + i - supp);
            supp++;
        }
        else
        {
            j++;
        }
    }
}

void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
         float threshold, std::vector<unsigned>& indices)
{
    indices.resize(boxes.size(), 0);
    for (int i = 1; i < boxes.size(); i++) 
    {
        indices[i] = i;
    }
    for (int i = 0; i < boxes.size(); i++)
    {
        for (int j = i + 1; j < boxes.size(); j++)
        {
            float i_ou = iou(boxes[i], boxes[j]);
            if (i_ou > threshold) 
            {
                if (probabilities[i] > probabilities[j])
                {
                    indices[j] = -1;
                }
                else
                {
                    indices[i] = -1;
                    break;
                }
            }
        }
        for (int k = i; k > 0; k--) 
        {
            if (indices[k] != -1 && indices[k - 1] != -1
                && probabilities[indices[k]] > probabilities[indices[k - 1]])
            {
                uint tmp = indices[k];
                indices[k] = indices[k - 1];
                indices[k - 1] = tmp;
            }
            else 
                break;            
        }
    }
    indices.erase(std::remove(indices.begin(), indices.end(), -1), indices.end());
  


}

float iou(const cv::Rect& a, const cv::Rect& b) 
{
    return static_cast<float>((a & b).area()) / (a.area() + b.area() - (a & b).area());
}
