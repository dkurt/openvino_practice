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
    

}


void Detector::detect(const cv::Mat& image,
                      float nmsThreshold,
                      float probThreshold,
                      std::vector<cv::Rect>& boxes,
                      std::vector<float>& probabilities,
                      std::vector<unsigned>& classes) {
    //CV_Error(Error::StsNotImplemented, "detect");
   
 

    
}


void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
         float threshold, std::vector<unsigned>& indices) {
    //CV_Error(Error::StsNotImplemented, "nms");
    unsigned count = 0;
    std::vector<unsigned> ind, nonind;
  
    
    for (int i = 0; i < boxes.size()-1; i++)
    {
        bool t= true;
        for (int j = 0; j < nonind.size(); j++)
        {
            if (i == nonind[j]) {
                t = false;
                break;
            }
        }

        if (t)
        for (int j = i+1; j < boxes.size(); j++)
        {
            
            if (iou(boxes[i], boxes[j])>threshold) {
                if (probabilities[i] > probabilities[j]) {
                    ind.push_back(i);
                    nonind.push_back(j);
                }
                else {
                    ind.push_back(j);
                    nonind.push_back(i);
                }
            }
        }
    }
    for (int i = 0; i < ind.size()-1; i++)
    {
        for (int j = 0; j < ind.size()-i-1; j++)
        {
            if (ind[j] < ind[j + 1])swap(ind[j], ind[j + 1]);
        }
    }
    indices = ind;
}

float iou(const cv::Rect& a, const cv::Rect& b) {
    //CV_Error(Error::StsNotImplemented, "iou");
    float s;
    if (abs((2*b.x+b.width-2*a.x)*(2*b.y+b.height-2*a.y))>abs((a.x+a.width-b.x)*(a.y+a.height-b.y)))
    {
        s = abs((a.x + a.width - b.x) * (a.y + a.height - b.y));
    }
    else
    {
        s = abs((2 * b.x + b.width - 2 * a.x) * (2 * b.y + b.height - 2 * a.y));
    }
 
    float SU = a.area() + b.area() - s;
    return s / SU;
}
