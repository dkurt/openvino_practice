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
    CV_Error(Error::StsNotImplemented, "detect");
}


void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
         float threshold, std::vector<unsigned>& indices) {
	std::vector<float> prob_que;
	std::vector<unsigned> ind_que;
	std::vector<unsigned> mark(boxes.size(), 0);
	int ind;
	float IOU;
	float maxProb;

	for (size_t i = 0; i < boxes.size(); i++)
	{
		if (mark[i] == 0)
		{
			prob_que.clear();
			ind_que.clear();
			prob_que.push_back(probabilities[i]);
			ind_que.push_back(i);
			maxProb = probabilities[i];
			ind = i;

			if (i != boxes.size() - 1)
			{
				for (size_t j = i + 1; j < boxes.size(); j++)
				{
					if (mark[j] == 0)
					{
						IOU = iou(boxes[i], boxes[j]);
						if (IOU >= threshold)
						{
							prob_que.push_back(probabilities[j]);
							ind_que.push_back(j);
						}
					}
				}
			}
			else
			{
				indices.push_back(boxes.size() - 1);
				break;
			}

			for (size_t j = 0; j < prob_que.size(); j++)
			{
				if (prob_que[j] > maxProb)
				{
					maxProb = prob_que[j];
					ind = j;
				}
			}

			indices.push_back(ind_que[ind]);
			mark[ind] = 1;
		}
	}
	std::sort(indices.begin(), indices.end(), std::greater<unsigned>());
}

float iou(const cv::Rect& a, const cv::Rect& b) {
	return (float)(a & b).area() / (float)((a.area() + b.area() - (a & b).area()));
}
