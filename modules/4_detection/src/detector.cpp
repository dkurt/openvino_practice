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

	InputInfo::Ptr inputInfo = net.getInputsInfo()["data"];

	inputInfo->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
	inputInfo->setLayout(Layout::NHWC);
	inputInfo->setPrecision(Precision::U8);
	outputName = net.getOutputsInfo().begin()->first;

	ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");

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
	Blob::Ptr input = wrapMatToBlob(image);

	req.SetBlob("data", input);

	req.Infer();

	float *output = req.GetBlob(outputName)->buffer();

	size_t out_size = req.GetBlob(outputName)->size();
	size_t reckt_num = out_size / 7;
	std::vector<float> probs;

	for (size_t i = 0; i < out_size; i++)
	{

	}
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
