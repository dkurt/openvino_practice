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
	Blob::Ptr input = wrapMatToBlob(image);

	// Pass blob as network's input. "data" is a name of input from .xml file
	req.SetBlob("image", input);

	// Launch network
	req.Infer();

	// Copy output. "prob" is a name of output from .xml file
	float* output = req.GetBlob(outputName)->buffer();
	int xmin, xmax, ymin, ymax;
	int numRect = req.GetBlob(outputName)->size() / 7;
	std::vector<Rect> _boxes;
	std::vector<float> _probabilities;
	std::vector<unsigned> _classes;
	for (int i = 0; i < numRect; i++)
	{
		int ind = i * 7;
		float prob = output[ind + 2];
		if (prob > probThreshold)
		{
			_probabilities.push_back(prob);
			int cl_ind = (int)output[ind + 1];
			_classes.push_back(cl_ind);
			xmin = output[ind + 3] * image.cols;
			ymin = output[ind + 4] * image.rows;
			xmax = output[ind + 5] * image.cols;
			ymax = output[ind + 6] * image.rows;
			Rect r(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
			_boxes.push_back(r);
		}

	}
	std::vector<unsigned> ind;
	nms(_boxes, _probabilities, nmsThreshold, ind);
	for (int i = 0; i < ind.size(); i++)
	{
		boxes.push_back(_boxes[ind[i]]);
		probabilities.push_back(_probabilities[ind[i]]);
		classes.push_back(_classes[ind[i]]);
	}
}


void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
         float threshold, std::vector<unsigned>& indices) {
	std::vector<cv::Rect> _boxes = boxes;
	std::vector<float> _probabilities = probabilities;
	std::vector<Rect> passed_box;
	while (_boxes.size() != 0)
	{
		float max_prob = 0.0;
		unsigned index_of_max_prob = 0;
		for (int i = 0; i < _probabilities.size(); i++)
			if (_probabilities[i] > max_prob)
			{
				max_prob = _probabilities[i];
				index_of_max_prob = i;
			}
		cv::Rect max_box = _boxes[index_of_max_prob];
		_boxes.erase(_boxes.begin() + index_of_max_prob);
		_probabilities.erase(_probabilities.begin() + index_of_max_prob);
		passed_box.push_back(max_box);
		for (int i = 0; i < _boxes.size(); i++)
			if (iou(_boxes[i], max_box) > threshold)
			{
				_boxes.erase(_boxes.begin() + i);
				_probabilities.erase(_probabilities.begin() + i);
			}
	}
	for (int i = 0; i < passed_box.size(); i++)
	{
		Rect search;
		search = passed_box[i];
		auto _ind = std::find(boxes.begin(), boxes.end(), search);
		int ind = std::distance(boxes.begin(), _ind);
		indices.push_back(ind);
	}
}

float iou(const cv::Rect& a, const cv::Rect& b) {
	float in = (float)((a&b).area());
	float un = (float)(a.area() + b.area() - (a&b).area());
	float ratio = in / un;
	return ratio;
}
