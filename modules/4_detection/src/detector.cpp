#include "detector.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <inference_engine.hpp>

using namespace cv;
using namespace InferenceEngine;

Blob::Ptr wrapMatToBlob(const Mat& m) {
	CV_Assert(m.depth() == CV_8U);
	std::vector<size_t> dims = { 1, (size_t)m.channels(), (size_t)m.rows, (size_t)m.cols };
	return make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC),
		m.data);
}

Detector::Detector() {
    Core ie;

    // Load deep learning network into memory
    auto net = ie.ReadNetwork(utils::fs::join(DATA_FOLDER, "face-detection-0104.xml"),
                              utils::fs::join(DATA_FOLDER, "face-detection-0104.bin"));
	// Specify preprocessing procedures
	// (NOTE: this part is different for different models!)
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
	// Create 4D blob from BGR image
	
	Blob::Ptr input = wrapMatToBlob(image);

	// Pass blob as network's input. "data" is a name of input from .xml file
	req.SetBlob("image", input);

	// Launch network
	req.Infer();

	// Copy output. "prob" is a name of output from .xml file
	float* output = req.GetBlob(outputName)->buffer();

	int N= req.GetBlob(outputName)->size()/7;
	std::vector<cv::Rect> tmp_boxes;
	std::vector<float> tmp_probabilities;
	std::vector<unsigned> tmp_classes;
	for (int i = 0; i < N; i++)
	{
		if (output[i + 2] >= probThreshold);
		{
			int x1 = output[i*7 + 3] * image.cols;
			int y1 = output[i*7 + 4] * image.rows;
			int x2 = output[i*7 + 5] * image.cols;
			int y2 = output[i*7 + 6] * image.rows;
		
			cv::Rect p(x1, y1, x2-x1 + 1, y2-y1 + 1);
			tmp_boxes.push_back(p);
			tmp_probabilities.push_back(output[i*7 + 2]);
			tmp_classes.push_back(output[i*7 + 1]);
		}
	}
	std::vector<unsigned> indices;
	nms(tmp_boxes, tmp_probabilities, nmsThreshold, indices);
	
	boxes = std::vector<cv::Rect>(indices.size());
	probabilities = std::vector<float>(indices.size());
	classes = std::vector<unsigned>(indices.size());
	for (int i = 0; i < indices.size(); i++)
	{
		unsigned pos = indices[i];
		boxes.push_back(tmp_boxes[pos]);
		probabilities.push_back(tmp_probabilities[pos]);
		classes.push_back(tmp_classes[pos]);
	}
}


void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
	float threshold, std::vector<unsigned>& indices) 
{
	indices.resize(boxes.size());
	int size = boxes.size();
	std::vector<unsigned> tmp_indices;
	for (int i = 1; i < boxes.size(); ++i)
		tmp_indices[i] = i;

	for (int i = 0; i < size; i++)
		for (int j = i + 1; j < size; j++)
		{
			if (iou(boxes[i], boxes[j]) > threshold)
			{
				if (probabilities[i] > probabilities[j])
				{
					tmp_indices[j] = -1;

				}
				else
				{
					tmp_indices[i] = -1;
					break;
				}

			}
		}
	tmp_indices.erase(std::remove(tmp_indices.begin(), tmp_indices.end(), -1), tmp_indices.end());
	for (int i = 1; i < indices.size(); ++i)
		indices.push_back(tmp_indices[i]);
}
float iou(const cv::Rect& a, const cv::Rect& b) {

	return static_cast<float>((a & b).area()) / (a.area() + b.area() - (a & b).area());
}
