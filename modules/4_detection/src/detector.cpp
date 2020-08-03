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
	std::vector<unsigned>& classes){
	// Create 4D blob from BGR image

	Blob::Ptr input = wrapMatToBlob(image);

	// Pass blob as network's input. "data" is a name of input from .xml file
	req.SetBlob("image", input);

	// Launch network
	req.Infer();

	// Copy output. "prob" is a name of output from .xml file
	float* output = req.GetBlob(outputName)->buffer();

	int N = req.GetBlob(outputName)->size() / 7;

	std::vector<cv::Rect> tmp_boxes;
	std::vector<float> tmp_probabilities;
	std::vector<unsigned> tmp_classes;

	for (int i = 0; i < N; i++)
	{
		if (output[i * 7 + 2] >= probThreshold)
		{
		  cv::Rect p(static_cast<int>(output[i * 7 + 3] * image.cols), static_cast<int>(output[i * 7 + 4] * image.rows), static_cast<int>(output[i * 7 + 5] * image.cols) - static_cast<int>(output[i * 7 + 3] * image.cols) + 1,static_cast<int>(output[i * 7 + 6] * image.rows) - static_cast<int>(output[i * 7 + 4] * image.rows) + 1);
	
			tmp_boxes.push_back(p);
			tmp_probabilities.push_back(output[i * 7 + 2]);
			tmp_classes.push_back(output[i * 7 + 1]);
		}
	}

	std::vector<unsigned> indices;

	nms(tmp_boxes, tmp_probabilities, nmsThreshold, indices);

	boxes.resize(indices.size());
	probabilities.resize(indices.size());
	classes.resize(indices.size());

	for (int i = 0; i < indices.size(); i++)
	{
		unsigned pos = indices[i];
		boxes[i]= tmp_boxes[pos];
		probabilities[i]= tmp_probabilities[pos];
		classes[i]=tmp_classes[pos];
	}
}



void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
	float threshold, std::vector<unsigned>& indices) 
{
	indices.resize(boxes.size());
	
	int size = boxes.size();

	for (int i = 0; i < size; i++) 
		indices[i] = i;

	for (int i = 0; i < size; i++) 
	{
		for (int j = i+1; j < size; j++) 
		{
			if (iou(boxes[i],boxes[j]) > threshold)
			{
				if (probabilities[i] > probabilities[j]) 
					indices[j] = -1;
			
				else
				{
					indices[i] = -1;
					break;
				}
			}
		}
		for (int k = i; k > 0; --k) 
		{
			if (indices[k] != -1 && indices[k - 1] != -1 && probabilities[indices[k]] > probabilities[indices[k - 1]]) 
			{
				unsigned pos=indices[k];
				indices[k] = indices[k - 1];
				indices[k - 1] = pos;
			}
			else 
				break;
		}
	}
	indices.erase(std::remove(indices.begin(), indices.end(), -1), indices.end());
}

float iou(const cv::Rect& a, const cv::Rect& b) {

	return static_cast<float>((a & b).area()) / (a.area() + b.area() - (a & b).area());
}
