#include "detector.hpp"
#include <utility> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <inference_engine.hpp>
#include <algorithm>

using namespace cv;
using namespace InferenceEngine;
using namespace cv::utils::fs;

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

Blob::Ptr wrapMatToBlob(const Mat& m) {
	CV_Assert(m.depth() == CV_8U);
	std::vector<size_t> dims = { 1, (size_t)m.channels(), (size_t)m.rows, (size_t)m.cols };
	return make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC),
		m.data);
}

void Detector::detect(const cv::Mat& image, const float nmsThreshold, 
					  const float probThreshold, std::vector<cv::Rect>& boxes,
                      std::vector<float>& probabilities,
                      std::vector<unsigned>& classes) {
	// Create 4D blob from BGR image
	Blob::Ptr input = wrapMatToBlob(image);

	req.SetBlob("image", input);

	// Launch network
	req.Infer();
	auto w = image.cols;
	auto h = image.rows;

	// Copy output. 
	float* output = req.GetBlob(outputName)->buffer();
	int size = req.GetBlob(outputName)->size();
	float score = 0;
	float cls = 0;
	float id = 0;
	for (int i = 0; i < size/7; i++)
	{
		score = output[i * 7 + 2];
		cls = output[i * 7 + 1];
		id = output[i * 7];
		if (id >= 0 && score > probThreshold && cls <= 1)
		{
			classes.push_back(cls);
			probabilities.push_back(score);
			// 3 - xmin, 4 - ymin, 5 - xmax, 6 - ymax
			int xmin = output[i * 7 + 3] * w;
			int xmax = output[i * 7 + 5] * w;
			float ymin = output[i * 7 + 4] * h;
			float ymax = output[i * 7 + 6] * h;
			boxes.push_back(Rect(xmin, ymin, int(xmax-xmin) + 1, int(ymax - ymin) + 1));
		}
	}
	std::vector<unsigned> indices;
	nms(boxes, probabilities, nmsThreshold, indices);
	//todo
}



float IoU(cv::Rect A, cv::Rect B) {
	float inters = (A & B).area();
	float uni = A.area() + B.area() - inters;
	float iou = inters / uni;

	return iou;
}


void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
	float threshold, std::vector<unsigned>& indices) {

	std::vector<cv::Rect> result;
	std::vector<cv::Rect> b;
	std::vector<float> prob;
	copy(boxes.begin(), boxes.end(), back_inserter(b));
	copy(probabilities.begin(), probabilities.end(), back_inserter(prob));

	while (!b.empty()) {
		//std::cout << "while has started " << std::endl;
		auto m = std::max_element(prob.begin(), prob.end()) - prob.begin();
		//std::cout << "m = " << m << std::endl;
		cv::Rect M;
		if (b.size() == 1)
		{
			M = b.at(0);
			b.clear();
		}
		else
		{
			M = b.at(m);
			b.erase(b.begin() + m);
			prob.erase(prob.begin() + m);
		}
		
		std::cout << "M " << M << std::endl;

		result.push_back(M);
		/*for (size_t i = 0; i < result.size(); i++)
		{
			std::cout << "result " << result[i] << std::endl;
		}
		
		
		for (size_t i = 0; i < b.size(); i++)
		{
			std::cout << "left boxes" << b[i] << std::endl;
		}*/

		for (size_t i = 0; i < b.size(); i++)
		{
			//std::cout << "loop begin" <<  std::endl;
			//std::cout << "iou " << IoU(M, b.at(i)) << std::endl;
			if (IoU(M, b.at(i)) > threshold)
			{
				b.erase(b.begin() + i);
				prob.erase(prob.begin() + i);
				/*for (size_t i = 0; i < b.size(); i++)
				{
					std::cout << "if worked.left boxes" << b[i] << std::endl;
				}*/
			}
		}
	}

	for (int i = 0; i < result.size(); i++)
	{
		cv::Rect d = result.at(i);
		auto it = std::find(boxes.begin(), boxes.end(), d);
		auto index = std::distance(boxes.begin(), it);
		indices.push_back(index);
	}
	for (size_t i = 0; i <probabilities.size(); i++)
	{
		std::cout << "probs are " << probabilities[i] << std::endl;
	}
}