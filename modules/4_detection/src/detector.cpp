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
    CV_Error(Error::StsNotImplemented, "detect");
}


void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
         float threshold, std::vector<unsigned>& indices) {
    CV_Error(Error::StsNotImplemented, "nms");
}

float iou(const cv::Rect& a, const cv::Rect& b) {
    //CV_Error(Error::StsNotImplemented, "iou");
    float in = (a & b).area() ;
    float un = a.area() + b.area() - (a & b).area();
    float ratio = in / un;
    return ratio;

}
