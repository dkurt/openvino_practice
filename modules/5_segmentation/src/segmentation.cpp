#include "segmentation.hpp"

#include <opencv2/core/utils/filesystem.hpp>

using namespace InferenceEngine;
using namespace cv;
using namespace cv::utils::fs;

float Dice(const Mat& a, const Mat& b) {
	float dice = 2.0 * countNonZero(a&b) / (countNonZero(a)+countNonZero(b));
	return dice;
}

ADAS::ADAS() {
    Core ie;

    // Load deep learning network into memory
    auto net = ie.ReadNetwork(join(DATA_FOLDER, "semantic-segmentation-adas-0001.xml"),
                              join(DATA_FOLDER, "semantic-segmentation-adas-0001.bin"));
	// Specify preprocessing procedures
	InputInfo::Ptr inputInfo = net.getInputsInfo()["data"];
	inputInfo->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
	inputInfo->setLayout(Layout::NHWC);
	inputInfo->setPrecision(Precision::U8);
	outputName = net.getOutputsInfo().begin()->first;

	// Initialize runnable object on CPU device
	ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");

	// Create a single processing thread
	req = execNet.CreateInferRequest();
}

static Blob::Ptr wrapMatToBlob(const Mat& m) {
	CV_Assert(m.depth() == CV_8U);
	std::vector<size_t> dims = { 1, (size_t)m.channels(), (size_t)m.rows, (size_t)m.cols };
	return make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC),
		(uint8_t*)m.data);
}

void ADAS::segment(const Mat& image, Mat& mask) {
	// Create 4D blob from BGR image
	Blob::Ptr input = wrapMatToBlob(image);

	req.SetBlob("data", input);

	// Launch network
	req.Infer();
	// Copy output. 
	int32_t* output = req.GetBlob(outputName)->buffer();
	int size = req.GetBlob(outputName)->size();
	
	for (size_t i = 0; i < size; i++)
	{
		mask.push_back(float(output[i]));	
	}
	mask = mask.reshape(1, 1024);
	resize(mask, mask, Size(image.cols, image.rows));
}
