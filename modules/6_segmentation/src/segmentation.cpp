#include "segmentation.hpp"

#include <opencv2/core/utils/filesystem.hpp>

using namespace InferenceEngine;
using namespace cv;
using namespace cv::utils::fs;

float Dice(const Mat& a, const Mat& b) {
	int _ab = countNonZero(a&b);
	int _a = countNonZero(a);
	int _b = countNonZero(b);
	float res = (float)(2.0 * _ab / (_a + _b));
	return res;
}

static Blob::Ptr wrapMatToBlob(const Mat& m) {
	CV_Assert(m.depth() == CV_8U);
	std::vector<size_t> dims = { 1, (size_t)m.channels(), (size_t)m.rows, (size_t)m.cols };
	return make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC),
		m.data);
}

ADAS::ADAS() {
    Core ie;

    // Load deep learning network into memory
    auto net = ie.ReadNetwork(join(DATA_FOLDER, "semantic-segmentation-adas-0001.xml"),
                              join(DATA_FOLDER, "semantic-segmentation-adas-0001.bin"));
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



void ADAS::segment(const Mat& image, Mat& mask) {
	Blob::Ptr input = wrapMatToBlob(image);

	// Pass blob as network's input. "data" is a name of input from .xml file
	req.SetBlob("data", input);

	// Launch network
	req.Infer();

	// Copy output. "prob" is a name of output from .xml file
	int* output = req.GetBlob(outputName)->buffer();
	mask = Mat(req.GetBlob(outputName)->size(), 1, CV_32SC1, output);
	mask = mask.reshape(1, 1024);
	mask.convertTo(mask, CV_32F);
	resize(mask, mask, Size(image.cols, image.rows));
	mask.convertTo(mask, CV_8UC1);
}
