#include "segmentation.hpp"

#include <opencv2/core/utils/filesystem.hpp>

using namespace InferenceEngine;
using namespace cv;
using namespace cv::utils::fs;

float Dice(const Mat& a, const Mat& b) {
	size_t aSize = countNonZero(a);
	size_t bSize = countNonZero(b);
	size_t a_and_b_size = countNonZero(a & b);

	return (float)(2 * a_and_b_size) / (float)(aSize + bSize);
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

	ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");

	req = execNet.CreateInferRequest();
}

Blob::Ptr wrapMatToBlob_segm(const Mat& m) {
	CV_Assert(m.depth() == CV_8U);
	std::vector<size_t> dims = { 1, (size_t)m.channels(), (size_t)m.rows, (size_t)m.cols };
	return make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC),
		m.data);
}


void ADAS::segment(const Mat& image, Mat& mask) {
	Blob::Ptr input = wrapMatToBlob_segm(image);
	req.SetBlob("data", input);

	req.Infer();

	int *output = req.GetBlob(outputName)->buffer();

	mask = Mat(1024, 2048, CV_8U);

	for (size_t i = 0; i < 1024; i++)
	{
		for (size_t j = 0; j < 2048; j++)
		{
			int ind = i * 2048;
			if (output[ind + j] != 13)
			{
				mask.at<uint8_t>(i, j) = 0;
			}
			else
			{
				mask.at<uint8_t>(i, j) = 13;
			}
		}
	}

	resize(mask, mask, image.size());
}
