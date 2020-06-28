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
	// (NOTE: this part is different for different models!)
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
	//CV_Assert(m.depth() == CV_8U);
	std::vector<size_t> dims = { 1, (size_t)m.channels(), (size_t)m.rows, (size_t)m.cols };
	return make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC),
		(uint8_t*)m.data);
}

void ADAS::segment(const Mat& image, Mat& mask) {
	// Create 4D blob from BGR image
	Mat resized;
	resize(image, resized, Size(1024, 2048));
	//resized.convertTo(resized, BGR);
	Blob::Ptr input = wrapMatToBlob(resized);

	req.SetBlob("data", input);

	// Launch network
	req.Infer();
	//auto w = image.cols;
	//auto h = image.rows;

	// Copy output. 
	float* output = req.GetBlob(outputName)->buffer();
	int size = req.GetBlob(outputName)->size();

	//std::memcpy(mask.data, output, image.rows * image.cols * sizeof(uchar));
	std::cout << "size is " << size << std::endl;
	int cars = 0;
	for (size_t i = 0; i < size; i++)
	{
		mask.push_back(output[i]);
		if (output[i] == 13)
		{
			cars += 1;
		}
		//std::cout << output[i] << " ";
		/*for (size_t j = 0; j < resized.cols; j++)
		{
			mask.at<float>(i, j) = output[i];
			
			//mask.push_back(output[i, j]);
		}*/
			
	}
	std::cout << "car segment " << cars << std::endl;
	mask = mask.reshape(1, resized.rows);
	imshow("original mask", mask);
	waitKey(0);
	resize(mask, mask, Size(image.cols, image.rows));
}
