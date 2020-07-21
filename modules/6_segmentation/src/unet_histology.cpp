#include "segmentation.hpp"

#include <opencv2/core/utils/filesystem.hpp>

using namespace InferenceEngine;
using namespace cv;
using namespace cv::utils::fs;

Blob::Ptr wrapMatToBlob(const Mat& m) {
    CV_Assert(m.depth() == CV_32F);
    std::vector<size_t> dims(&m.size[0], &m.size[0] + m.dims);
    return make_shared_blob<float>(TensorDesc(Precision::FP32, dims, Layout::ANY),
                                   (float*)m.data);
}

UNetHistology::UNetHistology() {
	Core ie;

	// Load deep learning network into memory
	CNNNetwork net = ie.ReadNetwork(join(DATA_FOLDER, "frozen_unet_histology.xml"),
                                    join(DATA_FOLDER, "frozen_unet_histology.bin"));

	outputName = net.getOutputsInfo().begin()->first;

	// Initialize runnable object on CPU device
	ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");

	// Create a single processing thread
	req = execNet.CreateInferRequest();
}

void UNetHistology::bgr2rgb(const Mat& src, Mat& dst) {
	cvtColor(src, dst, COLOR_BGR2RGB);
}


void UNetHistology::normalize(const Mat& src, Mat& dst) {
	dst.create(src.size(), CV_32FC3);
	Vec3d mean, std;

	meanStdDev(src, mean, std);

	for (size_t i = 0; i < dst.rows; i++)
	{
		for (size_t j = 0; j < dst.cols; j++)
		{
			dst.at<Vec3f>(i, j)[0] = (float)(src.at<Vec3b>(i, j)[0] - mean[0]) / std[0];
			dst.at<Vec3f>(i, j)[1] = (float)(src.at<Vec3b>(i, j)[1] - mean[1]) / std[1];
			dst.at<Vec3f>(i, j)[2] = (float)(src.at<Vec3b>(i, j)[2] - mean[2]) / std[2];
		}
	}
}

void UNetHistology::segment(const Mat& image, Mat& mask) {
	// Preprocessing
	Mat rgbImage;
	bgr2rgb(image, rgbImage);

	Mat paddedImage;
	padMinimum(rgbImage, 92, 92, paddedImage);

	Mat normalized;
	normalize(paddedImage, normalized);

	// Sanity checks before hardcoded dimensions usage
	static const int width = 1144;
	static const int height = 952;
	CV_CheckEQ(normalized.cols, width, "UNetHistology input width");
	CV_CheckEQ(normalized.rows, height, "UNetHistology input height");
	CV_CheckEQ(normalized.channels(), 3, "UNetHistology input channels");

	// Perform data permutation from 952x1144x3 to 3x952x1144
	Mat inp({1, 3, height, width, 1, 1}, CV_32F);
	std::vector<Mat> channels(3);
	for (int i = 0; i < 3; ++i)
		channels[i] = Mat(height, width, CV_32FC1, inp.ptr<float>(0, i));
	split(normalized, channels);

	Blob::Ptr inputBlob = wrapMatToBlob(inp);

	// TODO: Put inputBlob to the network, perform inference and return mask

	req.SetBlob("worker_0/validation/IteratorGetNext", inputBlob);

	req.Infer();

	int *output = req.GetBlob(outputName)->buffer();

	mask = Mat(772, 964, CV_8U);

	for (size_t i = 0; i < mask.rows; i++)
	{
		for (size_t j = 0; j < mask.cols; j++)
		{
			int ind = i * mask.cols;
			mask.at<uint8_t>(i, j) = output[ind + j];
		}
	}

	resize(mask, mask, image.size());
}

int UNetHistology::countGlands(const cv::Mat& segm) {
	int result;
	double max;
	Mat kernel = Mat::ones(3, 3, CV_8U);
	Mat sure_bg, sure_fg, dist_transform, opening;
	std::vector<std::vector<cv::Point> > conts;

	morphologyEx(segm, opening, MORPH_OPEN, kernel, Point(-1, -1), 2);

	distanceTransform(opening, dist_transform, DIST_L2, 5);
	minMaxLoc(dist_transform, NULL, &max);
	threshold(dist_transform, sure_fg, 0.6*max, 255, 0);

	sure_fg.convertTo(sure_fg, CV_8U);

	findContours(sure_fg, conts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	result = conts.size();
	return result;
}

void UNetHistology::padMinimum(const Mat& src, int width, int height, Mat& dst) {
    Mat minRow, minCol, globalMin;

    reduce(src, minRow, 0, REDUCE_MIN);
    reduce(src, minCol, 1, REDUCE_MIN);
    reduce(minCol, globalMin, 0, REDUCE_MIN);
    minRow = repeat(minRow, height, 1);
    minCol = repeat(minCol, 1, width);

    dst = repeat(globalMin, src.rows + 2*height, src.cols + 2*width);
    minRow.copyTo(dst.colRange(width, dst.cols - width).rowRange(0, height));
    minRow.copyTo(dst.colRange(width, dst.cols - width).rowRange(dst.rows - height, dst.rows));
    minCol.copyTo(dst.colRange(0, width).rowRange(height, dst.rows - height));
    minCol.copyTo(dst.colRange(dst.cols - width, dst.cols).rowRange(height, dst.rows - height));
    src.copyTo(dst.colRange(width, dst.cols - width).rowRange(height, dst.rows - height));
}
