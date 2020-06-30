#include "segmentation.hpp"

#include <opencv2/core/utils/filesystem.hpp>

using namespace InferenceEngine;
using namespace cv;
using namespace cv::utils::fs;
using namespace std;

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

    // Initialize runnable object on CPU device
    ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");

    // Create a single processing thread
    req = execNet.CreateInferRequest();
	outputName = net.getOutputsInfo().begin()->first;
}

void UNetHistology::bgr2rgb(const Mat& src, Mat& dst) {
	cv::cvtColor(src, dst, COLOR_BGR2RGB);
}

void UNetHistology::normalize(const Mat& src, Mat& dst) {
	cout << "normalization" << endl;
	Scalar mean, stdDev;
	meanStdDev(src, mean, stdDev);

	for (size_t i = 0; i < src.rows; i++)
	{
		for (size_t j = 0; j < src.cols; j++)
		{
			for (size_t k = 0; k < 3; k++)
			{
				dst.push_back((src.at<Vec3b>(i, j)(k) - mean[k]) / stdDev[k]);
			}
		}
	}
	cout << "convertion" << endl;
	dst = dst.reshape(3, src.rows);
	dst.convertTo(dst, CV_32F);
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
	// Launch network
	req.Infer();
	// Copy output. 
	int32_t* output = req.GetBlob(outputName)->buffer();
	int size = req.GetBlob(outputName)->size();

	std::cout << "size is " << size << std::endl;
	for (size_t i = 0; i < size; i++)
	{
		mask.push_back(float(output[i]));
	}
	std::cout << "width is " << image.cols << std::endl;

	std::cout << "height is " << image.rows << std::endl;
	mask = mask.reshape(1, 772);
	std::cout << "resizing "<< std::endl;
	resize(mask, mask, Size(image.cols, image.rows));
	mask.convertTo(mask, CV_8UC1);
}


int UNetHistology::countGlands(const cv::Mat& segm) {
	// noise removal
	Mat opening;

	morphologyEx(segm, opening, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);

	Mat dist_transform;
	distanceTransform(opening, dist_transform, cv::DIST_L2, 5);
	Mat sure_fg;
	double max, min;
	minMaxLoc(dist_transform, &min, &max);
	cv::threshold(dist_transform, sure_fg, 0.6*max, 255, 0);
	Mat markers;
	//Marker labelling
	sure_fg.convertTo(sure_fg, CV_8U);
	int res = connectedComponents(sure_fg, markers);
	
	return res;
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
