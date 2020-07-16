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

    // Initialize runnable object on CPU device
    ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");

    // Create a single processing thread
    req = execNet.CreateInferRequest();
	outputName = net.getOutputsInfo().begin()->first;
}

void UNetHistology::bgr2rgb(const Mat& src, Mat& dst) {
	cvtColor(src, dst, COLOR_BGR2RGB);
}


void UNetHistology::normalize(const Mat& src, Mat& dst) {
   Scalar mean;
   Scalar st;
   meanStdDev(src, mean, st);
   dst = src.clone();
   dst.convertTo(dst, CV_32F);
   dst = (dst - mean) / st;
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

	// Pass blob as network's input. "data" is a name of input from .xml file
	req.SetBlob("worker_0/validation/IteratorGetNext", inputBlob);

	// Launch network
	req.Infer();

	// Copy output. "prob" is a name of output from .xml file
	int* output = req.GetBlob(outputName)->buffer();
	mask = Mat(req.GetBlob(outputName)->size(), 1, CV_32SC1, output);
	mask = mask.reshape(1, 772);	
	mask.convertTo(mask, CV_32F);
	resize(mask, mask, Size(image.cols, image.rows));
	mask.convertTo(mask, CV_8UC1);
}

int UNetHistology::countGlands(const cv::Mat& segm) {
	Mat thresh, bg, fg;
	morphologyEx(segm, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);
	distanceTransform(thresh, thresh, DIST_L2, 5);
	double min, max;
	minMaxLoc(thresh, &min, &max);
	threshold(thresh, fg, max * 0.6, 255, 0);
	fg.convertTo(fg, CV_8U);
	Mat tmp;
	int count = connectedComponents(fg, tmp);
	return count;
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
