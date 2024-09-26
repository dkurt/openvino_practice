#include "mnist.hpp"
#include <fstream>
using namespace cv;
inline int readInt(std::ifstream& ifs) {
	int val;
	ifs.read((char*)&val, 4);
	// Integers in file are high endian which requires swap
	std::swap(((char*)&val)[0], ((char*)&val)[3]);
	std::swap(((char*)&val)[1], ((char*)&val)[2]);
	return val;
}
void loadImages(const std::string& filepath,
	std::vector<Mat>& images) {
	std::ifstream ifs(filepath.c_str(), std::ios::binary);
	CV_CheckEQ(ifs.is_open(), true, filepath.c_str());
	int magicNum = readInt(ifs);
	CV_CheckEQ(magicNum, 2051, "");
	// TODO: follow "FILE FORMATS FOR THE MNIST DATABASE" specification
	// at http://yann.lecun.com/exdb/mnist/
	int numImages = readInt(ifs);
	int numRows = readInt(ifs);
	int numColumns = readInt(ifs);
	for (int i = 0; i < numImages; i++)
	{
		Mat tmp(numRows, numColumns, CV_8UC1);
		for (int j = 0; j < numRows; j++)
		{
			for (int f = 0; f < numColumns; f++)
			{
				unsigned char val = 0;
				ifs.read((char*)&val, 1);
				tmp.at<unsigned char>(j, f) = val;
			}
		}
		images.push_back(tmp);
	}
}
void loadLabels(const std::string& filepath,
	std::vector<int>& labels) {
	std::ifstream ifs(filepath.c_str(), std::ios::binary);
	CV_CheckEQ(ifs.is_open(), true, filepath.c_str());
	int magicNum = readInt(ifs);
	CV_CheckEQ(magicNum, 2049, "");
	// TODO: follow "FILE FORMATS FOR THE MNIST DATABASE" specification
	// at http://yann.lecun.com/exdb/mnist/
	int numLabels = readInt(ifs);
	for (int i = 0; i < numLabels; i++)
	{
		unsigned char val = 0;
		ifs.read((char*)&val, 1);
		labels.push_back(val);
	}
}
void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
	int i = images.size();
	Mat sample(i, 28 * 28, CV_32FC1);
	for (int j = 0; j < i; j++)
	{
		Mat image_row = images[j].clone().reshape(1, 1);
		Mat row_j = sample.row(j);
		image_row.convertTo(row_j, CV_32FC1);
	}
	samples = sample;
}
Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
	const std::vector<int>& labels) {
	Ptr <ml::KNearest> a = ml::KNearest::create();
	Mat sample;
	prepareSamples(images, sample);
	a->train(sample, ml::ROW_SAMPLE, labels);
	return a;
}
float validate(Ptr<ml::KNearest> model,
	const std::vector<cv::Mat>& images,
	const std::vector<int>& labels) {
	Mat samples, results;
	prepareSamples(images, samples);
	model->findNearest(samples, 2, results);
	int numLab = labels.size();
	int numSuccess = 0;
	for (int i = 0; i < numLab; i++)
		if ((int)results.at<float>(i, 0) == labels[i])
			numSuccess++;
	return (float)numSuccess / (float)numLab;
}
int predict(Ptr<ml::KNearest> model, const Mat& image) {
	// TODO: resize image to 28x28 (cv::resize)
	Mat image_resize_to_28;
	resize(image, image_resize_to_28, Size(28, 28));
	// TODO: convert image from BGR to HSV (cv::cvtColor)
	Mat image_HSV;
	cvtColor(image_resize_to_28, image_HSV, COLOR_BGR2HSV);
	// TODO: get Saturate component (cv::split)
	std::vector<Mat> channels;
	split(image_HSV, channels);
	// TODO: prepare input - single row FP32 Mat
	std::vector<Mat> sat;
	sat.push_back(channels[1]);
	Mat image_in;
	prepareSamples(sat, image_in);
	// TODO: make a prediction by the model
	Mat result;
	model->findNearest(image_in, 64, result);
	return (int)result.at<float>(0, 0);
}
