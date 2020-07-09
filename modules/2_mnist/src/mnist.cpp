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

	int numImages = readInt(ifs);
	int numRows = readInt(ifs);
	int numColumns = readInt(ifs);

	for (int i = 0; i < numImages; i++)
	{
		Mat t(numRows, numColumns, CV_8UC1);
		for (int j = 0; j < numRows; j++)
		{
			for (int k = 0; k < numColumns; k++)
			{
				char tmp;
				ifs.read((char*)&tmp, 1);
				t.at<char>(j, k) = tmp;

			}

		}
		images.push_back(t);
	}
}

void loadLabels(const std::string& filepath,
	std::vector<int>& labels) {
	std::ifstream ifs(filepath.c_str(), std::ios::binary);
	CV_CheckEQ(ifs.is_open(), true, filepath.c_str());

	int magicNum = readInt(ifs);
	CV_CheckEQ(magicNum, 2049, "");

	int numLabels = readInt(ifs);
	for (int k = 0; k < numLabels; k++)
	{
		char tmp;
		ifs.read((char*)&tmp, 1);
		labels.push_back(tmp);

	}
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
	int count = images.size();
	Mat samp(count, 28 * 28, CV_32FC1);
	for (int i = 0; i < count; i++)
	{
		Mat row_for_image = images[i].clone().reshape(1, 1);
		Mat row_i = samp.row(i);
		row_for_image.convertTo(row_i, CV_32FC1);

	}
	samples = samp;

}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
	const std::vector<int>& labels) {
	Ptr<ml::KNearest> ptr = ml::KNearest::create();
	Mat samp;
	prepareSamples(images, samp);
	ptr->train(samp, ml::ROW_SAMPLE, labels);
	
	return ptr;
}

float validate(Ptr<ml::KNearest> model,
	const std::vector<cv::Mat>& images,
	const std::vector<int>& labels) {
	Mat sample, res;
	prepareSamples(images, sample);
	model->findNearest(sample, 2, res);
	int count_labels = labels.size();
	int count_Success = 0;
	for (int i = 0; i < count_labels; i++)
		if ((int)res.at<float>(i, 0) == labels[i])
			count_Success++;
	return (float)count_Success / (float)count_labels;


}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
	// TODO: resize image to 28x28 (cv::resize)
	Mat image_resize;
	resize(image,image_resize , Size(28,28));
	// TODO: convert image from BGR to HSV (cv::cvtColor)
	Mat image_HSV;
	cvtColor(image_resize, image_HSV, COLOR_BGR2HSV);
	// TODO: get Saturate component (cv::split)
	std::vector<Mat> channel;
	split(image_HSV, channel);
	// TODO: prepare input - single row FP32 Mat
	std::vector<Mat> st;
	st.push_back(channel[1]);
	Mat im;
	prepareSamples(st, im);
	// TODO: make a prediction by the model
	Mat res;
	model->findNearest(im, 64, res);
	return (int)res.at<float>(0, 0);
}
