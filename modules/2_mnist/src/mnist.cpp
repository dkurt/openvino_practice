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
	int numRow = readInt(ifs);
	int numCol = readInt(ifs);

	//creat Mat image 
	for (int i = 0; i < numImages; i++)
	{
		Mat img(numRow, numCol, CV_8UC1);
		for (int k = 0; k < numRow; k++)
			for (int m = 0; m < numCol; m++)
			{
				uchar val;
				ifs.read((char*)&val, sizeof(val));
				img.at<uchar>(k, m) = val;
			}
		images.push_back(img);
	}
}

void loadLabels(const std::string& filepath,
	std::vector<int>& labels) {
	std::ifstream ifs(filepath.c_str(), std::ios::binary);
	CV_CheckEQ(ifs.is_open(), true, filepath.c_str());

	int magicNum = readInt(ifs);
	CV_CheckEQ(magicNum, 2049, "");

	int numLabels = readInt(ifs);

	//create lable
	for (int i = 0; i < numLabels; i++)
	{
		uchar val;
		ifs.read((char*)&val, sizeof(val));
		labels.push_back(val);
	}
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {

	samples = Mat::zeros(images.size(), images[0].rows*images[0].cols, CV_32FC1);   //float
	for (int k = 0; k < images.size(); k++)
		for (int i = 0; i < images[0].rows; i++)
			for (int j = 0; j < images[0].cols; j++)
				samples.at<float>(k, i*images[0].cols + j) = (float)images[k].at<uint8_t>(i, j);  //copy
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
	const std::vector<int>& labels) {
	//create empty model witñh we will return
	Ptr<ml::KNearest> model = cv::ml::KNearest::create();

	//recive mat of image
	Mat samples;
	prepareSamples(images, samples);

	//trainig model, Row_Sample
	model->train(samples, ml::ROW_SAMPLE, labels);
	return model;
}

float validate(Ptr<ml::KNearest> model,
	const std::vector<cv::Mat>& images,
	const std::vector<int>& labels) {
	Mat samples, prediction;
	prepareSamples(images, samples);

	model->predict(samples, prediction);
	std::vector<int> vpr(prediction.clone());

	int VarCount = labels.size();
	int RelCount = 0;

	for (int i = 0; i < vpr.size(); i++)
	{
		if (labels[i] == vpr[i]) RelCount++;
	}

	float validate;
	validate = (float)RelCount / (float)VarCount;
	std::cout << VarCount << std::endl;
	std::cout << RelCount << std::endl;
	std::cout << validate << std::endl;
	return validate;
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {

	Mat tmp;
	resize(image, tmp, Size(28, 28));

	cvtColor(tmp, tmp, COLOR_BGR2HSV);

	Mat hsv_planes[3];
	split(tmp, &hsv_planes[0]);     //hsv_planes[1]- S channel

	std::vector<Mat> s_planes;
	s_planes.push_back(hsv_planes[1]);
	prepareSamples(s_planes, tmp);

	int pr = model->predict(tmp);
	return pr;
}

