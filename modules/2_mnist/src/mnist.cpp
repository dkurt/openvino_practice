#include "mnist.hpp"
#include <fstream>
#include <iostream>

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
		Mat newImage(numRows,numColumns,CV_8UC1);

		for (int j = 0; j < numRows; j++)
		{
			for (int k = 0; k < numColumns; k++)
			{
				char pixel;
				ifs.read((char *)&pixel, 1);			
				newImage.at<char>(j, k) = pixel;	
			}
		}	
		images.push_back(newImage);
	}  
}

void loadLabels(const std::string& filepath,
                std::vector<int>& labels) {
	std::ifstream ifs(filepath.c_str(), std::ios::binary);
	CV_CheckEQ(ifs.is_open(), true, filepath.c_str());

	int magicNum = readInt(ifs);
	CV_CheckEQ(magicNum, 2049, "");

	int numLabels = readInt(ifs);
	for (int i = 0; i < numLabels; i++)
	{
		char newLabel;
		ifs.read((char *)&newLabel, 1);
		labels.push_back((int)newLabel);
	}
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
	Mat Samples(0, images[0].cols*images[0].rows, CV_32FC1);

	for (int i = 0; i < images.size(); i++)
	{
		Mat vec = images[i].reshape(1, 1);
		vec.convertTo(vec, CV_32FC1);
		Samples.push_back(vec);
	}
	samples = Samples;
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
                        const std::vector<int>& labels) {
	Ptr<ml::KNearest> p = ml::KNearest::create();
	Mat trainingSamples;

	prepareSamples(images, trainingSamples);

	p->train(trainingSamples, ml::ROW_SAMPLE, labels);

	return p;
}

float validate(Ptr<ml::KNearest> model,
               const std::vector<cv::Mat>& images,
               const std::vector<int>& labels) {
	float acc;
	Mat Samples, results;
	int posRes = 0;

	prepareSamples(images, Samples);

	model->findNearest(Samples, 3, results);

	for (int i = 0; i < labels.size(); i++)
	{
		if (labels[i] == (int)results.at<float>(i, 0))
		{
			posRes++;
		}
	}

	acc = (float)posRes / labels.size();

	return acc;
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
	Mat img1;
	resize(image, img1, Size(28, 28));

	Mat img2;
	cvtColor(img1, img2, COLOR_BGR2HSV);

	std::vector<Mat> channels;
	split(img2, channels);

	std::vector<Mat> satur;
	satur.push_back(channels[1]);

	Mat sample;
	prepareSamples(satur, sample);

	int res = (int)model->predict(sample);

	return res;
}
