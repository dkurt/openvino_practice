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
    CV_Error(Error::StsNotImplemented, "prepareSamples");
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
                        const std::vector<int>& labels) {
    CV_Error(Error::StsNotImplemented, "train");
}

float validate(Ptr<ml::KNearest> model,
               const std::vector<cv::Mat>& images,
               const std::vector<int>& labels) {
    CV_Error(Error::StsNotImplemented, "validate");
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    // TODO: resize image to 28x28 (cv::resize)

    // TODO: convert image from BGR to HSV (cv::cvtColor)

    // TODO: get Saturate component (cv::split)

    // TODO: prepare input - single row FP32 Mat

    // TODO: make a prediction by the model

    CV_Error(Error::StsNotImplemented, "predict");
}
