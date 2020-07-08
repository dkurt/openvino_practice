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

    for (int i(0); i < numImages; i++) {
        Mat tmp(numRows, numColumns, CV_8UC1);

        for (int j(0); j < numRows; j++) {
            for (int k(0); k < numColumns; k++) {
                unsigned char val = 0;
                ifs.read((char*)&val, 1);
                tmp.at<unsigned char>(j, k) = val;
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

    int numLabels = readInt(ifs);

    for (int i(0); i < numLabels; i++) {
        unsigned char val = 0;
        ifs.read((char*)&val, 1);
        labels.push_back(val);
    }
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
    Mat sample(images.size(), 28 * 28, CV_32FC1);
    for (int i(0); i < images.size(); i++) {
        Mat imageRow = images[i].clone().reshape(1, 1);
        Mat row = sample.row(i);
        imageRow.convertTo(row, CV_32FC1);
    }
    samples = sample;
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
    const std::vector<int>& labels) {
    
    Ptr <ml::KNearest> ptr = ml::KNearest::create();
    Mat sample;
    prepareSamples(images, sample);
    ptr->train(sample, ml::ROW_SAMPLE, labels);
    return ptr;
}

float validate(Ptr<ml::KNearest> model,
    const std::vector<cv::Mat>& images,
    const std::vector<int>& labels) {
    
    Mat sampleMat, resultMat;
    prepareSamples(images, sampleMat);
    model->findNearest(sampleMat, 2, resultMat);
    int numLab = labels.size();
    int numSuccess = 0;
    for (int i = 0; i < numLab; i++)
        if ((int)resultMat.at<float>(i, 0) == labels[i])
            numSuccess++;
    return (float)numSuccess / (float)numLab;
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    Mat newImage;
    resize(image, newImage, Size(28, 28));
    Mat HSVImage;
    cvtColor(newImage, HSVImage, COLOR_BGR2HSV);
    
    Mat channels[3];
    split(HSVImage, &channels[0]);
    std::vector<Mat> saturationVector;
    saturationVector.push_back(channels[1]);
    Mat inputImage;
    prepareSamples(saturationVector, inputImage);
    
    int result;
    result = model->predict(inputImage);
    return result;
}