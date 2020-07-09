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
    int numCols = readInt(ifs);
    images.resize(numImages);
    
    for (int i = 0; i < numImages; i++)
    {
        Mat image= Mat::zeros(numRows, numCols, CV_8UC1);
        for (int j = 0; j < numRows;j++) {
            for (int k = 0; k < numCols; k++)
            {
                uint8_t tmp;
                ifs.read((char*)&tmp, 1);
                image.at<uint8_t>(j,k) += tmp;
            }
        }
        images[i]=image;
    }


    // TODO: follow "FILE FORMATS FOR THE MNIST DATABASE" specification
    // at http://yann.lecun.com/exdb/mnist/
}

void loadLabels(const std::string& filepath,
                std::vector<int>& labels) {
    std::ifstream ifs(filepath.c_str(), std::ios::binary);
    CV_CheckEQ(ifs.is_open(), true, filepath.c_str());

    int magicNum = readInt(ifs);
    CV_CheckEQ(magicNum, 2049, "");

    int numLabels = readInt(ifs);
    
    labels.resize(numLabels);
    for (int i = 0; i < numLabels; i++)
    {
        uint8_t tmp;
        ifs.read((char*)&tmp, 1);
        labels[i] = tmp;
    }

    // TODO: follow "FILE FORMATS FOR THE MNIST DATABASE" specification
    // at http://yann.lecun.com/exdb/mnist/
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
    //CV_Error(Error::StsNotImplemented, "prepareSamples");
    for (int i = 0; i < images.size(); i++)
    {
        Mat row = images[i].reshape(1, 1);
        samples.push_back(row);
    }

    samples.convertTo(samples, CV_32FC1);


}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
                        const std::vector<int>& labels) {
    //CV_Error(Error::StsNotImplemented, "train");
    Ptr<ml::KNearest> model = ml::KNearest::create();
    Mat samples;
    prepareSamples(images, samples);
    Ptr<ml::TrainData>& data = ml::TrainData::create(samples, ml::ROW_SAMPLE, labels);
    model->train(data);    
    return model;
}

float validate(Ptr<ml::KNearest> model,
               const std::vector<cv::Mat>& images,
               const std::vector<int>& labels) {
    //CV_Error(Error::StsNotImplemented, "validate");
    Mat samples, result;
    prepareSamples(images, samples);
    model->findNearest(samples, 5, result );
    int numLabels = labels.size();
    int count = 0;
    for (int i = 0; i < numLabels; i++)
    {
        if (labels[i] == result.at<float>(i)) count++;
    }

    return (float)count / numLabels;
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    // TODO: resize image to 28x28 (cv::resize)
    
    Mat img;
    resize(image, img, Size(28,28));
    
    // TODO: convert image from BGR to HSV (cv::cvtColor)

    cvtColor(img, img, COLOR_BGR2HSV);

    // TODO: get Saturate component (cv::split)
    Mat *tmp;
    tmp = new Mat[3];
    std::vector<Mat> channels;
    split(img, &tmp[0]);
    channels.push_back(tmp[1]);
    
    // TODO: prepare input - single row FP32 Mat
    Mat sample;
    
    prepareSamples(channels, sample);

    // TODO: make a prediction by the model
    return model->predict(sample)+0.5;
}
