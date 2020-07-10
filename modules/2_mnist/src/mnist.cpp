#include "mnist.hpp"
#include <fstream>

using namespace cv;

inline int readInt(std::ifstream& ifs) {
    int val;
    ifs.read((char*)&val, 4);
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
    int x = readInt(ifs);
    int y = readInt(ifs);

    for (int i = 0; i < numImages; ++i) {
        Mat image(x, y, CV_8UC1);
        for (int j = 0; j < x; ++j) {
            for (int k = 0; k < y; ++k) {
                uchar value;
                ifs.read((char*)&value, sizeof(value));
                image.at<uchar>(j, k) = value;
            }
        }
        images.push_back(image);
    }
}

void loadLabels(const std::string& filepath,
    std::vector<int>& labels) {
    std::ifstream ifs(filepath.c_str(), std::ios::binary);
    CV_CheckEQ(ifs.is_open(), true, filepath.c_str());

    int magicNum = readInt(ifs);
    CV_CheckEQ(magicNum, 2049, "");

    int numLabels = readInt(ifs);

    for (int i = 0; i < numLabels; ++i) {
        uchar value;
        ifs.read((char*)&value, sizeof(value));
        labels.push_back(value);
    }
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
    samples = Mat::zeros(images.size(), images[0].rows * images[0].cols, CV_32FC1);
    for (int i = 0; i < images.size(); ++i) {
        for (int j = 0; j < images[0].rows; ++j) {
            for (int k = 0; k < images[0].cols; ++k) {
                samples.at<double>(i, j * images[0].cols + k) = (double)images[i].at<uint8_t>(j, k);
            }
        }
    }
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images, const std::vector<int>& labels) {
    auto newModel = cv::ml::KNearest::create();
    Mat temp;
    prepareSamples(images, temp);
    newModel->train(temp, ml::ROW_SAMPLE, labels);
    return newModel;
    CV_Error(Error::StsNotImplemented, "validate");
}

float validate(Ptr<ml::KNearest> model, const std::vector<cv::Mat>& images, const std::vector<int>& labels) {
    Mat temp;
    prepareSamples(images, temp);
    Mat test;
    model->predict(temp, test);
    std::vector<int> copy(test.clone());
    int total = 0;
    for (int k = 0; k < labels.size(); ++k) {
        if (labels[k] == copy[k]) total++;
    }
    float var = (float)total / (float)labels.size();
    return var;
    CV_Error(Error::StsNotImplemented, "validate");
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    Mat temp;
    cv::resize(image, temp, Size(28, 28));
    cv::cvtColor(temp, temp, COLOR_BGR2HSV);
    Mat chanel[3];
    cv::split(temp, &chanel[0]);
    std::vector<Mat> img;
    img.push_back(chanel[1]);
    prepareSamples(img, temp);
    return (int)model->predict(temp);
    CV_Error(Error::StsNotImplemented, "predict");
}