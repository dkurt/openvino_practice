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

    int col = readInt(ifs);
    int row = readInt(ifs);

    uchar val;
    for (int i = 0; i < numImages; ++i)
    {
        Mat img(row, col, CV_8UC1);
        for (int j = 0; j < row; ++j)
        {
            for (int k = 0; k < col; ++k)
            {
                val = 0;
                ifs.read((char*)&val, sizeof(val));
                img.at<uchar>(j, k) = val;
            }
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
    uchar val;
    for (int i = 0; i < numLabels; ++i)
    {
        val = 0;
        ifs.read((char*)&val, sizeof(val));
        labels.push_back(val);
    }
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
    for (int i = 0; i < images.size(); ++i)
        samples.push_back(images[i].reshape(1, 1));
    samples.convertTo(samples, CV_32FC1);
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
                        const std::vector<int>& labels) {
    Mat samples;
    prepareSamples(images, samples);
    Ptr<ml::KNearest> trained_model = ml::KNearest::create();
    trained_model->train(samples, ml::SampleTypes::ROW_SAMPLE, labels);
    return trained_model;
}

float validate(Ptr<ml::KNearest> model,
               const std::vector<cv::Mat>& images,
               const std::vector<int>& labels) {
    Mat samples;
    prepareSamples(images, samples);
    Mat results;
    model->predict(samples, results);
    int correct = 0;
    for (int i = 0; i < labels.size(); ++i) {
        if (results.at<float>(i, 0) == labels[i]) {
            correct++;
        }
    }
    return static_cast<float>(correct) / (labels.size());
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    Mat tmpImg;
    resize(image, tmpImg, Size(28, 28));

    cvtColor(tmpImg, tmpImg, COLOR_BGR2HSV);

    Mat channels[3];
    split(tmpImg, &channels[0]);

    std::vector<Mat> saturation;
    saturation.push_back(channels[1]);

    Mat Input;
    prepareSamples(saturation, Input);

    int result = model->predict(Input);
    return result;
}
