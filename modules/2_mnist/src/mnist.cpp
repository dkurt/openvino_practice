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
    int numRows = readInt(ifs);
    int numCols = readInt(ifs);

    for (int i = 0; i < numImages; i++)
    {
        Mat img(numRows, numCols, CV_8UC1);
        for (int j = 0; j < numRows; j++)
        {
            for (int k = 0; k < numCols; k++)
            {
                uchar dest = 0;
                ifs.read((char*)&dest, sizeof(dest));
                img.at<char>(j, k) = dest;
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

    for (int i = 0; i < numLabels; i++)
    {
        uchar val = 0;
        ifs.read((char*)&val, sizeof(val));
        labels.push_back(val);
    }
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
    for (int i = 0; i < images.size(); i++)
    {
        Mat row = images[i].reshape(1, 1);
        samples.push_back(row);
    }
    samples.convertTo(samples, CV_32FC1);
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
    const std::vector<int>& labels) {
    Ptr<ml::KNearest> model = ml::KNearest::create();
    Mat s;
    prepareSamples(images, s);
    model->train(s, ml::SampleTypes::ROW_SAMPLE, labels);
    return model;
}

float validate(Ptr<ml::KNearest> model,
    const std::vector<cv::Mat>& images,
    const std::vector<int>& labels) {
    Mat s;
    prepareSamples(images, s);
    Mat result;
    model->findNearest(s, 5, result);
    int counter = 0;
    for (int i = 0; i < labels.size(); i++)
    {
        if (labels[i] == result.at<float>(i));
        counter++;
    }
    return (float)counter / (float)labels.size();
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    Mat img;
    resize(image, img, Size(28, 28));

    Mat HSVcolor;
    cvtColor(img, HSVcolor, COLOR_BGR2HSV);

    std::vector<Mat> channels;
    split(HSVcolor, channels);

    std::vector<Mat> s;
    s.push_back(channels[1]);
    Mat input;
    prepareSamples(s, input);

    int result = model->predict(input);
    return result;
}
