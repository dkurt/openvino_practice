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

void loadImages(const std::string& filepath, std::vector<Mat>& images) {
    std::ifstream ifs(filepath.c_str(), std::ios::binary);
    CV_CheckEQ(ifs.is_open(), true, filepath.c_str());

    int magicNum = readInt(ifs);
    CV_CheckEQ(magicNum, 2051, "");

    int numImages = readInt(ifs);
    int rows = readInt(ifs);
    int cols = readInt(ifs);

    for (int i = 0; i < numImages; i++) {
        Mat image(rows, cols, CV_8UC1);
        for (int j = 0; j < rows; j++) {
            for (int q = 0; q < cols; q++) {
                unsigned char val = 0;
                ifs.read((char*)&val, 1);
                image.at<unsigned char>(j, q) = val;
            }
        }
        images.push_back(image);
    }
}

void loadLabels(const std::string& filepath, std::vector<int>& labels) {
    std::ifstream ifs(filepath.c_str(), std::ios::binary);
    CV_CheckEQ(ifs.is_open(), true, filepath.c_str());

    int magicNum = readInt(ifs);
    CV_CheckEQ(magicNum, 2049, "");

    int numLabels = readInt(ifs);

    for (int i = 0; i < numLabels; i++) {
        unsigned char val = 0;
        ifs.read((char*)&val, 1);
        labels.push_back(val);
    }
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
    for (int i = 0; i < images.size(); i++) {
        samples.push_back(images[i].reshape(1, 1));
    }
    samples.convertTo(samples, CV_32FC1);
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images, const std::vector<int>& labels) {
    Mat samples;
    prepareSamples(images, samples);
    Ptr<ml::KNearest> tr = ml::KNearest::create();
    tr->train(samples, ml::ROW_SAMPLE, labels);
    return tr;
}

float validate(Ptr<ml::KNearest> model, const std::vector<cv::Mat>& images, const std::vector<int>& labels) {
    Mat samples,results;
    prepareSamples(images, samples);
    model->predict(samples, results);
    int tmp = 0;
    for (int i = 0; i < labels.size(); i++) {
        if (results.at<float>(i, 0) == labels[i]) {
            tmp++;
        }
    }
    return static_cast<float>(tmp) / (labels.size());
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    // TODO: resize image to 28x28 (cv::resize)
    Mat tmp_Img;
    resize(image, tmp_Img, Size(28, 28));

    // TODO: convert image from BGR to HSV (cv::cvtColor)
    Mat image_HSV;
    cvtColor(tmp_Img, image_HSV, COLOR_BGR2HSV);
    // TODO: get Saturate component (cv::split)
    std::vector<Mat> channels;
    split(image_HSV, channels);
    std::vector<Mat> saturation;
    saturation.push_back(channels[1]);

    // TODO: prepare input - single row FP32 Mat
    Mat Input;
    prepareSamples(saturation, Input);

    // TODO: make a prediction by the model
    int result = model->predict(Input);
    return result;
}
