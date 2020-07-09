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
    int numCols = readInt(ifs);

    images.resize(numImages);
    // TODO: follow "FILE FORMATS FOR THE MNIST DATABASE" specification
    // at http://yann.lecun.com/exdb/mnist/
    for (int i = 0; i < numImages; ++i) {
        Mat img(numRows, numCols, CV_8U);
        for (int r = 0; r < numRows; ++r) {
            for (int c = 0; c < numCols; ++c) {
                unsigned char pixel = 0;
                ifs.read((char*)&pixel, sizeof(pixel));
                img.at<unsigned char>(r, c) = pixel;
            }
        }
        images[i] = img;
    }
}

void loadLabels(const std::string& filepath,
                std::vector<int>& labels) {
    std::ifstream ifs(filepath.c_str(), std::ios::binary);
    CV_CheckEQ(ifs.is_open(), true, filepath.c_str());

    int magicNum = readInt(ifs);
    CV_CheckEQ(magicNum, 2049, "");

    int numLabels = readInt(ifs);
    labels.resize(numLabels, 0);
    for (int i = 0; i < numLabels; ++i) {
        unsigned char label = 0;
        ifs.read((char*)&label, sizeof(label));
        labels[i] = label;
    }
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {

    
    for (int i = 0; i < images.size(); ++i) {
        Mat image_pixels = images[i].reshape(1, 1);
        samples.push_back(image_pixels);
    }
    samples.convertTo(samples, CV_32FC1);
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
                        const std::vector<int>& labels) {

    Ptr<ml::KNearest> model = ml::KNearest::create();
    Ptr<ml::TrainData> training_data;
    Mat samples;

    prepareSamples(images, samples);

    training_data = ml::TrainData::create(samples, ml::SampleTypes::ROW_SAMPLE, labels);

    model->train(training_data);

    return model;
}

float validate(Ptr<ml::KNearest> model,
               const std::vector<cv::Mat>& images,
               const std::vector<int>& labels) {

    Mat samples, result;

    prepareSamples(images, samples);

    model->findNearest(samples, model->getDefaultK(), result);

    int _count = 0;
    for (int i = 0; i < labels.size(); ++i) {
        if (result.at<float>(i) == labels[i]) {
            _count++;
        }
    }
    return float(_count) / labels.size();
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    // TODO: resize image to 28x28 (cv::resize)
    Mat resized_image, converted_image, channels, sample;
    resize(image, resized_image, Size(28,28));

    // TODO: convert image from BGR to HSV (cv::cvtColor)
    cvtColor(resized_image, converted_image, COLOR_BGR2HSV);
    // TODO: get Saturate component (cv::split)
    split(converted_image, channels);

    // TODO: prepare input - single row FP32 Mat
    prepareSamples(image, sample);
    // TODO: make a prediction by the model
    int predict_value = model->predict(sample);
    return predict_value;
}
