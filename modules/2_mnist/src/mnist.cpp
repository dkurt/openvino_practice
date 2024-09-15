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
    int numrows = readInt(ifs);
    int numcolums = readInt(ifs);
    for (int i = 0; i < numImages; i++)
    {
        Mat image(numrows, numcolums, CV_8UC1);
        for (int j = 0; j < numrows; j++) {
            for (int k = 0; k < numcolums; k++) {
                unsigned char value = 0;
                ifs.read((char*)&value, 1);
                image.at<unsigned char>(j, k) = value;
            }
        }
        images.push_back(image);
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
    for (int i = 0; i < numLabels; i++)
    {
        unsigned char value = 0;
        ifs.read((char*)&value, 1);
        labels.push_back(value);
    }
    // TODO: follow "FILE FORMATS FOR THE MNIST DATABASE" specification
    // at http://yann.lecun.com/exdb/mnist/
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
    int size = int(images.size());
    int colums = images[0].rows*images[0].cols;
    samples=Mat::zeros(size, colums, CV_32FC1);
    for (int i = 0; i < size; i++){
        for (int j = 0; j < images[i].rows; j++) {
            for (int k = 0; k < images[i].cols; k++) {
                samples.at<float>(i, j*images[i].cols + k) = static_cast<float>(images[i].at<unsigned char>(j, k));
            }
        }
    }
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
                        const std::vector<int>& labels) {

    Ptr<ml::KNearest>kNear = ml::KNearest::create();
    Ptr<ml::TrainData>traindata;
    Mat samples;
    prepareSamples(images, samples);
    traindata = ml::TrainData::create(samples, ml::SampleTypes::ROW_SAMPLE, labels);
    kNear->train(traindata);
    return kNear;
}

float validate(Ptr<ml::KNearest> model,
               const std::vector<cv::Mat>& images,
               const std::vector<int>& labels) {
    Mat samples;
    prepareSamples(images, samples);

    Mat results;
    model->predict(samples, results);
    int accept=0;
    for (int i = 0; i < labels.size(); i++) {
        if (results.at<float>(i,0) == (float)labels[i])
            accept++;
    }
    float valid = (float)accept / (float)images.size();
    return valid;
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    // TODO: resize image to 28x28 (cv::resize)
    Mat newresize;
    resize(image, newresize, Size(28, 28));
    // TODO: convert image from BGR to HSV (cv::cvtColor)
    Mat hsv;
    cvtColor(newresize, hsv, COLOR_BGR2HSV);
    // TODO: get Saturate component (cv::split)
    Mat channels[3];
    split(hsv, &channels[0]);

    std::vector<Mat> saturate;
    saturate.push_back(channels[1]);
    // TODO: prepare input - single row FP32 Mat
    Mat put;
    prepareSamples(saturate, put);
    // TODO: make a prediction by the model
    int result;
    result = model->predict(put);
    return result;
}
