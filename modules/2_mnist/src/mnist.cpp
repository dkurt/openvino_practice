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


    // TODO: follow "FILE FORMATS FOR THE MNIST DATABASE" specification
    // at http://yann.lecun.com/exdb/mnist/
    for (int i = 0; i < numImages; ++i) {
        Mat img(numRows, numCols, CV_8U);
        for (int r = 0; r < numRows; ++r) {
            for (int c = 0; c < numCols; ++c) {
                unsigned char pixel = readInt(ifs);
                img.at<unsigned char>(r, c) = pixel;
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
    labels.resize(numLabels);
    for (int i = 0; i < numLabels; ++i) {
        unsigned char label = readInt(ifs);
        labels[i] = label;
    }
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {

    
    for (int i = 0; i < images.size(); ++i) {
        Mat image_pixels = images[i].reshape(1, 1);
        samples.push_back(image_pixels);
    }

    CV_Error(Error::StsNotImplemented, "prepareSamples");
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
                        const std::vector<int>& labels) {

    Mat a;
    prepareSamples(images, a);




    CV_Error(Error::StsNotImplemented, "train");
}

float validate(Ptr<ml::KNearest> model,
               const std::vector<cv::Mat>& images,
               const std::vector<int>& labels) {
    CV_Error(Error::StsNotImplemented, "validate");
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    // TODO: resize image to 28x28 (cv::resize)
    Mat resized_image, converted_image;
    resize(image, resized_image, Size(28,28));

    // TODO: convert image from BGR to HSV (cv::cvtColor)
    cvtColor(resized_image, converted_image, COLOR_BGR2HSV);
    // TODO: get Saturate component (cv::split)

    // TODO: prepare input - single row FP32 Mat

    // TODO: make a prediction by the model

    CV_Error(Error::StsNotImplemented, "predict");
}
