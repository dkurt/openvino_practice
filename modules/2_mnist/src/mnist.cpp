#include "mnist.hpp"
#include <fstream>

using namespace cv;
using namespace std;


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

    int rows = readInt(ifs);
    int ñols = readInt(ifs);

    for (int i = 0; i < numImages; ++i)
    {
        Mat image(rows, ñols, CV_8UC1);
        for (int j = 0; j < rows; ++j)
        {
            for (int k = 0; k < ñols; ++k)
            {
                char val = 0;
                ifs.read((char*)&val, sizeof(val));
                image.at<char>(j, k) = val;
            }
        }
        images.push_back(image);
        image.release();
    }
}

void loadLabels(const std::string& filepath, std::vector<int>& labels) {
    std::ifstream ifs(filepath.c_str(), std::ios::binary);
    CV_CheckEQ(ifs.is_open(), true, filepath.c_str());

    int magicNum = readInt(ifs);
    CV_CheckEQ(magicNum, 2049, "");

    int numLabels = readInt(ifs);

    // TODO: follow "FILE FORMATS FOR THE MNIST DATABASE" specification
    // at http://yann.lecun.com/exdb/mnist/
    for (int i = 0; i < numLabels; i++) {
        unsigned char val = 0;
        ifs.read((char*)&val, sizeof(val));
        labels.push_back(val);
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
    Mat tmp_image;
    resize(image, tmp_image, Size(28, 28));

    // TODO: convert image from BGR to HSV (cv::cvtColor)
    cvtColor(tmp_image, tmp_image, COLOR_BGR2HSV);

    // TODO: get Saturate component (cv::split)
    Mat channels[3];
    split(image, &channels[0]);
    vector<Mat> saturate_comp(1, channels[1]);

    // TODO: prepare input - single row FP32 Mat
    prepareSamples(saturate_comp, tmp_image);

    // TODO: make a prediction by the model
    int res = model->predict(tmp_image);
    return res;
}
