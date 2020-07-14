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

    for (int i = 0; i < numImages; i++) {
        Mat image(numRows, numCols, CV_8UC1);
        for (int j = 0; j < numRows; j++) {
            for (int k = 0; k < numCols; k++) {
                uchar val;
                ifs.read((char*)&val, sizeof(val));
                image.at<uchar>(j, k) = val;
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
    for (int k = 0; k < numLabels; k++) {
        uchar val;
        ifs.read((char*)&val, sizeof(val));
        labels.push_back(val);
    }

    // TODO: follow "FILE FORMATS FOR THE MNIST DATABASE" specification
    // at http://yann.lecun.com/exdb/mnist/
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
    int images_size = images.size();
    int images_rows = images[0].rows;
    int images_cols = images[0].cols;
    samples = Mat::zeros(images_size, images_rows * images_cols, CV_32FC1);
    for (int k = 0; k < images_size; k++) { //rows sample
        for (int i = 0; i < images_rows; i++) {
            for (int j = 0; j < images_cols; j++) {
                samples.at<float>(k, i * images_cols + j) = (float)images[k].at<uint8_t>(i, j);
            }
        }   
    }
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
                        const std::vector<int>& labels) {
    Ptr<ml::KNearest> modelKnn = ml::KNearest::create();
    Ptr<ml::TrainData> trainData;

    Mat samples;
    prepareSamples(images, samples);
    trainData = ml::TrainData::create(samples, ml::SampleTypes::ROW_SAMPLE, labels);
    modelKnn->train(trainData);
    return modelKnn;
}

float validate(Ptr<ml::KNearest> model, const std::vector<cv::Mat>& images, const std::vector<int>& labels) {
    Mat samples, result;
    prepareSamples(images, samples);

    model->predict(samples, result);
    std::vector<int> resultsModel = result.clone();

    int numMatch = 0;
    for (int i = 0; i < labels.size(); i++) {
        if (resultsModel[i] == labels[i]) {
            numMatch++;
        }
    }

    double positivePercent = (double)numMatch / (double)images.size();
    return positivePercent;
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    Mat tmpImage;
    resize(image, tmpImage, Size(28, 28)); // resize image to 28x28 (cv::resize)
    
    cvtColor(tmpImage, tmpImage, COLOR_BGR2HSV); // convert image from BGR to HSV (cv::cvtColor)
    
    Mat channels[3];
    split(tmpImage, &channels[0]); // get Saturate component (cv::split)

    std::vector<Mat> saturation;
    saturation.push_back(channels[1]); // take S-channel
    
    Mat Input;
    prepareSamples(saturation, Input);
    int result = model->predict(Input); // make a prediction by the model
    return result;
}
