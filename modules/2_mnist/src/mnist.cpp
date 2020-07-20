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
    images = std::vector<Mat>(numImages);
    for (int k = 0; k < numImages; k++) {
        Mat image = Mat::zeros(numRows, numCols, CV_8UC1);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                uint8_t temp = 0;
                ifs.read((char*)&temp, 1);
                image.at<uint8_t>(i, j) += temp;
            }
        }
        images[k] = image;
    }
}

void loadLabels(const std::string& filepath,
                std::vector<int>& labels) {
    std::ifstream ifs(filepath.c_str(), std::ios::binary);
    CV_CheckEQ(ifs.is_open(), true, filepath.c_str());

    int magicNum = readInt(ifs);
    CV_CheckEQ(magicNum, 2049, "");

    int numLabels = readInt(ifs);
    labels = std::vector<int>(numLabels);
    for (int i = 0; i < numLabels; i++) {
        uint8_t temp = 0;
        ifs.read((char*)&temp, 1);
        labels[i] = temp;
    }
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
    int numRows = images[0].rows;
    int numCols = images[0].cols;
    int sampleNumRows = images.size();
    int sampleNumCols = numRows * numCols;
    samples = Mat::zeros(sampleNumRows, sampleNumCols, CV_32FC1);
    for (int k = 0; k < sampleNumRows; k++) {
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                samples.at<float>(k, i * numCols + j) = (float)images[k].at<uint8_t>(i, j);
            }
        }
    }
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
                        const std::vector<int>& labels) {
    Ptr<ml::KNearest> model = cv::ml::KNearest::create();
    model->setDefaultK(5);

    // create traindata 
    Mat samples, response = Mat::zeros(labels.size(), 1, CV_32FC1);
    for (int i = 0; i < response.rows; i++) {
        response.at<float>(i, 0) = (float)labels[i];
    }
    prepareSamples(images, samples);
    const Ptr<ml::TrainData>& trainData = ml::TrainData::create(samples, ml::ROW_SAMPLE, response);

    model->train(trainData);
    return model;
}

float validate(Ptr<ml::KNearest> model,
               const std::vector<cv::Mat>& images,
               const std::vector<int>& labels) {
    Mat samples, response;
    prepareSamples(images, samples);
    model->findNearest(samples, 5, response);
    int numLabels = labels.size();
    int numBadResponses = 0;
    for (int i = 0; i < numLabels; i++) {
        if (labels[i] != (int)response.at<float>(i, 0)) {
            numBadResponses++;
        }
    }
    return 1.0f - (float)numBadResponses / (float)(numLabels);
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    Mat image28;
    resize(image, image28, Size(28, 28));

    Mat imageHSV;
    cvtColor(image28, imageHSV, COLOR_BGR2HSV);

    Mat* digit = new Mat[3];
    split(imageHSV, digit);

    Mat inputImage;
    std::vector<Mat> digits;
    digits.push_back(digit[1]);
    prepareSamples(digits, inputImage);

    Mat response;
    model->findNearest(inputImage, 7, response);

    return (int)response.at<float>(0,0);
}
