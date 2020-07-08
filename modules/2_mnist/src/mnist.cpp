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
    int size = numRows * numCols;
    for (int i = 0; i < numImages; ++i)
    {
        Mat img(numRows, numCols, CV_32FC1);
        for (int r = 0; r < numRows; ++r)
        {
            for (int c = 0; c < numCols; ++c)
            {
                unsigned char temp = 0;
                ifs.read((char*)&temp, sizeof(temp));
                img.at<float>(r,c)= (double)temp;
            }
        }
        images.push_back(img);
        img.release();

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
    for (int i = 0; i < numLabels; i++) {
        unsigned char temp = 0;
        ifs.read((char*)&temp, sizeof(temp));
        labels.push_back(temp);
    }

    // TODO: follow "FILE FORMATS FOR THE MNIST DATABASE" specification
    // at http://yann.lecun.com/exdb/mnist/
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
    int numRows = images[0].rows;
    int numCols = images[0].cols;
    int s_Rows = images.size();
    int s_Cols = numRows * numCols;
    samples = Mat::zeros(s_Rows, s_Cols, CV_32FC1);
    for (int i = 0; i < s_Rows; ++i) {
        for (int r = 0; r < numRows; ++r) {
            for (int c = 0; c < numCols; ++c) {
                samples.at<float>(i, r * numCols + c) = (float)images[i].at<uint8_t>(r, c);
            }
        }
    }
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
                        const std::vector<int>& labels) {
    cv::Ptr<cv::ml::KNearest> model = cv::ml::KNearest::create();
    cv::Ptr<cv::ml::TrainData> trainingData;
    Mat samples;
    prepareSamples(images, samples);
    trainingData = cv::ml::TrainData::create(samples, ml::SampleTypes::ROW_SAMPLE, labels);
    model->train(trainingData);
    return model;
}

float validate(Ptr<ml::KNearest> model,
               const std::vector<cv::Mat>& images,
               const std::vector<int>& labels) {
    Mat samples;
    prepareSamples(images, samples);
    Mat res;
    model->predict(samples, res);
    int k;
    for (int i = 0; i < labels.size(); ++i) {
        if (labels[i] == (float)res.at<int>(i,0)) {
            ++k;
        }
    }
    return (float(k) / float(labels.size()));
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    // TODO: resize image to 28x28 (cv::resize)
    Mat newImg;
    resize(image, newImg, Size(28, 28));

    // TODO: convert image from BGR to HSV (cv::cvtColor)
    Mat cvtImg;
    cvtColor(newImg, cvtImg, COLOR_BGR2HSV);

    // TODO: get Saturate component (cv::split)
    Mat channel[3];
    split(cvtImg, &channel[0]);
    std::vector<Mat>splitImg;
    splitImg.push_back(channel[1]);

    // TODO: prepare input - single row FP32 Mat
    Mat inputImg;
    prepareSamples(splitImg, inputImg);

    // TODO: make a prediction by the model
    int result = model->predict(inputImg);
    return result;
}
