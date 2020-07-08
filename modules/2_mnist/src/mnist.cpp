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

    for (int i = 0; i < numImages; ++i)
    {
        Mat image(numRows, numCols, CV_8UC1);
        for (int j = 0; j < numRows; ++j)
        {
            for (int k = 0; k < numCols; ++k)
            {
                unsigned char val = 0;
                ifs.read((char*)&val, sizeof(val));
                image.at<unsigned char>(j, k) = val;
            }
        }
        images.push_back(image);
        image.release();
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
        unsigned char val = 0;
        ifs.read((char*)&val, sizeof(val));
        labels.push_back(val);
    }

    // TODO: follow "FILE FORMATS FOR THE MNIST DATABASE" specification
    // at http://yann.lecun.com/exdb/mnist/
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
    Ptr<ml::KNearest> knn = ml::KNearest::create();
    Ptr<ml::TrainData> trainingData;

    Mat samples;
    prepareSamples(images, samples);
    trainingData = ml::TrainData::create(samples, ml::SampleTypes::ROW_SAMPLE, labels);
    knn->train(trainingData);
    return knn;
}

float validate(Ptr<ml::KNearest> model,
               const std::vector<cv::Mat>& images,
               const std::vector<int>& labels) {
    Mat samples;
    prepareSamples(images, samples);

    Mat resultsMat;
    model->predict(samples, resultsMat);
    std::vector<int> resultsPredicting = resultsMat.clone();
    int size = resultsMat.rows;

    int positiveAttempts = 0;
    for (int i = 0; i < labels.size(); i++) {
        if (resultsPredicting[i] == labels[i]) {
            positiveAttempts++;
        }
    }

    double positivePercent = (double)positiveAttempts / (double)images.size();
    return positivePercent;
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    // TODO: resize image to 28x28 (cv::resize)
    resize(image, image, Size(28, 28));

    // TODO: convert image from BGR to HSV (cv::cvtColor)
    cvtColor(image, image, COLOR_BGR2HSV);

    // TODO: get Saturate component (cv::split)
    std::vector<Mat> channels;
    channels.resize(image.channels());
    split(image, &channels[0]);
    Mat saturation = channels[1];

    // TODO: prepare input - single row FP32 Mat
    Mat inputImage(saturation, 1, CV_32FC1);

    std::vector<int> results;
    // TODO: make a prediction by the model
    model->findNearest(saturation, model->getDefaultK(), results);

    for (auto result: results)
        std::cout << result;

}
