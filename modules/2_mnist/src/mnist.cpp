#include "mnist.hpp"
#include "mnist_reader.hpp"
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
    MnistImageReader imgDispatcher(filepath);

    printf("Images are loaded, %d\n", imgDispatcher.count());

    images = imgDispatcher.getAllImages();
}

void loadLabels(const std::string& filepath,
                std::vector<int>& labels) {
    MnistLabelReader lblDispatcher(filepath);

    printf("Labels are loaded, %d\n", lblDispatcher.count());

    labels = lblDispatcher.getAllLabels();
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
    // assume each image has the same constraints
    cv::Size imgConstraints = images[0].size();
    samples = cv::Mat(images.size(), imgConstraints.width * imgConstraints.height, CV_8U);

    for (int i = 0; i < images.size(); i++) {
        images[i].reshape(1, 1).copyTo(samples.row(i));
    }

    samples.convertTo(samples, CV_32F);
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
                        const std::vector<int>& labels) {
    Ptr<ml::KNearest> knn(ml::KNearest::create());
    knn->setDefaultK(3);

    cv::Mat samples;
    prepareSamples(images, samples);

    knn->train(samples, ml::SampleTypes::ROW_SAMPLE, labels);

    return knn;
}

float validate(Ptr<ml::KNearest> model,
               const std::vector<cv::Mat>& images,
               const std::vector<int>& labels) {
    int correctGuesses = 0;

    cv::Mat samples;
    prepareSamples(images, samples);

    cv::Mat resultMat;
    model->findNearest(samples, model->getDefaultK(), resultMat);

    for (int i = 0; i < images.size(); i++) {
        correctGuesses += resultMat.at<int>(0, i) == labels[i];
    }

    return (float)correctGuesses / images.size() * 100;
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    cv::Mat image_hsv;
    // resize
    cv::resize(image, image_hsv, cv::Size(28, 28));

    // bgr to hsv
    cv::cvtColor(image_hsv, image_hsv, COLOR_BGR2HSV);

    // split hsv, hsvComponents[1] holds the saturation component
    std::vector<cv::Mat> hsvComponents(3);
    cv::split(image_hsv, hsvComponents);

    // flatten & convert to unsigned byte
    cv::Size imageSize = hsvComponents[1].size();
    cv::Mat preparedImage(1, imageSize.width * imageSize.height, CV_8U);

    hsvComponents[1].reshape(1, 1).copyTo(preparedImage.row(0));

    preparedImage.convertTo(preparedImage, CV_32F);

    // make a prediction by the model
    cv::Mat resultMat;
    model->findNearest(preparedImage, model->getDefaultK(), resultMat);

    return resultMat.at<int>(0, 0);
}