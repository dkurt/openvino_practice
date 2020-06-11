#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "mnist.hpp"

using namespace cv;

TEST(Model, knearest) {
    std::vector<Mat> trainImages, testImages;
    std::vector<int> trainLabels, testLabels;
    loadImages(utils::fs::join(DATA_FOLDER, "train-images.idx3-ubyte"), trainImages);
    loadLabels(utils::fs::join(DATA_FOLDER, "train-labels.idx1-ubyte"), trainLabels);
    loadImages(utils::fs::join(DATA_FOLDER, "t10k-images.idx3-ubyte"), testImages);
    loadLabels(utils::fs::join(DATA_FOLDER, "t10k-labels.idx1-ubyte"), testLabels);

    Ptr<ml::KNearest> model = train(trainImages, trainLabels);
    float accuracy = validate(model, testImages, testLabels);
    ASSERT_GE(accuracy, 0.96);
}

TEST(Model, ocr) {
    // Train model
    std::vector<Mat> trainImages;
    std::vector<int> trainLabels;
    loadImages(utils::fs::join(DATA_FOLDER, "train-images.idx3-ubyte"), trainImages);
    loadLabels(utils::fs::join(DATA_FOLDER, "train-labels.idx1-ubyte"), trainLabels);
    Ptr<ml::KNearest> model = train(trainImages, trainLabels);

    // Load image
    Mat image = imread(utils::fs::join(DATA_FOLDER, "counter.png"));

    std::vector<Rect> boxes = {Rect(737, 325, 56, 56), Rect(781, 323, 56, 56),
                               Rect(830, 320, 56, 56)};
    int ref[] = {6, 6, 7};
    for (int i = 0; i < boxes.size(); ++i) {
        Mat roi = image(boxes[i]);
        ASSERT_EQ(predict(model, roi), ref[i]);
    }
}
