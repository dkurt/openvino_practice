#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "mnist.hpp"

using namespace cv;

TEST(DataLoader, TrainImages) {
    std::vector<Mat> images;
    loadImages(utils::fs::join(DATA_FOLDER, "train-images.idx3-ubyte"), images);

    ASSERT_EQ(images.size(), 60000);

    for (int i = 0; i < 2; ++i) {
        auto path = utils::fs::join(DATA_FOLDER, format("mnist_%d.png", i + 1));
        Mat ref = imread(path, IMREAD_GRAYSCALE);
        ASSERT_EQ(countNonZero(ref != images[i]), 0);
    }
}

TEST(DataLoader, TestImages) {
    std::vector<Mat> images;
    loadImages(utils::fs::join(DATA_FOLDER, "t10k-images.idx3-ubyte"), images);
    ASSERT_EQ(images.size(), 10000);
}

TEST(DataLoader, TrainLabels) {
    std::vector<int> labels;
    loadLabels(utils::fs::join(DATA_FOLDER, "train-labels.idx1-ubyte"), labels);
    ASSERT_EQ(labels.size(), 60000);
    ASSERT_EQ(labels[0], 5);
    ASSERT_EQ(labels[1], 0);
}

TEST(DataLoader, prepareSamples) {
    std::vector<Mat> images(3);
    images[0] = Mat(28, 28, CV_8UC1, Scalar(11));
    images[1] = Mat(28, 28, CV_8UC1, Scalar(22));
    images[2] = Mat(28, 28, CV_8UC1, Scalar(33));

    Mat samples;
    prepareSamples(images, samples);

    ASSERT_EQ(samples.rows, 3);
    ASSERT_EQ(samples.cols, 28*28);
    ASSERT_EQ(countNonZero(samples.row(0) != 11), 0);
    ASSERT_EQ(countNonZero(samples.row(1) != 22), 0);
    ASSERT_EQ(countNonZero(samples.row(2) != 33), 0);
}
