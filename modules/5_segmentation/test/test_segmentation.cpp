#include <fstream>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "segmentation.hpp"

using namespace cv;
using namespace cv::utils::fs;

TEST(segmentation, Dice) {
    Rect a(10, 20, 100, 200);
    Rect b(30, 60, 70, 60);

    Mat aMask(300, 300, CV_8UC1, Scalar(0));
    Mat bMask(300, 300, CV_8UC1, Scalar(0));
    rectangle(aMask, a, /*color*/ 255, FILLED);
    // Method should be invariant to different non-zero values - the masks are binary
    rectangle(bMask, b, /*color*/ 127, FILLED);

    float ref = 2.0f * (a & b).area() / (a.area() + b.area());
    float score = Dice(aMask, bMask);
    ASSERT_EQ(score, ref);
}

TEST(ADAS, segment) {
    ADAS model;

    Mat img = imread(join(DATA_FOLDER, "car.jpg"));
    Mat mask;
    model.segment(img, mask);

    ASSERT_EQ(mask.rows, img.rows);
    ASSERT_EQ(mask.cols, img.cols);
    ASSERT_EQ(mask.channels(), 1);
    CV_CheckType(mask.type(), mask.type() == CV_8UC1, "Segmentation mask type");

    Mat carMask = mask == 13;

    Mat refMask = imread(join(DATA_FOLDER, "car_mask.png"), IMREAD_GRAYSCALE);
    ASSERT_GE(Dice(carMask, refMask), 0.96);
}
