#include <fstream>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "segmentation.hpp"

using namespace cv;
using namespace cv::utils::fs;

// Ignore this test - the method is implemented by default
TEST(UNetHistology, padMinimum_1ch) {
    uint8_t data[] = { 1, 2, 3, 4 };
    uint8_t ref[] = { 1,   1,   1,   2,   1,   1,
                      1,   1,   1,   2,   1,   1,
                      1,   1,   1,   2,   1,   1,
                      1,   1,   1,   2,   1,   1,
                      3,   3,   3,   4,   3,   3,
                      1,   1,   1,   2,   1,   1,
                      1,   1,   1,   2,   1,   1,
                      1,   1,   1,   2,   1,   1 };
    Mat src(2, 2, CV_8UC1, data);
    Mat dst;
    UNetHistology::padMinimum(src, 2, 3, dst);

    ASSERT_EQ(norm(Mat(8, 6, CV_8UC1, ref), dst, NORM_INF), 0);
}

// Ignore this test - the method is implemented by default
TEST(UNetHistology, padMinimum_3ch) {
    uint8_t data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Mat src(2, 2, CV_8UC3, data);
    Mat dst, ref;
    UNetHistology::padMinimum(src, 3, 2, dst);

    std::vector<Mat> srcChannels, refChannels(3);
    split(src, srcChannels);

    for (int i = 0; i < 3; ++i)
        UNetHistology::padMinimum(srcChannels[i], 3, 2, refChannels[i]);
    merge(refChannels, ref);

    ASSERT_EQ(norm(ref, dst, NORM_INF), 0);
}

TEST(UNetHistology, bgr2rgb) {
    Mat src(4, 5, CV_8UC3, Scalar(1, 2, 3));
    Mat dst;
    UNetHistology::bgr2rgb(src, dst);

    ASSERT_EQ(dst.size(), src.size());
    ASSERT_EQ(dst.channels(), 3);

    std::vector<Mat> channels;
    split(dst, channels);

    ASSERT_EQ(countNonZero(channels[0] != 3), 0);
    ASSERT_EQ(countNonZero(channels[1] != 2), 0);
    ASSERT_EQ(countNonZero(channels[2] != 1), 0);
}

TEST(UNetHistology, normalize) {
    uint8_t data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    Mat src(2, 2, CV_8UC3, data), dst;

    UNetHistology::normalize(src, dst);

    ASSERT_EQ(dst.size(), src.size());
    ASSERT_EQ(dst.channels(), 3);
    ASSERT_EQ(dst.depth(), CV_32F);

    Scalar mean, stdDev;
    meanStdDev(dst, mean, stdDev);

    ASSERT_EQ(mean, Scalar(0, 0, 0));
    ASSERT_LE(norm(stdDev, Scalar(1, 1, 1), NORM_INF), 1e-5);
}

TEST(UNetHistology, segment) {
    UNetHistology model;

    Mat img = imread(join(DATA_FOLDER, "colon_histology.jpg"));
    Mat mask;
    model.segment(img, mask);

    ASSERT_EQ(mask.rows, img.rows);
    ASSERT_EQ(mask.cols, img.cols);
    ASSERT_EQ(mask.channels(), 1);
	CV_CheckType(mask.type(), mask.type() == CV_8UC1, "Segmentation mask type");

    Mat ref = imread(join(DATA_FOLDER, "unet_histology_mask.png"), IMREAD_GRAYSCALE);
    ASSERT_GE(Dice(ref, mask), 0.95);
}

TEST(UNetHistology, countGlands) {
    UNetHistology model;

    Mat img = imread(join(DATA_FOLDER, "colon_histology.jpg"));
    Mat mask;
    model.segment(img, mask);

    int numGlands = UNetHistology::countGlands(mask);

    ASSERT_EQ(numGlands, 24);
}
