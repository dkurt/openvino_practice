#include <gtest/gtest.h>

#include "opencv_coins.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv;
using namespace cv::utils::fs;

TEST(opencv, coins) {
    Mat img = imread(join(DATA_FOLDER, "coins.jpg"));
    unsigned sum = countCoins(img);
    ASSERT_EQ(sum, 22);
}
