#include <gtest/gtest.h>

#include "opencv_imgproc.hpp"

TEST(opencv, bgr2gray) {
    Mat src(480, 640, CV_8UC3);
    randu(src, 0, 256);
    Mat gray = bgr2gray(src);
    EXPECT_EQ(gray.rows, src.rows);
    EXPECT_EQ(gray.cols, src.cols);
    EXPECT_EQ(gray.channels(), 1);
}
