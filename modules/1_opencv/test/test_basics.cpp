#include <gtest/gtest.h>

#include "opencv_basics.hpp"

TEST(opencv, createMat) {
    Mat m = createMat(10, 11, 4, CV_8U);
    EXPECT_EQ(m.rows, 10);
    EXPECT_EQ(m.cols, 11);
    EXPECT_EQ(m.channels(), 4);
    EXPECT_EQ(m.depth(), CV_8U);

    Mat m2 = createMat(5, 4, 3, CV_32F);
    EXPECT_EQ(m2.rows, 5);
    EXPECT_EQ(m2.cols, 4);
    EXPECT_EQ(m2.channels(), 3);
    EXPECT_EQ(m2.depth(), CV_32F);
}
