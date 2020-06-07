#include <fstream>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "detector.hpp"

using namespace cv;

// Simple NMS test
TEST(detection, nms) {
    std::vector<Rect> boxes = {Rect(24, 17, 75, 129), Rect(38, 38, 88, 134), Rect(30, 88, 150, 105)};
    std::vector<float> probs = {0.81f, 0.89f, 0.92f};
    std::vector<unsigned> indices;

    // Replace #if 0 to #if 1 for debug visualization.
#if 0
    Mat img(300, 300, CV_8UC1, Scalar(0));
    for (int i = 0; i < boxes.size(); ++i) {
        rectangle(img, boxes[i], 255);
        putText(img, format("%.2f", probs[i]), Point(boxes[i].x, boxes[i].y - 2),
                FONT_HERSHEY_SIMPLEX, 0.5, 255);
    }
    imshow("nms", img);
    waitKey();
#endif
    nms(boxes, probs, 0.4, indices);

    ASSERT_EQ(indices.size(), 2);
    ASSERT_EQ(indices[0], 2);
    ASSERT_EQ(indices[1], 1);
}
