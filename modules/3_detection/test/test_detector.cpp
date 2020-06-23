#include <fstream>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "detector.hpp"

using namespace cv;
using namespace cv::utils::fs;

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

TEST(detection, faces) {
    const float nmsThreshold = 0.45f;
    const float probThreshold = 0.3f;

    Mat img = imread(join(DATA_FOLDER, "conference.png"));
    std::vector<Rect> refBoxes = {
        Rect(276, 267, 24, 31), Rect(529, 244, 23, 31), Rect(127, 268, 22, 29),
        Rect(53, 263, 23, 29),  Rect(600, 248, 23, 30), Rect(453, 242, 22, 31),
        Rect(428, 196, 22, 30), Rect(198, 264, 23, 29), Rect(570, 267, 22, 29),
        Rect(605, 193, 21, 29), Rect(94, 250, 22, 28),  Rect(486, 197, 20, 27),
        Rect(432, 263, 23, 30), Rect(370, 196, 20, 28), Rect(534, 194, 20, 26),
        Rect(328, 236, 20, 29), Rect(627, 276, 22, 28), Rect(108, 173, 21, 31),
        Rect(368, 261, 24, 31), Rect(571, 173, 20, 27), Rect(166, 241, 22, 30),
        Rect(252, 202, 21, 29), Rect(243, 240, 23, 30), Rect(387, 238, 21, 30),
        Rect(195, 205, 21, 30), Rect(162, 193, 17, 25), Rect(492, 262, 22, 29),
        Rect(319, 210, 19, 27), Rect(31, 238, 23, 30)
    };
    std::vector<float> refProbs = {
        0.9884441494941711, 0.9876197576522827, 0.9838215112686157,
        0.9810286164283752, 0.9808980822563171, 0.9801519513130188,
        0.9767189025878906, 0.9733858704566956, 0.9640806317329407,
        0.9604413509368896, 0.9576659798622131, 0.9413405060768127,
        0.9145927429199219, 0.9032665491104126, 0.8889322280883789,
        0.8799750208854675, 0.8717259168624878, 0.8642042279243469,
        0.8107402324676514, 0.8074663877487183, 0.7997952699661255,
        0.7943004965782166, 0.7681276202201843, 0.7119753956794739,
        0.6937134861946106, 0.6185194849967957, 0.5806660652160645,
        0.4284421503543854, 0.3059006929397583
    };

    Detector model;
    std::vector<Rect> boxes;
    std::vector<float> probs;
    std::vector<unsigned> classes;
    model.detect(img, boxes, probs, classes,nmsThreshold, probThreshold);

    // Replace #if 0 to #if 1 for debug visualization.
#if 0
	for (int i = 0; i < boxes.size(); ++i) {
		rectangle(img, boxes[i], Scalar(0, 0, 255));
		rectangle(img, refBoxes[i], Scalar(0, 255, 0));
		putText(img, format("%.2f", probs[i]), Point(boxes[i].x, boxes[i].y - 2),
			FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
	}
    imshow("Detection", img);
    waitKey();

#endif

    ASSERT_GE(boxes.size(), refBoxes.size());
    ASSERT_GE(probs.size(), refProbs.size());
    ASSERT_GE(classes.size(), refBoxes.size());

    int i;
    for (i = 0; i < boxes.size(); ++i) {
        if (probs[i] < probThreshold)
            break;

		ASSERT_EQ(classes[i], 1);
		ASSERT_EQ(boxes[i], refBoxes[i]);
		std::cout << "i " << i << std::endl;
		ASSERT_LE(fabs(probs[i] - refProbs[i]), 1e-5f);
		
    }
    ASSERT_EQ(i, boxes.size());
}
