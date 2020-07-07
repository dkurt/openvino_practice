#include "opencv_coins.hpp"
#include <string>
#include <iostream>
int dist_transform;
using namespace cv;
Mat img = imread("C:\\Users\\k_kab\\Picture\\coins.jpg");

unsigned countCoins(const Mat& img) {
    Mat gray, thresh, sure_bg, sure_fg, dist_transform, markers;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
    // Remove unfilled black holes by morphological closing
    morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 2);
    // noise removal
    // sure background area
    dilate(thresh, sure_bg, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);
    // Finding sure foreground area
    distanceTransform(sure_bg, dist_transform, cv:: DIST_L2, 5);
    double maxVal;
    minMaxLoc(dist_transform, 0, &maxVal);
    threshold(dist_transform, sure_fg, 0.7 * maxVal, 255, 0);
    Mat errorImage;
    sure_fg.convertTo(sure_fg, CV_8U);
    std::vector<std::vector<Point>> contours;
    findContours(sure_fg, contours, RETR_LIST, CHAIN_APPROX_NONE);
    int sum = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        double r = sqrt(contourArea(contours[i]) / CV_PI);
        if (r > 11)
            sum += 2;
        else
            sum++;
    }
    return sum;
}

