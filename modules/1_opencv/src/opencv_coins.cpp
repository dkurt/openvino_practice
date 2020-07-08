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
    morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 2);
    dilate(thresh, sure_bg, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);
    distanceTransform(thresh, dist_transform, cv::DIST_L2, 5);
    distanceTransform(sure_bg, dist_transform, cv::DIST_L2, 5);
    double maxVal;
    Point maxLoc;
    minMaxLoc(dist_transform, 0, &maxVal, &maxLoc);
    minMaxLoc(dist_transform, 0, &maxVal);
    threshold(dist_transform, sure_fg, 0.7 * maxVal, 255, 0);
    //Mat errorImage;
    //subtract(sure_bg, sure_fg, errorImage);
    sure_fg.convertTo(sure_fg, CV_8U);
    std::vector<std::vector<Point>> contours;
    findContours(sure_fg, contours, RETR_LIST, CHAIN_APPROX_NONE);
    int sum = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        double r = sqrt(contourArea(contours[i]) / CV_PI);
        if (r > 13)
            sum += 2;
        else
            sum++;
    }
    return sum;
}