#include "opencv_coins.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>

using namespace cv;

unsigned countCoins(const Mat& img) {
    Mat gray, thresh;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
    // Remove unfilled black holes by morphological closing
    morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);

    // TODO: implement an algorithm from https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
    dilate(thresh, thresh, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);
    distanceTransform(thresh, thresh, DIST_L2, 5);
    double dist_transform, minVal;
    Point min_Loc, max_Loc;
    minMaxLoc(thresh, &minVal, &dist_transform, &min_Loc, &max_Loc);
    threshold(thresh, thresh, dist_transform*0.7, 255, THRESH_BINARY);
    thresh.convertTo(thresh, CV_8U, 1, 0);
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    int ncomp = contours.size();
    int sum = 0;
    double radius;
    for (int i = 0; i < ncomp; ++i) {
        radius = sqrt(contourArea(contours[i]) / 3.14);
        if (radius > 11) {
            sum + +2;
        }
        else {
            sum += 1;
        }
    }
    return sum;
}