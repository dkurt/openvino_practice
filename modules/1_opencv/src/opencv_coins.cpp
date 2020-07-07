#include "opencv_coins.hpp"

using namespace cv;

unsigned countCoins(const Mat& img) {
    Mat gray, thresh, bkground, unknown;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
    // Remove unfilled black holes by morphological closing
    morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);

    dilate(thresh, bkground, 3);
    distanceTransform(thresh, thresh, DIST_L2, 5);

    double minVal, maxVal;
    Point minLoc, maxLoc;

    minMaxLoc(thresh, &minVal, &maxVal, &minLoc, &maxLoc);
    threshold(thresh, thresh, maxVal*0.7, 255, THRESH_BINARY);

    thresh.convertTo(thresh, CV_8U, 1, 0);

    std::vector<std::vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    int coinsValue = 0;

    for (auto contour : contours)
    {
        double radius = contourArea(contour) / 3.14;
        if (radius > 200)
        {
            coinsValue += 2;
        }
        else
        {
            coinsValue += 1;
        }
    }

    return coinsValue;


    // TODO: implement an algorithm from https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
    CV_Error(Error::StsNotImplemented, "countCoins");
}
