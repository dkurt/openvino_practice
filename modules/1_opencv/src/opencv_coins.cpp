#include "opencv_coins.hpp"

using namespace cv;

unsigned countCoins(const Mat& img) {
    Mat gray, thresh, bg, fg, dist_transform, markers, unknown, opening;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
    // Remove unfilled black holes by morphological closing
    morphologyEx(thresh, opening, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);

    // TODO: implement an algorithm from https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
    dilate(thresh, bg, Mat(3, 3, CV_8U), Point(-1, -1), 3);

    distanceTransform(opening, dist_transform, DIST_L2, CV_32F);
    double minVal, maxVal;
    minMaxLoc(dist_transform, &minVal, &maxVal);
    threshold(dist_transform, fg, maxVal * 0.7, 255, THRESH_BINARY);
    fg.convertTo(fg, CV_8U, 1, 0);
    std::vector<std::vector<Point>> contours;
    findContours(fg, contours, RETR_LIST, CHAIN_APPROX_NONE);
    unsigned int sum = 0, numofcoin = 0;

    for (int i = 0; i < contours.size(); i++)
    {
        double radius = sqrt(contourArea(contours[i]) / CV_PI);
        if (radius > 15) sum += 2;
        else sum++;
    }

    return sum;
    //CV_Error(Error::StsNotImplemented, "countCoins");
}
