#include "opencv_coins.hpp"
#include <cmath>

using namespace cv;

unsigned countCoins(const Mat& img) {
    Mat gray, thresh;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
    // Remove unfilled black holes by morphological closing
    morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);


    dilate(thresh, thresh, 3);
    distanceTransform(thresh, thresh, DIST_L2, 5);
    double min_value, max_value;
    Point minLoc, maxLoc;
    minMaxLoc(thresh, &min_value, &max_value, &minLoc, &maxLoc);
    threshold(thresh, thresh, max_value * 0.7, 255, THRESH_BINARY);

    thresh.convertTo(thresh, CV_8U);

    std::vector<std::vector<Point>> money_contours;
    // imshow("image", thresh);
     //waitKey();
    findContours(thresh, money_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    int sum = 0;
    for (int i = 0; i < money_contours.size(); i++) {
        double radius = sqrt(contourArea(money_contours[i]) / CV_PI);
        if (radius > 11) sum += 2;
        else sum++;
    }
    return sum;
}
