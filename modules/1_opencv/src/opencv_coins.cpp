#include "opencv_coins.hpp"

using namespace cv;

unsigned countCoins(const Mat& img) {
    Mat gray, thresh, dist, foreground, without_noise;
    cvtColor(img, gray, COLOR_BGR2GRAY);   

    threshold(gray, thresh, 180, 255, THRESH_BINARY_INV);
    morphologyEx(thresh, without_noise, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, 1), 3);
    distanceTransform(without_noise, dist, DIST_L2, 5);

    double maxValue;
    minMaxLoc(dist, 0, &maxValue);

    threshold(dist, foreground, 0.7 * maxValue, 255, 0);
    Mat converted_foreground;
    foreground.convertTo(converted_foreground, CV_8U);

    std::vector<std::vector<Point>> contours;
    findContours(converted_foreground, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    int count_coins = 0;
    for (auto contour : contours) 
    {
        if (contourArea(contour) / CV_PI > 200) 
        {
            count_coins += 2;
        }
        else 
        {
            count_coins += 1;
        }
    }

    return count_coins;
    CV_Error(Error::StsNotImplemented, "countCoins");
}