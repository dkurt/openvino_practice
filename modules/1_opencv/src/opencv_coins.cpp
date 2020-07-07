#include "opencv_coins.hpp"
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace cv;

unsigned countCoins(const Mat& img) {
    cv::Mat image = cv::imread("C:\\Users\\Alexandra\\openvino_practice\\data\\coins.jpg");
    Mat gray, thresh,sure_bg;
    int sum = 0;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
    // Remove unfilled black holes by morphological closing
    morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);

    // TODO: implement an algorithm from https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
    CV_Error(Error::StsNotImplemented, "countCoins");
    dilate(thresh, sure_bg, 3);
    distanceTransform(thresh, thresh, DIST_L2, 5);
    double maxValue, minValue;
    Point maxLoc, minLoc;
    minMaxLoc(thresh, &minValue, &maxValue, &minLoc, &maxLoc);
    threshold(thresh, thresh, 0.7 * maxValue, 255, 0);
    thresh.convertTo(thresh, CV_8U);
    std::vector<std::vector<Point> > contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); i++)
    {
        double radius = sqrt(contourArea(contours[i]) / CV_PI);
        if (radius > 12)
        {
            sum += 2;
        }
        else sum += 1;
    }
    return sum;
}
