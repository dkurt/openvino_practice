#include "opencv_coins.hpp"

using namespace cv;

unsigned countCoins(const Mat& img) {
    Mat gray, thresh, dist, sure_bg, sure_fg;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
    
    // Remove unfilled black holes by morphological closing
    Mat kernel = Mat::ones(3, 3, CV_8U);
    morphologyEx(thresh, thresh, MORPH_CLOSE, kernel, Point(-1, -1), 3);

    distanceTransform(thresh, dist, DIST_L2, 5);
    double maxVal = 0.0;
    minMaxLoc(dist, 0, &maxVal);
    threshold(dist, sure_fg, 0.7*maxVal, 255, 0);

    cv::Mat sure_fg_8u;
    sure_fg.convertTo(sure_fg_8u, CV_8U);

    std::vector<std::vector<cv::Point> > contours;
    findContours(sure_fg_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    std::vector<double> radiuses2;
    for (size_t i = 0; i < contours.size(); i++) {
        radiuses2.push_back(contourArea(contours[i]) / CV_PI);
    }

    // 105.0625 = 10.25^2; 10.25mm - radius 1 ruble
    // 132.25 = 11.5^2; 11.5mm - radius 1 ruble
    int sum = 0;
    for (size_t i = 0; i < radiuses2.size(); i++) {
        if (fabs(radiuses2[i] - 105.0625) > fabs(radiuses2[i] - 132.25)) {
            sum += 2;
        }
        else {
            sum += 1;
        }
    }
    
    return sum;
}
