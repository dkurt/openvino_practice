#include "opencv_coins.hpp"

using namespace cv;

unsigned countCoins(const Mat& img) {
    Mat gray, thresh, opening;
    int total = 0;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

    morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);

    cv::dilate(thresh, thresh, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);

    cv::distanceTransform(thresh, opening, cv::DIST_L2, 5);
    cv::normalize(opening, opening, 0, 1., cv::NORM_MINMAX);

    cv::threshold(opening, opening, .5, 1., 0);

    cv::Mat coins;
    opening.convertTo(coins, CV_8U);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(coins, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    int counts = contours.size();

    Mat dst = Mat::zeros(img.rows, img.cols, CV_8UC3);
    Scalar color(0, 255, 0);
    std::vector<float> radious;
    for (int i = 0; i < counts; i++) {
        cv::drawContours(dst, contours, i, color);
        Point2f points;
        float rad;
        cv::minEnclosingCircle(contours[i], points, rad);
        radious.push_back(rad);
    }

    float minEl = *min_element(begin(radious), end(radious));
    float maxEl = *max_element(begin(radious), end(radious));

    for (float item : radious) {
        float temp1 = maxEl - item;
        float temp2 = item - minEl;
        if (temp1 > temp2) total++;
        else total += 2;
    }

    return total;

    CV_Error(Error::StsNotImplemented, "countCoins");

}
