#include "opencv_coins.hpp"

#define COIN_RADIUS_STEP 30
#define HIGHER_COIN_VALUE 2
#define LOWER_COIN_VALUE 1

unsigned countCoins(const cv::Mat& img) {
    const cv::Mat COMMON_KERNEL = cv::Mat::ones(3, 3, CV_8U);

    // rgb to grayscale
    cv::Mat gray, thresh;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

    // separate the coins
    cv::Mat sure_fg;
    cv::morphologyEx(thresh, sure_fg, cv::MORPH_CLOSE, COMMON_KERNEL, cv::Point(-1, -1), 2);
    cv::distanceTransform(sure_fg, sure_fg, cv::DIST_L2, 3);

    double min, max;
    cv::minMaxLoc(sure_fg, &min, &max);
    cv::threshold(sure_fg, sure_fg, 0.45 * max, 255, cv::THRESH_BINARY);
    cv::dilate(sure_fg, sure_fg, COMMON_KERNEL, cv::Point(-1, -1), 3);

    // get contour of each separate coin
    std::vector<std::vector<cv::Point>> contours;
    sure_fg.convertTo(sure_fg, CV_8U);
    findContours(sure_fg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // count
    unsigned sum = 0;
    for (std::vector<cv::Point>& contour : contours) {
        sum += sqrt(cv::contourArea(contour) / CV_PI) > COIN_RADIUS_STEP ? HIGHER_COIN_VALUE : LOWER_COIN_VALUE;
    }

    return sum;
}
