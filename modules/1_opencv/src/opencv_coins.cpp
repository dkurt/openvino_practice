#include "opencv_coins.hpp"
#include<math.h>
using namespace cv;

unsigned countCoins(const Mat& img) {
    Mat gray, thresh, sure_bg,  sure_fg;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
    // Remove unfilled black holes by morphological closing
    morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);
    //sure background area
    sure_bg = thresh;
    dilate(thresh, sure_bg,Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);
    // Finding sure foreground area
    erode(thresh, sure_fg, Mat::ones(3, 3, CV_8U), Point(-1, -1), 5);
    distanceTransform(sure_fg, sure_fg, DIST_L2, 5);
    double min_thresh, max_threh;
    minMaxLoc(sure_fg, &min_thresh, &max_threh);
    threshold(sure_fg, sure_fg,0.7 * max_threh,255, THRESH_BINARY);
    sure_fg.convertTo(sure_fg, CV_8U, 1, 0);
    std::vector<std::vector<Point>> contours_coins;
    findContours(sure_fg, contours_coins, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    int s = 0;
    double R = 0;
    for (int i = 0; i < contours_coins.size(); i++)
    {
        R = sqrt(contourArea(contours_coins[i]) / CV_PI);
        if (R > 11) //11 approximately shows the radius difference between coins
            s += 2;
        else
            s++;

    }
    return s;
}

