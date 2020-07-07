#include "opencv_coins.hpp"
#include<algorithm>
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
    thresh.convertTo(thresh, CV_8U, 1, 0);
    imshow("ex_4", sure_fg);
    waitKey(10000);
    std::vector<std::vector<Point>> contours;
    findContours(sure_fg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    int sum = 0;
    double r = 0;
    std::vector<double> area;
    for (int i = 0; i < contours.size(); i++)
    {
        area.push_back(contourArea(contours[i]));
        sort(area.begin(), area.end());
    }
    double r_min_ofst_2 = sqrt(area[10] / CV_PI);
    for (int i = 0; i < area.size(); i++)
    {
        r = sqrt(area[i] / CV_PI);
        if (r >= r_min_ofst_2)
            sum = sum + 2;
        else
            sum++;
    }



   



}

