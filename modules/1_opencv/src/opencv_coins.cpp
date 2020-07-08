#include "opencv_coins.hpp"

using namespace cv;
using namespace std;

unsigned countCoins(const Mat& img) {
    Mat gray, thresh, dist;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
    morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);

    dilate(thresh, thresh, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);
    distanceTransform(thresh, thresh, DIST_L2, 5);

    double maxV;
    Point maxL;

    minMaxLoc(thresh, 0, &maxV, 0, &maxL);
    threshold(thresh, thresh, maxV*0.7, 255, THRESH_BINARY);
    thresh.convertTo(thresh, CV_8U, 1, 0);

    vector<vector<Point>> cont;
    findContours(thresh, cont, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    int sum = 0;
    double radius;
    for (int i = 0; i < cont.size(); i++)
    {
        radius = sqrt(contourArea(cont[i]) / CV_PI);
        if (radius > 14)
        {
            sum += 2;
        }
        else
        {
            sum++;
        }   
    }
    return sum;
}
