#include "opencv_coins.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <algorithm>
using namespace cv;

unsigned countCoins(const Mat& img) {
    Mat gray, thresh, newThresh;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
    // Remove unfilled black holes by morphological closing
    morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);
        
    //imshow("sad", thresh);
    //waitKey(9000);
    
    dilate(thresh, newThresh, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);
    //imshow("22", newThresh);
    //waitKey(9000);

    erode(thresh, thresh, Mat::ones(3, 3, CV_8U), Point(-1, -1), 5);
    distanceTransform(thresh, thresh, DIST_L2, 3);
    //imshow("33", thresh);
    //waitKey(9000);

    double minDistance, maxDistance;
    minMaxLoc(thresh, &minDistance, &maxDistance);
    threshold(thresh, thresh, maxDistance * 0.7, 255, THRESH_BINARY);
    //imshow("55", thresh);
    //waitKey(9000);
    
    thresh.convertTo(thresh, CV_8U, 1, 0);

    std::vector<std::vector<Point>> contoursVector;
    findContours(thresh, contoursVector, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    int coinsSum(0);
    /*
    double R(0);
    std::vector<double> area;
    for (int i(0); i < contoursVector.size(); i++) {
        area.push_back(contourArea(contoursVector[i]));
        sort(area.begin(), area.end());
    }
    double minRadius2 = sqrt(area[10] / CV_PI);
    for (int i(0); i < area.size(); i++) {
        R = sqrt(area[i] / CV_PI);
        R >= minRadius2 ? coinsSum += 2 : coinsSum++;
    }
    */

    for (auto contour : contoursVector) {
        
        double R = sqrt(contourArea(contour) / CV_PI);
        //std::cout << R << std::endl;  //radiuses of circles
        R < 10 ? coinsSum++ : coinsSum += 2;
    }
    
    
    return coinsSum;

    // TODO: implement an algorithm from https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
    CV_Error(Error::StsNotImplemented, "countCoins");
}
