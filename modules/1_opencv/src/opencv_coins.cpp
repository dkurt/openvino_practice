#include "opencv_coins.hpp"


using namespace cv;


//Mat img = imread("C:\\Users\\????\\openvino_practice\\data\\coins.jpg");


unsigned countCoins(const Mat& img) {

    Mat gray, thresh;
    Mat bground;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
    //  Remove unfilled black holes by morphological closing
    morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);
    // TODO: implement an algorithm from https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
    dilate(thresh, bground, 3);
    distanceTransform(thresh, thresh, DIST_L2, 5);

    double minVal, maxVal;
    Point minLoc, maxLoc;

    minMaxLoc(thresh, &minVal, &maxVal, &minLoc, &maxLoc);
    threshold(thresh, thresh, 0.7 * maxVal, 255, THRESH_BINARY);


    thresh.convertTo(thresh, CV_8U, 1, 0);
    /*imshow("image", thresh);
    waitKey();*/

    std::vector<std::vector<Point>> circuits;
    findContours(thresh, circuits, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    int sum = 0;
    for (int i = 0; i < circuits.size(); i++)
    {

        double r = sqrt(contourArea(circuits[i]) / CV_PI);
        if (r > 12)
            sum += 2;
        else
            sum += 1;
    }

    return sum;
}