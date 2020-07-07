#include "opencv_coins.hpp"

using namespace cv;

unsigned countCoins(const Mat& img) {
    Mat gray, thresh, sure_bg, dist_transform, sure_fg, res, markers;

    cvtColor(img, gray, COLOR_RGB2GRAY);

    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

    morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);

    dilate(thresh, sure_bg, Mat(), Point(-1, -1), 2, 1, 1);

    distanceTransform(thresh, dist_transform, DIST_L2, 5);
    double maxVal;
    Point maxLoc;
    minMaxLoc(dist_transform, 0, &maxVal, 0, &maxLoc);
    threshold(dist_transform, sure_fg, 0.7 * maxVal, 255, 0);

    sure_fg.convertTo(sure_fg, CV_8U);
    std::vector<std::vector<Point>> conturs;
    findContours(sure_fg, conturs, RETR_LIST, CHAIN_APPROX_NONE);
    unsigned rub = 0;
    for (int i = 0; i < conturs.size(); ++i) {
        double radius = sqrt(contourArea(conturs[i]) / CV_PI);
        if (radius > 15) {
            rub += 2;
        }
        else {
            rub += 1;
        }
    }

    /*subtract(sure_bg, sure_fg, res);
    namedWindow("res");
    imshow("res", res);
    waitKey(0);
    connectedComponents(sure_fg, markers);
    markers = markers + 1;
    for (int y = 0; y < markers.rows; ++y) {
        for (int x = 0; x < markers.cols; ++x) {
            Vec3b bgr = res.at<Vec3b>(y, x);
            if (bgr[0] == 255 && bgr[1] == 255 && bgr[2] == 255) {
                markers.at<uint8_t>(y, x) = 0;
            }
        }
    }

    watershed(img, markers);
    namedWindow("ima");
    imshow("ima", img);
    waitKey(0);*/

    /*
    // TODO: implement an algorithm from https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
    CV_Error(Error::StsNotImplemented, "countCoins");*/
    return rub;
}
