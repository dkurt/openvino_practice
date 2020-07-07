#include "opencv_coins.hpp"
#include "algorithm"
#include "math.h"
#include <iostream>

using namespace cv;
using namespace std;

unsigned countCoins(const Mat& img) {
	Mat gray, thresh, bg, fg, unknown;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
	// Remove unfilled black holes by morphological closing
	morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);

	dilate(thresh, bg, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);
	erode(thresh, fg, Mat::ones(3, 3, CV_8U), Point(-1, -1), 5);
	distanceTransform(fg, fg, DIST_L2, 5);
	double min, max;
	minMaxLoc(fg, &min, &max);
	threshold(fg, fg, max*0.7, 255, THRESH_BINARY);
	vector<vector<Point>> contours;
	fg.convertTo(fg, CV_8U);
	findContours(fg, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	int sum = 0;
	double r = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		r = sqrt(contourArea(contours[i]) / CV_PI);
		if (r > 11) //11 approximately shows the radius difference between coins
			sum += 2;
		else
			sum++;
	}
	return sum;
	// TODO: implement an algorithm from https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
}