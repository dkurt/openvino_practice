#include "opencv_coins.hpp"
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;

unsigned countCoins(const Mat& img) 
{
	Mat gray, thresh, sure_bg, sure_fg, dist_transform, unknown, markedimg;
	Mat kernel = Mat::ones(3, 3, CV_8U);

	double max = -100000.0;
	unsigned int result = 0;

	cvtColor(img, gray, COLOR_BGR2GRAY);
	threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

	morphologyEx(thresh, thresh, MORPH_CLOSE, kernel, Point(-1, -1), 3);

	dilate(thresh, sure_bg, kernel, Point(-1, -1), 3);

	distanceTransform(thresh, dist_transform, DIST_L2, 5);

	minMaxLoc(dist_transform, NULL, &max);
	threshold(dist_transform, sure_fg, 0.7*max, 255, 0);

	sure_fg.convertTo(sure_fg, CV_8U);
	subtract(sure_bg, sure_fg, unknown);

	std::vector<std::vector<cv::Point> > ar;
	findContours(sure_fg, ar, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	double Radius;

	//1 Rouble coin radius:10,25 mm
	//2 Rouble coin radius:11,5 mm

	for (int i = 0; i < ar.size(); i++)
	{
		Radius = sqrt(contourArea(ar[i]) / M_PI);
		
		if (Radius > 10.75)   //obtained average radius of 1 Rouble coin is less than 10.75
		{
			result += 2;
		}
		else
		{
			result++;
		}
	}  

	return result;
}
