#include "opencv_coins.hpp"

using namespace cv;

unsigned countCoins(const Mat& img) {
	Mat gray, thresh;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
	morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);
	dilate(thresh, thresh, 3);
	distanceTransform(thresh, thresh, DIST_L2, 5);

	double maxValue, minValue;
	Point maxLoc, minLoc;
	minMaxLoc(thresh, &minValue, &maxValue, &minLoc, &maxLoc);
	threshold(thresh, thresh, 0.7*maxValue, 255, 0);

	thresh.convertTo(thresh, CV_8U);
	std::vector<std::vector<Point> > contours;
	findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	int Sum = 0;
	for (int i = 0; i < contours.size(); i++){
          double radius = arcLength(contours[i], true) / (2 * CV_PI);
          if (radius > 15){
             Sum += 2;
          }
          else Sum += 1;
      }
      return Sum;
}
