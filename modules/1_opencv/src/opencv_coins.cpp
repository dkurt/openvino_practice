#include "opencv_coins.hpp"

using namespace cv;

unsigned countCoins(const Mat& img) {
  Mat gray, thresh;
  cvtColor(img, gray, COLOR_BGR2GRAY);
  threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
  morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);

  dilate(thresh, thresh, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);
  distanceTransform(thresh, thresh, DIST_L2, CV_32F);
  normalize(thresh, thresh, 0, 1., NORM_MINMAX);

  Point minLoc, maxLoc;
  double minVal, maxVal;
  minMaxLoc(thresh, &minVal, &maxVal, &minLoc, &maxLoc);
  threshold(thresh, thresh, maxVal * 0.7, 255, THRESH_BINARY);

  thresh.convertTo(thresh, CV_8U, 1, 0);
  std::vector<std::vector<Point>> contours;
  findContours(thresh, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

  unsigned coins_sum = 0;
  for (int i = 0; i < contours.size(); ++i) {
    float radius = sqrt(contourArea(contours[i]) / CV_PI);
    coins_sum += (radius > 13) ? 2 : 1;
  }
  return coins_sum;
}
