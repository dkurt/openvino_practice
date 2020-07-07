#pragma once

#include <opencv2/opencv.hpp>

double const M_PI = 3.14159265358979323846;

// Count total sum of numbers on the test image with coins
unsigned countCoins(const cv::Mat& img);
