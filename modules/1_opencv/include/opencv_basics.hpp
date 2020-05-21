#include <string>

#include <opencv2/opencv.hpp>

using namespace cv;

// Method 
Mat createMat(int rows, int cols, int channels, int depth);

Mat readImage(const std::string& path);