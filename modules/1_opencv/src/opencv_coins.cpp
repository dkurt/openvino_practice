#include "opencv_coins.hpp"

#include<iostream>
#include<opencv2/opencv.hpp>
using namespace cv;

unsigned countCoins(const Mat& img) {
	Mat gray, thresh, distance_transform;
	double maxVal;
	cv::Point minLoc;
	unsigned count_coins=0;
	cvtColor(img, gray, COLOR_BGR2GRAY);   //get source image img, give the same img,channel gray
	threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);  //aplies a fixed level threshol to each element of the array; gray- input,thresh-ouyput, threashold  0 (black)- 255(white), type threshold 

	//A complex morphological transformation is performed. Thresh-source image (and received), type of morphol operation (preservation of background areas)
	morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);  

	//image extension - increases the border of the object to the background
	cv::dilate(thresh, thresh, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);

	//Calculates the distance to the nearest zero pixel for each pixel in the original image.
	distanceTransform(thresh, distance_transform, cv::DIST_L2, 5);
	normalize(distance_transform, distance_transform, 0, 1., NORM_MINMAX);

	/*minMaxLoc(distance_transform, 0, &maxVal, &minLoc);
	threshold(distance_transform, distance_transform, 0.7*maxVal,255,0);*/   //all is black
	threshold(distance_transform, distance_transform, .45, 1., THRESH_BINARY);

	//create the CV_8U version of the image distance
	Mat dist_transf_8u;
	distance_transform.convertTo(dist_transf_8u, CV_8U);

	// we create a vector in which the contour will be stored (image of a kotnur)
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(dist_transf_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	//count the number of coins
	char count = contours.size();
	putText(distance_transform, std::to_string(count), cv::Point(0, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));  //brought the number of coins to the picture

	Point2f center[16];
    float radius[16];

	  for (int i = 0; i< contours.size(); i++)
	  {
		  minEnclosingCircle(contours[i], center[i], radius[i]);
		  if (radius[i] > 37)
			  count_coins += 2;
		  else
			  count_coins += 1;
	  }

	  char countC = count_coins;
	  putText(distance_transform, std::to_string(countC), cv::Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
	  
	cv::imwrite("newp2.jpg", distance_transform);

	return count_coins;
	
}

