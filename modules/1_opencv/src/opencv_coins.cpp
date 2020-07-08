#include "opencv_coins.hpp"
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv;

unsigned countCoins(const Mat& img) {
    Mat gray, thresh;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
    // Remove unfilled black holes by morphological closing
    morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);

    // TODO: implement an algorithm from https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
    //CV_Error(Error::StsNotImplemented, "countCoins");

	Mat sure_bg;
	dilate(thresh, sure_bg, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);
	sure_bg.convertTo(sure_bg, CV_8U);

	Mat dist_transform;
	distanceTransform(thresh, dist_transform, DIST_L2, 5);
	dist_transform.convertTo(dist_transform, CV_8U);

	// Find the max in the dist_transform
	uint8_t max;
	for (int y = 0; y < dist_transform.rows; y++)
	{
		for (int x = 0; x < dist_transform.cols; x++)
		{
			if (dist_transform.at<uint8_t>(Point(x, y)) > max)
			{
				max = dist_transform.at<uint8_t>(Point(x, y));
			}
		}
	}

	Mat sure_fg;
	double thresholdValue = threshold(dist_transform, sure_fg,
		0.7 * max, 255, 0);
	sure_fg.convertTo(sure_fg, CV_8U);

	Mat unknown;
	subtract(sure_bg, sure_fg, unknown, noArray(), CV_8U);

	Mat markers;
	int markers_amount = connectedComponents(sure_fg, markers);
	//std::cout << markers_amount << std::endl;
	markers.convertTo(markers, CV_8U);
	markers += 1;
	
	for (int y = 0; y < markers.rows; y++)
	{
		for (int x = 0; x < markers.cols; x++)
		{
			if (unknown.at<uint8_t>(Point(x, y)) == 255)
			{
				markers.at<uint8_t>(Point(x, y)) = 0;
			}
		}
	}
	markers.convertTo(markers, CV_32S);
	watershed(img, markers);

	// Store the area of each coin here
	std::vector<int> coins;
	//std::vector<int> colors;

	int value;
	// Now let's ruin our grayscale image :D (for debug visualisation purposes)
	for (int y = 0; y < markers.rows; y++)
	{
		for (int x = 0; x < markers.cols; x++)
		{
			value = markers.at<int>(Point(x, y));
			// This can be replaced with a single if, but i'm not gonna bother
			switch (value)
			{
			case 1:
				//gray.at<uint8_t>(Point(x, y)) = 0;
				break;
			case -1:
				//gray.at<uint8_t>(Point(x, y)) = 255;
				break;
			default:
				/*if (colors.size() < value - 1)
				{
					// Assign a random grey color to each region (except bg ofc)
					colors.push_back(rand() % 128 + 64);
				}
				gray.at<uint8_t>(Point(x, y)) = colors[value-2];
				*/
				if (coins.size() < value - 1)
				{
					coins.push_back(0);
				}
				coins[value - 2]++;
			}
		}
	}
	// Sort the coins (not much sense in it though)
	std::sort(coins.begin(), coins.end());


	// Output the areas (debug)
	/*for (int i = 0; i < coins.size(); i++)
	{
		std::cout << coins[i] << ' ';
	}
	std::cout << std::endl;*/

	// Calculate the sum. Size threshold at 6000 px (1 rub is 5000-somethin px, 2 rub is 7000-something px)
	unsigned sum = 0;
	for (int area : coins)
	{
		if (area > 6000)
		{
			sum += 2;
		}
		else
		{
			sum += 1;
		}
	}

	// Debug
	//imshow("image", gray);
	//waitKey();
	//std::cout << sum << std::endl;
	return sum;
}
