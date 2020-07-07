#include "opencv_coins.hpp"
#include<iostream>
#include<opencv2/opencv.hpp>
using namespace cv;

unsigned countCoins(const Mat& img) {
	Mat gray, thresh, distance_transform;
	double maxVal;
	cv::Point minLoc;
	unsigned count_coins=0;
	cvtColor(img, gray, COLOR_BGR2GRAY);   //принимает исходную картинку img, возвращает тот же img,код цвета серый, канал серый
	threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);  //Применяет порог фиксированного уровня к каждому элементу массива; gray входной массив,thresh-выходной, пороговое значение от 0 (черный) до 255(белый), тип порога 

																	   // Remove unfilled black holes by morphological closing
	morphologyEx(thresh, thresh, MORPH_CLOSE, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);  //Выполняет сложные морфологические преобразования, thresh-исходное изображение (и получаемое), тип морфол операции (сохранение фоновых областей), ядро 3х3, якорная позиция (в центре ядра), 3 - кол-во применений эрозии и расширения 

	//расширение изображения - увеличивает границу обьекта до фона
	cv::dilate(thresh, thresh, Mat::ones(3, 3, CV_8U), Point(-1, -1), 3);

	//Вычисляет расстояние до ближайшего нулевого пикселя для каждого пикселя исходного изображения
	distanceTransform(thresh, distance_transform, cv::DIST_L2, 5);
	normalize(distance_transform, distance_transform, 0, 1., NORM_MINMAX);

	/*minMaxLoc(distance_transform, 0, &maxVal, &minLoc);
	threshold(distance_transform, distance_transform, 7*maxVal,255,0);*/   //все черное
	threshold(distance_transform, distance_transform, .45, 1., THRESH_BINARY);

	//создаем версию CV_8U расстояния изображения
	Mat dist_transf_8u;
	distance_transform.convertTo(dist_transf_8u, CV_8U);

	// создаем вектор, в котром будет хранится контур (изображение котнура)
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(dist_transf_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	//считаем кол-во монет
	char count = contours.size();
	putText(distance_transform, std::to_string(count), cv::Point(0, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));  //вывели кол-во монет на экран

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
	cv::namedWindow("Img", cv::WINDOW_NORMAL);
	cv::imshow("Img", distance_transform);
	cv::waitKey();
	return count_coins;
    
	
	

	
}




