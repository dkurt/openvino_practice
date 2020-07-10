#pragma once
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>


void loadImages(const std::string& filepath,
                std::vector<cv::Mat>& images);


void loadLabels(const std::string& filepath,
                std::vector<int>& labels);


void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples);


cv::Ptr<cv::ml::KNearest> train(const std::vector<cv::Mat>& images,
                                const std::vector<int>& labels);


float validate(cv::Ptr<cv::ml::KNearest> model,
               const std::vector<cv::Mat>& images,
               const std::vector<int>& labels);


int predict(cv::Ptr<cv::ml::KNearest> model, const cv::Mat& image);
