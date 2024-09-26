//
// Created by Andrey Ilyichev on 09/07/2020.
//
#pragma once

#include <opencv2/opencv.hpp>
#include <fstream>

class MnistImageReader {
public:
    typedef int32_t MNIST_INT;
    explicit MnistImageReader(const std::string& imgPath);

    MNIST_INT count();
    cv::Mat getNthImageRow(MNIST_INT i);
    std::vector<cv::Mat> getAllImages();

private:
    static const MNIST_INT EXPECTED_MAGIC = 0x00000803;
    MNIST_INT imagesCount{0};
    std::pair<MNIST_INT, MNIST_INT> imgConstraints{0, 0};
    std::ifstream ifs{nullptr};
    bool checkMagic();
    void readMeta();
};

class MnistLabelReader {
public:
    typedef int32_t MNIST_INT;
    explicit MnistLabelReader(const std::string& imgPath);

    MNIST_INT count();
    cv::Mat getNthLabel(MNIST_INT i);
    std::vector<int> getAllLabels();

private:
    static const MNIST_INT EXPECTED_MAGIC = 0x00000801;
    MNIST_INT labelsCount{0};
    std::ifstream ifs{nullptr};
    bool checkMagic();
};