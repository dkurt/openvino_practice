//
// Created by Andrey Ilyichev on 09/07/2020.
//
#include "mnist_reader.hpp"
#include <fstream>

void reverseInt32(int &i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;

    i = ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

MnistImageReader::MnistImageReader(const std::string &imgPath)
        : ifs(imgPath.c_str(), std::ios::binary) {
    CV_CheckEQ(ifs.is_open(), true, imgPath.c_str());
    CV_CheckEQ(checkMagic(), 1, "Wrong magic");

    readMeta();
}

bool MnistImageReader::checkMagic() {
    ifs.clear();
    ifs.seekg(0, std::ios::beg);

    int val(0);
    ifs.read((char *) &val, 4);
    // Integers in file are high endian which requires swap
    reverseInt32(val);

    return val == EXPECTED_MAGIC;
}

MnistImageReader::MNIST_INT MnistImageReader::count() {
    return imagesCount;
}

cv::Mat MnistImageReader::getNthImageRow(const MnistImageReader::MNIST_INT i) {
    const int GENERAL_OFFSET = 16;
    const int IMAGE_SQUARE = imgConstraints.first * imgConstraints.second;
    cv::Mat image(1, IMAGE_SQUARE, CV_8U);

    ifs.clear();
    ifs.seekg(GENERAL_OFFSET + IMAGE_SQUARE * i, std::ios::beg);

    for (int i = 0; i < IMAGE_SQUARE; i++) {
        char pixel;
        ifs.read(&pixel, 1);
        image.at<unsigned char>(0, i) = pixel;
    }

    return image;
}

std::vector<cv::Mat> MnistImageReader::getAllImages() {
    const int GENERAL_OFFSET = 16;
    const int IMAGE_SQUARE = imgConstraints.first * imgConstraints.second;
    std::vector<cv::Mat> images;

    ifs.clear();
    ifs.seekg(GENERAL_OFFSET, std::ios::beg);

    for (int imageIndex = 0; imageIndex < imagesCount; imageIndex++) {
        char *pixels = (char *)malloc(sizeof(char) * IMAGE_SQUARE);
        ifs.read(pixels, IMAGE_SQUARE);
        cv::Mat image(cv::Mat(1, IMAGE_SQUARE, CV_8U, pixels).clone());

        free(pixels);

        images.push_back(image.reshape(1, imgConstraints.first));
    }

    return images;
}

void MnistImageReader::readMeta() {
    ifs.clear();
    ifs.seekg(4, std::ios::beg);

    ifs.read((char *) &imagesCount, 4);
    reverseInt32(imagesCount);

    ifs.read((char *) &imgConstraints.first, 4);
    reverseInt32(imgConstraints.first);

    ifs.read((char *) &imgConstraints.second, 4);
    reverseInt32(imgConstraints.second);
}




MnistLabelReader::MnistLabelReader(const std::string &imgPath)
        : ifs(imgPath.c_str(), std::ios::binary) {
    CV_CheckEQ(ifs.is_open(), true, imgPath.c_str());
    CV_CheckEQ(checkMagic(), 1, "Wrong magic");

    ifs.clear();
    ifs.seekg(4, std::ios::beg);

    ifs.read((char *) &labelsCount, 4);
    reverseInt32(labelsCount);
}

bool MnistLabelReader::checkMagic() {
    ifs.clear();
    ifs.seekg(0, std::ios::beg);

    int val(0);
    ifs.read((char *) &val, 4);
    // Integers in file are high endian which requires swap
    reverseInt32(val);

    return val == EXPECTED_MAGIC;
}

MnistLabelReader::MNIST_INT MnistLabelReader::count() {
    return labelsCount;
}

cv::Mat MnistLabelReader::getNthLabel(const MnistLabelReader::MNIST_INT i) {
    return cv::Mat();
}

std::vector<int> MnistLabelReader::getAllLabels() {
    const int GENERAL_OFFSET = 8;
    std::vector<int> labels;

    ifs.clear();
    ifs.seekg(GENERAL_OFFSET, std::ios::beg);

    for (int labelIndex = 0; labelIndex < labelsCount; labelIndex++) {
        int labelValue = 0;
        ifs.read((char *) &labelValue, 1);
        //reverseInt32(labelValue);

        labels.push_back(labelValue);
    }

    return labels;
}
