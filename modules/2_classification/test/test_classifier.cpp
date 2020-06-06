#include <fstream>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "classifier.hpp"

using namespace cv;

TEST(classification, topK) {
    std::vector<float> src = {0.92f, 0.14f, 0.76f, 0.49f, 0.45f,
                              0.44f, 0.94f, 0.28f, 0.12f, 0.54f};
    std::vector<float> dst;
    std::vector<unsigned> indices;
    static const unsigned k = 5;

    topK(src, k, dst, indices);

    ASSERT_EQ(dst.size(), k);
    ASSERT_EQ(indices.size(), k);

    ASSERT_EQ(dst[0], 0.94f);
    ASSERT_EQ(dst[1], 0.92f);
    ASSERT_EQ(dst[2], 0.76f);
    ASSERT_EQ(dst[3], 0.54f);
    ASSERT_EQ(dst[4], 0.49f);

    std::vector<char> letters = {'n', 'a', 't', 'l', 'x',
                                 'c', 'i', 'a', 'v', 'e'};
    std::string word = "";
    for (auto& idx : indices) {
        word += letters[idx];
    }
    ASSERT_EQ(word, "intel");
}

TEST(classification, SoftMax) {
    std::vector<float> src = {0.92f, 0.14f, 0.76f, 0.49f, 0.45f,
                              0.44f, 0.94f, 0.28f, 0.12f, 0.54f};
    softmax(src);

    float sum = 0;
    for (int i = 0; i < src.size(); ++i) {
        ASSERT_GE(src[i], 0.0f);
        ASSERT_LE(src[i], 1.0f);
        sum += src[i];
    }
    ASSERT_LE(fabs(sum - 1.0f), 1e-5f);
}

TEST(classification, SoftMaxLarge) {
    std::vector<float> src = {61.32f, 44.11f, 65.31f,  69.65f,  3.52f,
                              65.19f, 43.12f,  3.74f , 97.98f, 30.85f};
    softmax(src);

    float sum = 0;
    for (int i = 0; i < src.size(); ++i) {
        ASSERT_GE(src[i], 0.0f);
        ASSERT_LE(src[i], 1.0f);
        sum += src[i];
    }
    ASSERT_LE(fabs(sum - 1.0f), 1e-5f);
}

// In this test run image classification network and get top 5 classes with the
// highest probabilities. Use implemented topK and SoftMax methods to pass tests
TEST(classification, DenseNet) {
    Mat image = imread(utils::fs::join(DATA_FOLDER, "tram.jpg"));
    std::vector<float> probabilities;
    std::vector<float> top5_scores;
    std::vector<unsigned> top5_classes;

    // Load file with classes names
    std::vector<std::string> classesNames;
    std::ifstream ifs(utils::fs::join(DATA_FOLDER, "classification_classes_ILSVRC2012.txt"));
    std::string line;
    while (std::getline(ifs, line))
        classesNames.push_back(line);

    // TODO: add code here

    ASSERT_EQ(probabilities.size(), 1000);
    ASSERT_EQ(top5_scores.size(), 5);
    ASSERT_EQ(top5_classes.size(), 5);

    for (int i = 0; i < top5_classes.size(); ++i) {
        std::cout << format("%.5f", top5_scores[i]) << " ";
        std::cout << classesNames[top5_classes[i]] << std::endl;
    }

    ASSERT_GE(top5_scores[0], 0.9);
    ASSERT_LE(top5_scores[0], 1.0);
    ASSERT_EQ(classesNames[top5_classes[0]], "streetcar, tram, tramcar, trolley, trolley car");
}
