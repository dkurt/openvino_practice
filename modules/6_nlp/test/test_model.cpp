#include <fstream>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "model.hpp"

using namespace cv;
using namespace cv::utils::fs;

std::string readFile(const std::string& filepath) {
    std::ifstream ifs(filepath.c_str());
    CV_Assert(ifs.is_open());
    std::ostringstream iss;
    iss << ifs.rdbuf();

    std::string content = iss.str();
    std::replace(content.begin(), content.end(), '\n', ' ');
    return content;
}

TEST(model, embedding) {
    SQuADModel model;

    std::string question = "How to recognize face?";
    std::string source = "The most popular computer vision appoach for face recognition is to map face image to floating point vector called embedding and then compute a cosine distance between them.";

    std::string answer = model.getAnswer(question, source);
    ASSERT_EQ(answer, "to map face image to floating point vector called embedding");
}

TEST(model, openvino) {
    SQuADModel model;

    std::string question = "What I can use for deep learning on CPU?";
    std::string source = "Use OpenVINO toolkit if you want to have fast deep learning inference for computer vision on CPU.";

    std::string answer = model.getAnswer(question, source);
    ASSERT_EQ(answer, "openvino toolkit");
}

TEST(model, squad) {
    SQuADModel model;

    std::string question = readFile(join(DATA_FOLDER, "squad_question.txt"));
    std::string source = readFile(join(DATA_FOLDER, "squad_source.txt"));

    std::string answer = model.getAnswer(question, source);
    ASSERT_EQ(answer, "the garden");
}
