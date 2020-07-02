#include <fstream>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "tokenizer.hpp"

using namespace cv;
using namespace cv::utils::fs;

Tokenizer tokenizer(join(DATA_FOLDER, "bert-large-uncased-vocab.txt"));

class basicTokenizeTest :public ::testing::TestWithParam<std::tuple<const char*, std::vector<std::string> >> {};

TEST_P(basicTokenizeTest, basicTokenize) {
    std::string sentence = std::get<0>(GetParam());
    std::vector<std::string> tokens = basicTokenize(sentence);
    std::vector<std::string> ref = std::get<1>(GetParam());
    ASSERT_EQ(ref.size(), tokens.size());
    for (int i = 0; i < ref.size(); ++i)
        ASSERT_EQ(ref[i], tokens[i]);
}

INSTANTIATE_TEST_CASE_P(
        /**/,
        basicTokenizeTest,
        ::testing::Values(
                std::make_tuple("This is NLP practice", std::vector<std::string>({"this", "is", "nlp", "practice"})),
                std::make_tuple("  Just    multiple   whitespaces", std::vector<std::string>({"just", "multiple", "whitespaces"})),
                std::make_tuple("Hi! I'm here", std::vector<std::string>({"hi", "!", "i", "'", "m", "here"})),
                std::make_tuple("This is C++ tokenizer", std::vector<std::string>({"this", "is", "c", "+", "+", "tokenizer"}))
));

TEST(tokenizer, embedding) {
    std::string sentence = "What is embedding?";
    std::vector<std::string> tokens = tokenizer.tokenize(sentence);
    std::vector<std::string> ref = {"what", "is", "em", "##bed", "##ding", "?"};
    ASSERT_EQ(ref.size(), tokens.size());
    for (int i = 0; i < ref.size(); ++i) {
        ASSERT_EQ(ref[i], tokens[i]);
    }
}

TEST(tokenizer, simple) {
    std::string sentence = "Can you tell  me, how to go to the library?";
    std::vector<std::string> ref = {"can", "you", "tell", "me", ",", "how", "to", "go", "to", "the", "library", "?"};
    std::vector<int> refIndices = {2064, 2017, 2425, 2033, 1010, 2129, 2000, 2175, 2000, 1996, 3075, 1029, 102};

    std::vector<std::string> tokens = tokenizer.tokenize(sentence);
    std::vector<int> indices = tokenizer.tokensToIndices(tokens, refIndices.size());
    ASSERT_EQ(ref.size(), tokens.size());
    for (int i = 0; i < ref.size(); ++i) {
        ASSERT_EQ(ref[i], tokens[i]);
        ASSERT_EQ(refIndices[i], indices[i]);
    }
}

TEST(tokenizer, longestMatchFirst) {
    std::string sentence = "This module demonstrates DistilBERT in C++ with OpenVINO";
    std::vector<std::string> ref = {"this", "module", "demonstrates", "di", "##sti", "##lbert",
                                    "in", "c", "+", "+", "with", "open", "##vino"};
    std::vector<int> refIndices = {2023, 11336, 16691, 4487, 16643, 23373, 1999,
                                   1039, 1009, 1009, 2007, 2330, 26531, 102};

    std::vector<std::string> tokens = tokenizer.tokenize(sentence);
    std::vector<int> indices = tokenizer.tokensToIndices(tokens, refIndices.size() + 5);
    ASSERT_EQ(ref.size(), tokens.size());
    ASSERT_EQ(refIndices.size() + 5, indices.size());
    for (int i = 0; i < ref.size(); ++i) {
        ASSERT_EQ(ref[i], tokens[i]);
        ASSERT_EQ(refIndices[i], indices[i]);
    }
    for (int i = 0; i < 5; ++i)
        ASSERT_EQ(indices[refIndices.size() + i], 0);
}
