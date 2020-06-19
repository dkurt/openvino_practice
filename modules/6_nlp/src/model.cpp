#include "model.hpp"

#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <inference_engine.hpp>

using namespace InferenceEngine;
using namespace cv;
using namespace cv::utils::fs;

SQuADModel::SQuADModel() : tokenizer(join(DATA_FOLDER, "bert-large-uncased-vocab.txt")) {
    Core ie;

    // Load deep learning network into memory
    CNNNetwork net = ie.ReadNetwork(join(DATA_FOLDER, "distilbert.xml"),
                                    join(DATA_FOLDER, "distilbert.bin"));

    // Initialize runnable object on CPU device
    ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");

    // Create a single processing thread
    req = execNet.CreateInferRequest();
}

std::string SQuADModel::getAnswer(const std::string& question, const std::string& source) {
    std::vector<std::string> questionTokens = tokenizer.tokenize(question);
    std::vector<std::string> sourceTokens = tokenizer.tokenize(source);
    std::vector<std::string> tokens;
    tokens.push_back("[CLS]");
    tokens.reserve(tokens.size() + questionTokens.size());
    tokens.insert(tokens.end(), questionTokens.begin(), questionTokens.end());

    tokens.push_back("[SEP]");
    tokens.reserve(tokens.size() + sourceTokens.size());
    tokens.insert(tokens.end(), sourceTokens.begin(), sourceTokens.end());
    tokens.push_back("[SEP]");

    std::vector<int> indices = tokenizer.tokensToIndices(tokens);

    // TODO: forward indices through the network and return an answer

    return "";
}
