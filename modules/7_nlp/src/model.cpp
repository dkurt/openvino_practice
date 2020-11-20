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
	InputInfo::Ptr inputInfo = net.getInputsInfo()["input.1"];
	inputInfo->setPrecision(Precision::I32);
    // Initialize runnable object on CPU device
    ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");

    // Create a single processing thread
    req = execNet.CreateInferRequest();
	outputName = net.getOutputsInfo().begin()->first;
}

Blob::Ptr wrapVecToBlob(const std::vector<int> str) {
	std::vector<size_t> dims = { 1, 128 };
	return make_shared_blob<int>(TensorDesc(Precision::I32, dims, Layout::NC),
		(int*)str.data());
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
	Blob::Ptr input = wrapVecToBlob(indices);
	req.SetBlob("input.1", input);
	req.Infer();
	float* output = req.GetBlob(outputName)->buffer();
    return "";
}
