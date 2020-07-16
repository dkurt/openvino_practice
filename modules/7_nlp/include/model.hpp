#pragma once
#include <string>

#include <inference_engine.hpp>

#include "tokenizer.hpp"

class SQuADModel {
public:
    SQuADModel();

    std::string getAnswer(const std::string& question, const std::string& source);

private:
    Tokenizer tokenizer;
    InferenceEngine::InferRequest req;
	std::string outputName;
};
