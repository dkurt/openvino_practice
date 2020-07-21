#include "tokenizer.hpp"

#include <fstream>
#include <algorithm>

#include <opencv2/opencv.hpp>

std::vector<std::string> basicTokenize(const std::string& text) {
    std::vector<std::string> basicTokens;
    std::string currToken = "";
    for (auto ch : text) {
        if (isspace(ch)) {
            if (!currToken.empty())
                basicTokens.push_back(currToken);
            currToken = "";
        } else if (ispunct(ch)) {
            if (!currToken.empty())
                basicTokens.push_back(currToken);
            currToken = ""; currToken += ch;
            basicTokens.push_back(currToken);
            currToken = "";
        } else {
            currToken += tolower(ch);
        }
    }
    if (!currToken.empty())
        basicTokens.push_back(currToken);
    return basicTokens;
}

std::vector<std::string> wordTokenize(const std::string& word,
                                      const std::map<std::string, int>& vocab) {
    std::vector<std::string> tokens;
    int start = 0;
    while (start < word.size()) {
        bool found = false;
        for (int end = word.size() - 1; end >= start; --end) {
            std::string token = word.substr(start, end - start + 1);
            if (start)
                token = "##" + token;

            if (vocab.find(token) != vocab.end()) {
                tokens.push_back(token);
                found = true;
                start = end + 1;
                break;
            }
        }
        if (!found) {
            tokens.push_back("[UNK]");
        }
    }
    return tokens;
}

Tokenizer::Tokenizer(const std::string& vocabFile) {
    std::ifstream ifs(vocabFile.c_str());
    std::string line;
    while (std::getline(ifs, line)) {
        vocabMap[line] = vocab.size();
        vocab.push_back(line);
    }
}

std::vector<std::string> Tokenizer::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    for (auto& basicToken : basicTokenize(text)) {
        for (auto& wordToken : wordTokenize(basicToken, vocabMap)) {
            tokens.push_back(wordToken);
        }
    }
    return tokens;
}

std::vector<int> Tokenizer::tokensToIndices(const std::vector<std::string>& tokens, int maxNumTokens) {
    CV_CheckLE((int)tokens.size(), maxNumTokens, "Maximum number of tokens");

    std::vector<int> indices(maxNumTokens, 0);
    for (int i = 0; i < tokens.size(); ++i) {
        auto it = vocabMap.find(tokens[i]);
        if (it == vocabMap.end())
            CV_Error(cv::Error::StsObjectNotFound, "Cannot find in vocabulary: " + tokens[i]);
        indices[i] = vocabMap[tokens[i]];
    }
    return indices;
}
