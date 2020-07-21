#pragma once
#include <string>
#include <vector>
#include <map>

// This method splits input text on words by whitepaces and punctuation.
// It returns a vector of tokens - words without whitespaces or
// separate punctuation characters. All the characters are lowercased.
// [inp] text - input text
std::vector<std::string> basicTokenize(const std::string& text);

// This class performs vocabulary based tokenization.
class Tokenizer {
public:
    Tokenizer(const std::string& vocabFile);

    // Method splits input text on tokens by whitespace and punctuations.
    // Additionally, it checks tokens in the vocabulary and in case of missed word
    // splits by largest known words using special "##" symbols. In example,
    // "What is embedding" -> ["what", "is", "em", "##bed", "##ding", "?"]
    std::vector<std::string> tokenize(const std::string& text);

    // Maps text tokens to corresponding position indices from the vocabulary.
    // Returns a vector of size <maxNumTokens>
    // [inp] tokens   - set of tokens
    // [maxNumTokens] - if number of input tokens less than this value - fill indices by zeros.
    std::vector<int> tokensToIndices(const std::vector<std::string>& tokens, int maxNumTokens=128);

public:
    std::vector<std::string> vocab;
private:
    std::map<std::string, int> vocabMap;
};
