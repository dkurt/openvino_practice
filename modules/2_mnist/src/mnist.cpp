#include "mnist.hpp"
#include <fstream>

using namespace cv;

inline int readInt(std::ifstream& ifs) {
    int val;
    ifs.read((char*)&val, 4);
    std::swap(((char*)&val)[0], ((char*)&val)[3]);
    std::swap(((char*)&val)[1], ((char*)&val)[2]);
    return val;
}

void loadImages(const std::string& filepath,
                std::vector<Mat>& images) {
    std::ifstream ifs(filepath.c_str(), std::ios::binary);
    CV_CheckEQ(ifs.is_open(), true, filepath.c_str());

    int magicNum = readInt(ifs);
    CV_CheckEQ(magicNum, 2051, "");

    int numImages = readInt(ifs);
    int rows = readInt(ifs);
    int cols = readInt(ifs);

    for (int i = 0; i < numImages; i++) {
        Mat img(rows, cols, CV_8UC1);
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                char val = 0;
                ifs.read(&val, sizeof(val));
                img.at<char>(j, k) = val;
            }
        }
        images.push_back(img);
        img.release();
    }

}

void loadLabels(const std::string& filepath,
    std::vector<int>& labels) {
    std::ifstream ifs(filepath.c_str(), std::ios::binary);
    CV_CheckEQ(ifs.is_open(), true, filepath.c_str());

    int magicNum = readInt(ifs);
    CV_CheckEQ(magicNum, 2049, "");
    
    int numLabels = readInt(ifs);
    labels = std::vector<int>(numLabels);
    for (int i = 0; i < numLabels; i++) {
        char val = 0;
        ifs.read((char*)&val, 1);
        labels[i] = val;
    }

}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
    samples = Mat::zeros(images.size(), images[0].rows * images[0].cols, CV_32FC1);
    for (int i = 0; i < images.size(); i++) {
        for (int j = 0; j < images[i].rows; j++) {
            for (int k = 0; k < images[i].cols; k++) {
                samples.at<float>(i, j * images[i].cols + k) = static_cast<float>(images[i].at<uint8_t>(j, k));
            }
        }
    }
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
    const std::vector<int>& labels) {
    Ptr <ml::KNearest> model = cv::ml::KNearest::create();
    Mat samples;
    prepareSamples(images, samples);
    model->train(samples, ml::ROW_SAMPLE, labels);
    return model;
}

float validate(Ptr<ml::KNearest> model,
               const std::vector<cv::Mat>& images,
               const std::vector<int>& labels) {

    Mat samples;
    prepareSamples(images, samples);
    Mat res;
    model->predict(samples, res);
    int corr = 0;
    for (int i = 0; i < labels.size(); ++i) {
        if (labels[i] == res.at<float>(i, 0)) {
            corr++;
        }
    }
    return static_cast<float>(corr) / (labels.size());
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    Mat pred;
    resize(image, pred, Size(28, 28));
    cvtColor(pred, pred, COLOR_BGR2HSV);
    Mat channels[3];
    split(pred, &channels[0]);
    std::vector<Mat> sat_comp(1, channels[1]);
    prepareSamples(sat_comp, pred);
    int res = model->predict(pred);
    return res;
}
