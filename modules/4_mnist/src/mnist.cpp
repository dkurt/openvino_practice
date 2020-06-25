Skip to content
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 
@MariaMedvede 
MariaMedvede
/
openvino_practice
forked from dkurt/openvino_practice
0
02
Code
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
openvino_practice/modules/4_mnist/src/mnist.cpp
@MariaMedvede
MariaMedvede mnist practice
Latest commit ed6de6b 1 hour ago
 History
 2 contributors
@MariaMedvede@dkurt
151 lines (128 sloc)  4.03 KB
  
#include "mnist.hpp"
#include <fstream>

using namespace cv;

inline int readInt(std::ifstream& ifs) {
    int val;
    ifs.read((char*)&val, 4);
    // Integers in file are high endian which requires swap
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

	int n_rows = readInt(ifs);
	int n_cols = readInt(ifs);
	char* pixels = new char[n_rows * n_cols];
	for (int item_id = 0; item_id < numImages; ++item_id) {
		ifs.read(pixels, n_rows * n_cols);
		cv::Mat image_tmp(n_rows, n_cols, CV_8UC1, pixels);
		Mat m2 = image_tmp.clone();
		images.push_back(m2);
	}
}

void loadLabels(const std::string& filepath,
	std::vector<int>& labels) {
	std::ifstream ifs(filepath.c_str(), std::ios::binary);
	CV_CheckEQ(ifs.is_open(), true, filepath.c_str());

	int magicNum = readInt(ifs);
	CV_CheckEQ(magicNum, 2049, "");

	int numLabels = readInt(ifs);
	char label;
	// TODO: follow "FILE FORMATS FOR THE MNIST DATABASE" specification
	// at http://yann.lecun.com/exdb/mnist/

	for (int item_id = 0; item_id < numLabels; ++item_id) {
		// read label
		ifs.read(&label, 1);
		labels.push_back(int(label));
	}
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
	
	for (int i = 0; i < images.size(); i++)
	{
		samples.push_back(images[i].reshape(1, 1));
	}
	samples.convertTo(samples, CV_32F);
}

using namespace ml;

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
                        const std::vector<int>& labels) {
	Ptr<ml::TrainData> trainingData;
	Ptr<ml::KNearest> kclassifier = KNearest::create();
	cv::Mat samples;
	prepareSamples(images, samples);
	trainingData = TrainData::create(samples,
		SampleTypes::ROW_SAMPLE, labels);
	kclassifier->setIsClassifier(true);
	kclassifier->setAlgorithmType(KNearest::Types::BRUTE_FORCE);
	kclassifier->setDefaultK(1);
	kclassifier->train(trainingData);
	return kclassifier;
}

float validate(Ptr<ml::KNearest> model,
               const std::vector<cv::Mat>& images,
               const std::vector<int>& labels) {
	//std::vector<int> results(images.size());
	cv::Mat results;
	cv::Mat samples;
	cv::Mat lab;

	std::cout << "preparing samples" << std::endl;
	prepareSamples(images, samples);
	std::cout << "predicting values" << std::endl;
	//model->findNearest(samples, 1, results);
	model->predict(samples, results);
	std::vector <int> res = results.clone();
	int size = results.rows;

	/*std::cout << "results are:" << std::endl;
	for (int i = 0; i < size; i++)
	{
		std::cout << res[i] << " ";
		
	}
	std::cout << "labels are:" << std::endl;
	for (size_t i = 0; i < labels.size(); i++)
	{
		std::cout <<labels[i] << " ";
	}*/
	int correct = 0;
	for (size_t i = 0; i < labels.size(); i++)
	{
		if (res[i] == labels[i])
		{
			correct += 1;
		}
	}
	std::cout << "predicted values" << std::endl;
	//float correct = countNonZero(res == labels);
	float accuracy = correct / float(images.size());
	return accuracy;
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
	// TODO: resize image to 28x28
	std::cout << "resize" << std::endl;
	Mat resimage;
	cv::resize(image, resimage, cv::Size(28, 28), INTER_AREA);
    // TODO: convert image from BGR to HSV
	std::cout << "convert" << std::endl;
	cv::cvtColor(resimage, resimage, cv::COLOR_BGR2HSV);
    // TODO: get Saturate component
	std::cout << "saturate" << std::endl;
	Mat channels[3];
	cv::split(resimage,channels);

    // TODO: prepare input - single row FP32 Mat
	std::cout << "channel " << std::endl;
	std::vector <cv::Mat> sample;
	sample.push_back(channels[1]);
	
	cv::Mat newim;
	std::cout << "prepare" << std::endl;
	prepareSamples(sample, newim);
    // TODO: make a prediction by the model
	std::cout << "predict" << std::endl;
	int result;
	result = model->predict(newim);
	return result;
}