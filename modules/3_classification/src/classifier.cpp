#include "classifier.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <inference_engine.hpp>

using namespace InferenceEngine;
using namespace cv;
using namespace cv::utils::fs;



void topK(const std::vector<float>& src, unsigned k,
          std::vector<float>& dst,
          std::vector<unsigned>& indices) {
    //CV_Error(Error::StsNotImplemented, "topK");

	std::vector<float> src_copy = src;
	std::sort(src_copy.rbegin(), src_copy.rend());

	//unsigned index;
	for (int i = 0; i < k; i++)
	{
		dst.push_back(src_copy[i]);
		for (unsigned index = 0; index < src.size(); index++)
		{
			if (src[index] == src_copy[i])
			{
				indices.push_back(index);
				//std::cout << "Value " << src_copy[i] << " at " << i << std::endl;
				break;
			}
		}
	}

}

void softmax(std::vector<float>& values) {

	long double sum = 0;
	for (float value : values)
	{
		sum += exp((long double)value);
		//std::cout << sum << ' ';
	}
	//std::cout << std::endl;
	for (int i = 0; i < values.size(); i++)
	{
		long double exponent = exp((long double)values[i]);
		values[i] = static_cast<float>(exponent / sum);
		//std::cout << values[i] << ' ';
	}
	//std::cout << std::endl;
	
}

Blob::Ptr wrapMatToBlob(const Mat& m) {
    CV_Assert(m.depth() == CV_8U);
    std::vector<size_t> dims = {1, (size_t)m.channels(), (size_t)m.rows, (size_t)m.cols};
    return make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC),
                                     m.data);
}

Classifier::Classifier() {
    Core ie;

    // Load deep learning network into memory
    CNNNetwork net = ie.ReadNetwork(join(DATA_FOLDER, "DenseNet_121.xml"),
                                    join(DATA_FOLDER, "DenseNet_121.bin"));

    // Specify preprocessing procedures
    // (NOTE: this part is different for different models!)
    InputInfo::Ptr inputInfo = net.getInputsInfo()["data"];
    inputInfo->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
    inputInfo->setLayout(Layout::NHWC);
    inputInfo->setPrecision(Precision::U8);
    outputName = net.getOutputsInfo().begin()->first;

    // Initialize runnable object on CPU device
    ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");

    // Create a single processing thread
    req = execNet.CreateInferRequest();
}

void Classifier::classify(const cv::Mat& image, int k, std::vector<float>& probabilities,
                          std::vector<unsigned>& indices) {
    // Create 4D blob from BGR image
    Blob::Ptr input = wrapMatToBlob(image);

    // Pass blob as network's input. "data" is a name of input from .xml file
    req.SetBlob("data", input);

    // Launch network
    req.Infer();

    // Copy output. "prob" is a name of output from .xml file
    float* output = req.GetBlob(outputName)->buffer();
	
	// Transfer output values to vector
	std::vector<float> res;
	for (int i = 0; i < req.GetBlob(outputName)->size(); i++)
	{
		res.push_back(output[i]);
	}
	topK(res, k, probabilities, indices);

	// Normalize the output
	for (int i = 0; i < k; i++)
	{
		probabilities[i] /= probabilities[0];
	}
}
