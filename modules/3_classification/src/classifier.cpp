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
//    CV_Error(Error::StsNotImplemented, "topK");
   
    int size = src.size();
    dst.resize(k);
    indices.resize(k);
    for (unsigned i = 0; i < k; i++)
    {
        dst[i] = src[i];
        indices[i] = i;
    }
    

    for (unsigned i = k; i < size; i++)
    {
        float min = dst[0];
        unsigned ind = 0;
        for (unsigned j = 1; j < k; j++)
        {
            if (dst[j] < min) {
                min = dst[j];
                ind = j;
            }
        }
        if (min < src[i]) {
            dst[ind] = src[i];
            indices[ind] = i;
        }
    }

    for (unsigned i = 0; i < k-1; i++)
    {
        for (unsigned j = 0; j < k-1-i; j++)
        {
            if (dst[j] < dst[j+1]) {
                int ti = indices[j];
                float tmp = dst[j];
                indices[j] = indices[j + 1];
                dst[j] = dst[j + 1];
                indices[j + 1] = ti;
                dst[j + 1] = tmp;
            }
        }
    }

}

void softmax(std::vector<float>& values) {
    //_Error(Error::StsNotImplemented, "softmax");
    unsigned size = values.size();
    std::vector<float> tmp(size);
    float max = values[0];
    for (int i = 1; i < size; i++)
    {
        if (max < values[i]) max = values[i];
    }
    for (int i = 0; i < size; i++)
    {
        values[i] -= max;
    }
    for (unsigned i = 0; i < size; i++)
    {
        float  sum = 0;
        
        for (unsigned  j= 0; j < size; j++)
        {
            sum += expf(values[j]);
        }
        tmp[i] = expf(values[i]) / sum;
    }
    values = tmp;
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
    int size = req.GetBlob(outputName)->size();
    std::vector<float> tmp(size);
    for (int i = 0; i < size; i++)
    {
        tmp[i] = output[i];
    }
    topK(tmp, k, probabilities, indices);
    softmax(probabilities);



}
