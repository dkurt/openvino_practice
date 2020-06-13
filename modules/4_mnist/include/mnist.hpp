#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// Load MNIST dataset images
// [inp] filepath - path to images binary file
// [out] images   - vector of images
void loadImages(const std::string& filepath,
                std::vector<cv::Mat>& images);

// Load MNIST dataset labels
// [inp] filepath - path to labels binary file
// [out] labels   - vector of digits labels (values in range 0..9)
void loadLabels(const std::string& filepath,
                std::vector<int>& labels);

// Internal procedure to prepare input images to pass to statistical model
// [inp] images  - set of images of the same type and size
// [out] samples - output Mat with number number of rows equals to number of
//                 images and number of columns equals to product of width and
//                 height of input images
void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples);

// Train OpenCV K-nearest neighbor classifier. Returns pointer to trained model
// [inp] images - set of train images
// [inp] labels - set of corresponding labels for digits
cv::Ptr<cv::ml::KNearest> train(const std::vector<cv::Mat>& images,
                                const std::vector<int>& labels);

// Validate trained model on test data. Returns accuracy (ratio of correct
// predictions over total number of test samples)
// [inp] model  - trained model
// [inp] images - set of test images
// [inp] labels - set of groundtruth labels
float validate(cv::Ptr<cv::ml::KNearest> model,
               const std::vector<cv::Mat>& images,
               const std::vector<int>& labels);

// Implements OCR on test water counter image. Returns predicted digit
// [inp] model - trained model
// [inp] image - digit BGR image
int predict(cv::Ptr<cv::ml::KNearest> model, const cv::Mat& image);
