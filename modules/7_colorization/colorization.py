import numpy as np
import cv2 as cv
from openvino.inference_engine import IECore

# This class runs DeOldify (https://github.com/jantic/DeOldify/)
# image colorization network using OpenVINO
class Colorizer():

    # Constructor
    def __init__(self, xml_path, bin_path):
        print('constructor is not implemented')
        exit(1)

    # Performs grayscale image colorization. Returns BGR image.
    # [inp] img - numpy array which contains image data (after cv.imread)
    def colorize(self, img):
        print('colorize is not implemented')
        exit(1)
    
    # Performs network's output postprocessing:
    # 1. Scale by channels with (58.395, 57.12, 57.375)
    # 2. Add by channels (123.675, 116.28, 103.53)
    # 3. Transpose array from [1 x 3 x 480 x 640] to [480 x 640 x 3]
    # 4. Convert from RGB to BGR color space
    # 5. Clip values to range [0, 255]
    # 6. Convert FP32 to U8 data type
    # 7. Resize to origin image's size
    def postprocess(self, out, height, width):
        print('postprocess is not implemented')
        exit(1)

