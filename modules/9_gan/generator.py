import numpy as np
import cv2 as cv
from openvino.inference_engine import IECore

# This class runs a simple generator network that
# produces an image from the distribution of CelebA using OpenVINO
class Generator():
    # Constructor
    def __init__(self, xml_path, bin_path):
        print('constructor is not implemented')
        exit(1)

    # Performs generation of an image from CelebA distributiuon. Returns a BGR image.
    # Here you need to request the input shape from the loaded generator net, create a
    # normally distributed random input array of this shape via numpy,
    # feed it into the generator and call postprocess to get image
    def generate(self):
        print('generate is not implemented')
        exit(1)

    # Performs network's output postprocessing:
    # 1. Transpose array from [1 x 3 x 64 x 64] to [64 x 64 x 3]
    # 2. Add 1.0
    # 3. Scale with 0.5 * 255.0
    # 4. Convert FP32 to U8 data type
    # 5. Convert from RGB to BGR color space
    def postprocess(self, out):
        print('postprocess is not implemented')
        exit(1)
