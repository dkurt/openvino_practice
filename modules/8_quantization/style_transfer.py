import time
import numpy as np
import cv2 as cv
from openvino.inference_engine import IECore

# This class runs style transfer model (https://github.com/pytorch/examples/tree/master/fast_neural_style)
# using OpenVINO
class StyleTransfer():

    # Constructor
    def __init__(self, xml_path, bin_path):
        self.ie = IECore()
        self.net = self.ie.read_network(xml_path, bin_path)
        l = self.net.layers  # Do not remove! This is a workaround for reshape bug


    # Performs style transfer
    def process(self, img):
        # Reshape network to work with origin image size
        h, w = img.shape[0], img.shape[1]
        self.net.reshape({'input': [1, 3, h, w]})

        inp = self._preprocess(img)
        exec_net = self.ie.load_network(self.net, 'CPU')

        start = time.time()
        out = exec_net.infer({'input': inp})
        print('Processing time: %.2f seconds' % (time.time() - start))

        return self._postprocess(out['output'])


    # Performs image preprocessing
    # 1. BGR to RGB conversion
    # 2. Transpose from HWC to NCHW layout
    # 3. Convert to FP32
    def _preprocess(self, img):
        print('_preprocess is not implemented')
        exit(1)


    # Performs output postprocessing
    # 1. Transpose from NCHW to HWC
    # 2. Clip values to [0, 255] range
    # 3. Convert to U8
    # 4. Convert from RGB to BGR
    def _postprocess(self, out):
        print('_postprocess is not implemented')
        exit(1)
