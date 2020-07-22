#!/usr/bin/env python3
import os
import numpy as np
import cv2 as cv
from style_transfer import StyleTransfer


def test_FP32():
    print('\nTest FP32 network')
    xml_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'candy.xml')
    bin_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'candy.bin')
    img_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tram.jpg')
    ref_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tram_candy.png')
    model = StyleTransfer(xml_path, bin_path)

    img = cv.imread(img_path)
    ref = cv.imread(ref_path)
    stylized = model.process(img)
    cv.imwrite("fp32.jpg", stylized)
    assert(cv.norm(stylized, ref, cv.NORM_INF) <= 1)


def test_INT8():
    print('\nTest INT8 network')
    xml_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'candy_int8.xml')
    bin_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'candy_int8.bin')
    img_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tram.jpg')
    ref_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tram_candy_int8.png')
    model = StyleTransfer(xml_path, bin_path)

    img = cv.imread(img_path)
    ref = cv.imread(ref_path)
    stylized = model.process(img)
    cv.imwrite("i8.jpg", stylized)
    print(cv.PSNR(stylized, ref))
    assert(cv.PSNR(stylized, ref) >= 35)


test_FP32()
test_INT8()
