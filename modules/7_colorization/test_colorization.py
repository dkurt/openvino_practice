#!/usr/bin/env python3
import os
import cv2 as cv
from colorization import Colorizer

data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Read input image
img_path = os.path.join(data_folder, 'conference.png')
grayscale = cv.imread(img_path)

# Create a model object
xml_path = os.path.join(data_folder, 'deoldify.xml')
bin_path = os.path.join(data_folder, 'deoldify.bin')
model = Colorizer(xml_path, bin_path)

# Run an algorithm
colored = model.colorize(grayscale)

# Uncomment to see results
# cv.imshow('Colorization', colored)
# cv.waitKey()

# Compare output with reference image
ref_path = os.path.join(data_folder, 'conference_color.png')
ref = cv.imread(ref_path)

assert(cv.norm(colored, ref, cv.NORM_INF) <= 1)
