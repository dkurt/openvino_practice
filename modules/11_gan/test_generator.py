#!/usr/bin/env python3
import os
import cv2 as cv
import numpy as np
from generator import Generator

data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Create a model object
xml_path = os.path.join(data_folder, 'cifar_generator.xml')
bin_path = os.path.join(data_folder, 'cifar_generator.bin')
model = Generator(xml_path, bin_path)

# Run the algorithm

grid_size = 6
img_size = 64
all_images = np.zeros((img_size * grid_size, img_size * grid_size, 3), dtype=np.uint8)

for i in range(grid_size * grid_size):
    generated = model.generate()
    row = i // grid_size
    col = i % grid_size
    all_images[img_size * row : img_size * (row + 1),
               img_size * col : img_size * (col + 1), :] = generated

cv.imwrite('../../data/generated_img.png', all_images)
# Uncomment to see results
# cv.imshow('Generated', all_images)
# cv.waitKey()
