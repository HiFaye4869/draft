# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 20:22:34 2023

@author: Christine
"""

import os
from PIL import Image
import numpy as np

# Define the path to the patches and output directories
patches_dir = './debug_patch'
output_dir = 'mask_output'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the list of all patches
patches = os.listdir(patches_dir)

# Get the number of images
n_images = max(int(patch.split('_')[0]) for patch in patches) + 1

p = 80  # replace with the actual size

# Loop over each image
for i in range(n_images):
    # Initialize an empty image
    image = np.zeros((4*p, 4*p), dtype=np.uint8)

    # Loop over each patch in the image
    for j in range(16):
        # Construct the patch filename
        patch_filename = f'{i}_{j}.jpg'

        # If the patch exists, set the corresponding patch in the output image to white
        if patch_filename in patches:
            row = (j // 4) * p
            col = (j % 4) * p
            image[row:row+p, col:col+p] = 255

    # Convert the image to a PIL Image object
    image = Image.fromarray(image)

    # Save the image
    image.save(os.path.join(output_dir, f'{i}.jpg'))