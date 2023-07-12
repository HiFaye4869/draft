# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 19:42:50 2023

@author: Christine
"""

import os
import torch
from torchvision import utils as vutils
import argparse
import albumentations
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import math
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def get_patches(args):
    os.makedirs(args.output_dir, exist_ok=True)

    test_dataset = ImagePaths(args.dataset_path, size=args.image_size)
    #test_dataset.images.sort(key = lambda x: int(x.replace('wave/','')[:-4]))  # Sort the image paths; replace the non-digit parts with '' to sort them

    with torch.no_grad():
        for i, img_path in enumerate(test_dataset.images):
            img = test_dataset.preprocess_image(img_path)
            img = torch.from_numpy(img).unsqueeze(0).to(device=args.device)
            print(img.shape)
            
            patch_size = 64
            stride = 64
            
            # Unfold the image tensor to extract patches
            patches = img.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
            num_patches = patches.size(2) * patches.size(3)
            
            # Reshape the patches tensor to [N, C, H, W]
            patches = patches.contiguous().view(1, 3, -1, patch_size, patch_size)
            #print(patches.shape)
            # Save the patches as JPEG images
            for j in range(num_patches):
                # Extract a single patch
                patch = patches[:, :, j, :, :]
            
                # Convert the patch tensor to PIL image
                patch_image = TF.to_pil_image(patch.squeeze())
                
                image_path = os.path.join(args.output_dir, f"{i}_{j}.png")
            
                # Save the patch image as JPEG (f"{image_path}/image.png")
                patch_image.save(f'{image_path}')
    num_patches = num_patches * args.num_images
    print(f"{num_patches} patches have been generated and saved to {args.output_dir}.")
'''
def patches_to_image(args):
    os.makedir(args.output_back, exist_ok=True)
    patch_filenames = ImagePaths(args.output_dir, size=args.image_size)
    # Parameters for patch size and stride
    patch_size = 64
    stride = 64
    
    # Calculate the number of patches in each dimension
    image_width = 256
    image_height = 256
    num_patches_w = (image_width - patch_size) // stride + 1
    num_patches_h = (image_height - patch_size) // stride + 1
    
    # Create an empty tensor to store the combined image
    combined_image = torch.zeros(1, 3, image_height, image_width)
    
    # Loop through each patch and combine it into the image tensor
    for i, patch_filename in enumerate(patch_filenames):
        # Load the patch image
        patch_image = Image.open(patch_filename)
    
        # Convert the patch image to a PyTorch tensor
        patch = TF.to_tensor(patch_image)
    
        # Calculate the row and column indices of the patch in the image tensor
        row_idx = i // num_patches_w
        col_idx = i % num_patches_w
    
        # Calculate the coordinates of the patch in the image tensor
        start_h = row_idx * stride
        start_w = col_idx * stride
        end_h = start_h + patch_size
        end_w = start_w + patch_size
    
        # Combine the patch into the image tensor
        combined_image[:, :, start_h:end_h, start_w:end_w] = patch
    
    # Convert the combined image tensor to a PIL image
    combined_image = TF.to_pil_image(combined_image.squeeze())
    
    # Save the combined image as JPEG
    combined_image.save('combined_image.jpg')
'''

def patch_to_image(args):
    
    # Directory path where the patches are stored
    patches_directory = args.output_dir
    
    # Parameters for patch size and stride
    patch_size = 64
    stride = 64
    num_rows = args.image_size // patch_size
    num_cols = num_rows
    
    # Create a dictionary to store patches for each image
    image_patches = {}
    
    # List all files in the patches directory
    patch_filenames = os.listdir(patches_directory)
    
    for patch_filename in patch_filenames:
        # Load the patch image
        patch_image = Image.open(os.path.join(patches_directory, patch_filename))
    
        # Convert the patch image to a PyTorch tensor
        patch = TF.to_tensor(patch_image)
    
        # Extract the indices from the patch filename
        indices = patch_filename.split('.')[0].split('_')
        #print(indices[0])
        image_idx = int(indices[0])
        patch_idx = int(indices[1])
    
        # Check if the image index is already in the dictionary
        if image_idx in image_patches:
            # Append the patch to the existing image's patch list
            image_patches[image_idx].append((patch, patch_idx))
        else:
            # Create a new patch list for the image index
            image_patches[image_idx] = [(patch, patch_idx)]
    
    # Combine patches for each image into complete images
    for image_idx, patches in image_patches.items():
        # Create an empty tensor to store the combined image
        combined_image = torch.zeros(1, 3, num_rows * patch_size, num_cols * patch_size)
    
        # Loop through each patch and place it in the original location in the image tensor
        for patch, patch_idx in patches:
            # Calculate the row and column indices of the patch in the image tensor
            row_idx = patch_idx // num_cols
            col_idx = patch_idx % num_cols
    
            # Calculate the coordinates of the patch in the image tensor
            start_h = row_idx * patch_size
            start_w = col_idx * patch_size
            end_h = start_h + patch_size
            end_w = start_w + patch_size
    
            # Place the patch in the original location in the image tensor
            combined_image[:, :, start_h:end_h, start_w:end_w] = patch
    
        # Convert the combined image tensor to a PIL image
        combined_image = TF.to_pil_image(combined_image.squeeze())
        
        os.makedirs(args.output_back, exist_ok=True)
        image_path = os.path.join(args.output_back, f"{image_idx}.png")
        combined_image.save(f'{image_path}')
    print(f"{args.num_images} images have been reconstructed and saved to {args.output_back}.")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Images using Trained VQGAN")
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the generation is on')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size for generating images (default: 20)')
    
    parser.add_argument('--num-images', type=int, default=10, help='Number of images to generate (default: 10)')
    parser.add_argument('--dataset-path', type=str, default='./input', help='Path to the folder of input images')
    parser.add_argument('--output-dir', type=str, default='generated_images',
                        help='Path to the directory for saving generated images (default: generated_images)')
    parser.add_argument('--output-back', type=str, default='image_back',
                        help='Path to the directory for saving generated images (default: generated_images)')
    args = parser.parse_args()

    get_patches(args)
    patch_to_image(args)