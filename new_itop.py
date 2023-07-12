# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:20:31 2023

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
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

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

    
def split_tensor(tensor, tile_size=540):
    mask = torch.ones_like(tensor)
    # use torch.nn.Unfold
    stride  = tile_size
    unfold  = nn.Unfold(kernel_size=(tile_size, tile_size), stride=stride)
    # Apply to mask and original image
    #print(mask.shape)
    mask_p  = unfold(mask)
    patches = unfold(tensor)
	
    patches = patches.reshape(3, tile_size, tile_size, -1).permute(3, 0, 1, 2)
    if tensor.is_cuda:
        patches_base = torch.zeros(patches.size(), device=tensor.get_device())
    else: 
        patches_base = torch.zeros(patches.size())
	
    tiles = []
    for t in range(patches.size(0)):
         tiles.append(patches[[t], :, :, :])
         
    #print(tensor.size(3))
    print(tensor.shape)
    return tiles, mask_p, patches_base, (tensor.size(2), tensor.size(3))

def rebuild_tensor(tensor_list, mask_t, base_tensor, t_size, tile_size=540):
    stride  = tile_size  
    # base_tensor here is used as a container
    for t, tile in enumerate(tensor_list):
         #print(tile.size())
         base_tensor[[t], :, :] = tile  
	 
    base_tensor = base_tensor.permute(1, 2, 3, 0).reshape(3*tile_size*tile_size, base_tensor.size(0)).unsqueeze(0)
    fold = nn.Fold(output_size=(t_size[0], t_size[1]), kernel_size=(tile_size, tile_size), stride=stride)
    # https://discuss.pytorch.org/t/seemlessly-blending-tensors-together/65235/2?u=bowenroom
    output_tensor = fold(base_tensor)/fold(mask_t)
    # output_tensor = fold(base_tensor)
    return output_tensor

def get_patches(args):
    os.makedirs(args.output_dir, exist_ok=True)

    test_dataset = ImagePaths(args.dataset_path, size=args.image_size)
    test_dataset.images.sort(key = lambda x: int(x.replace(f'{args.dataset_path}/','')[:-4]))  # Sort the image paths; replace the non-digit parts with '' to sort them

    with torch.no_grad():
        for i, img_path in enumerate(test_dataset.images):
            img = test_dataset.preprocess_image(img_path)
            #img = torch.from_numpy(img).unsqueeze(0).to(device=args.device)
            img = torch.from_numpy(img).unsqueeze(0)
            print(img.shape)
            
            patches, mask_t, base_tensor, t_size = split_tensor(img)
            print(patches[0].shape)
            # Put tiles back together
            # output_tensor = rebuild_tensor(patches, mask_t, base_tensor, t_size)
            for j in range(len(patches)):
            
                # Convert the patch tensor to PIL image
                #patch_image = TF.to_pil_image(patches[j].squeeze())
                
                image_path = os.path.join(args.output_dir, f"{i}_{j}.jpg")
                image_path1 = os.path.join(args.output_dir, f"{j}.jpg")
            
                #patches[j].save(f'{image_path}')
                vutils.save_image(patches[j][0][0], image_path1)
                vutils.save_image(patches[j][0], image_path)
            
    num_patches = args.num_images * 4
    print(f"{num_patches} patches have been generated and saved to {args.output_dir}.")
    return patches, mask_t, base_tensor, t_size
'''
def patch_to_image(args, mask_t, base_tensor, t_size):
    
    # Directory path where the patches are stored
    patches_directory = args.output_dir
    
    # Parameters for patch size and stride
    patch_size = 64
    stride = 64
    num_rows = args.image_size // patch_size
    num_cols = num_rows
    
    # Create a dictionary to store patches for each image
    image_patches = {}
    #image_patches = []

    
    # List all files in the patches directory
    patch_filenames = os.listdir(patches_directory)
    patch_filenames.sort(key = lambda x: int(x.replace('{args.output_dir}/','')[:-4]))
    for patch_filename in patch_filenames:
        # Load the patch image
        patch_image = Image.open(os.path.join(patches_directory, patch_filename))
    
        # Convert the patch image to a PyTorch tensor
        patch = TF.to_tensor(patch_image)
    
        # Extract the indices from the patch filename
        indices = patch_filename.split('.')[0].split('_')
        image_idx = int(indices[0])
        patch_idx = int(indices[1])
        #print(indices)
        #print(image_patches)
        # Check if the image index is already in the dictionary
        if image_idx in image_patches:
            # Append the patch to the existing image's patch list
            #image_patches[image_idx].append((patch, patch_idx))
            image_patches[image_idx].append((patch_idx, patch))
        else:
            # Create a new patch list for the image index
            #image_patches[image_idx] = [(patch, patch_idx)]
            image_patches[image_idx] = [(patch, patch_idx)]
            
    image_patches = list(image_patches.values())
    print(image_patches[0])
    for i in range(args.num_images):
        image_back = rebuild_tensor(image_patches[i], mask_t, base_tensor, t_size)
        combined_image = TF.to_pil_image(image_back.squeeze())
        
        os.makedirs(args.output_back, exist_ok=True)
        image_path = os.path.join(args.output_back, f"{image_idx}.jpg")
        combined_image.save(f'{image_path}')
    

    print(type(image_patches))
    # print(len(image_patches[0]))
'''
'''
def save_patches_from_image(image_path, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    image = Image.open(image_path)
    image_array = np.array(image)
    height, width, _ = image_array.shape
    tile_size = height // 4
    num_rows, num_cols = 4,4
    
    for row in range(num_rows):
        for col in range(num_cols):
            patch = image_array[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size]
            patch_image = Image.fromarray(patch)
            patch_image.save(os.path.join(output_directory, f"{row}_{col}.jpg"))

def reconstruct_image_from_patches(patch_directory, output_path):
    patch_files = os.listdir(patch_directory)
    patches = []
    max_x, max_y = 0, 0

    for patch_file in patch_files:
        if patch_file.endswith(".jpg"):
            row, col = map(int, os.path.splitext(patch_file)[0].split("_"))
            max_x = max(max_x, row)
            max_y = max(max_y, col)
            patch_path = os.path.join(patch_directory, patch_file)
            patch_image = Image.open(patch_path)
            patch_array = np.array(patch_image)
            patches.append((row, col, patch_array))

    tile_size = patches[0][2].shape[0]
    output_shape = ((max_x + 1) * tile_size, (max_y + 1) * tile_size)
    reconstructed_image = np.zeros(output_shape + (3,), dtype=np.uint8)
    
    for row in range(4):
        for col in range(4):
            patch_array = next(patch[2] for patch in patches if patch[0] == row and patch[1] == col)
            resized_patch = np.array(Image.fromarray(patch_array).resize((tile_size, tile_size)))
            reconstructed_image[row * tile_size:(row + 1) * tile_size, col * tile_size:(col + 1) * tile_size] = resized_patch

    reconstructed_image = Image.fromarray(reconstructed_image)
    reconstructed_image.save(output_path)
''' 
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
    args.image_size=2160
    '''
    mask_t, base_tensor, t_size = get_patches(args)
    patch_to_image(args, mask_t, base_tensor, t_size)
    #output_tensor = rebuild_tensor(tile_tensors, mask_t, base_tensor, t_size)
    '''
    
    test_image = 'tensor_testing/10.jpg'
    image_size = 2160
    Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    input_tensor = Loader(Image.open(test_image).convert('RGB')).unsqueeze(0).cuda()
    #print(input_tensor.shape)

    tile_tensors, mask_t, base_tensor, t_size = get_patches(args)
    #tile_tensors, mask_t, base_tensor, t_size = split_tensor(input_tensor)
    
    output_tensor = rebuild_tensor(tile_tensors, mask_t, base_tensor, t_size)
    
    os.makedirs(args.output_back, exist_ok=True)
    image_path = os.path.join(args.output_back, 'new.jpg')
    vutils.save_image(output_tensor, image_path)
    
    '''
    image_path = "tensor_testing/10.jpg"
    output_directory = "image_testing"
    output_path = "image_back/reconstructed_image.jpg"
    
    # Split the image into patches and save them
    save_patches_from_image(image_path, output_directory)
    
    # Reconstruct the original image from the patches
    reconstruct_image_from_patches(output_directory, output_path)
    '''
    
    
    
    