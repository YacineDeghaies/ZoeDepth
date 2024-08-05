from torchvision import transforms
import random
import torch
import torch.nn as nn
import torch.optim as optim
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def train_preprocess(image, depth_gt, augment=True):
    if augment:
        # Ensure depth_gt is 3D
        if depth_gt.ndim == 2:
            depth_gt = np.expand_dims(depth_gt, axis=2)
        
        # Ensure depth_gt has 3 channels to match image channels if required
        if depth_gt.shape[2] == 1:
            depth_gt = np.repeat(depth_gt, 3, axis=2)
        
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[::-1, :, :]).copy()
            depth_gt = (depth_gt[::-1, :, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = augment_image(image)

    return image, depth_gt

def augment_image(image):
    # Placeholder for actual augmentation logic
    # Example: apply random gamma correction
    gamma = random.uniform(0.9, 1.1)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Example usage
image = cv2.imread("/vol/fob-vol3/mi20/deghaisa/code/shot_0003/1_source_sequence/shot_0003_source_0000.png", cv2.IMREAD_UNCHANGED)
depth_gt = cv2.imread("/vol/fob-vol3/mi20/deghaisa/code/shot_0003/2_gt_depth/shot_0003_dm_gt_s_0000.png", cv2.IMREAD_UNCHANGED)

# Test the preprocessing
image_aug, depth_gt_aug = train_preprocess(image, depth_gt)
print("Augmented depth_gt shape:", depth_gt_aug.shape)
