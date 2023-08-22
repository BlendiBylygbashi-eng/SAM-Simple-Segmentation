pip install git+https://github.com/facebookresearch/segment-anything.git

import os
import cv2
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np

from PIL import Image
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor
from torch.utils.data import Dataset, DataLoader
import glob

sam_checkpoint = '/content/drive/MyDrive/sam_vit_l_0b3195.pth'
model_type = "vit_l"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam)

from PIL import Image

# Load the image
image_path = "/content/drive/MyDrive/traffic.webp"
pil_image = Image.open(image_path)

# Resize the image while preserving aspect ratio
desired_size = 256
aspect_ratio = pil_image.width / pil_image.height
resized_image = pil_image.resize((int(desired_size * aspect_ratio), desired_size))

# Convert the resized image to a NumPy array
image_array = np.array(resized_image)

# Generate masks
masks = mask_generator.generate(image_array)

import random
import numpy as np

def show_anns(anns, axes=None):
    if len(anns) == 0:
        return
    if axes:
        ax = axes
    else:
        ax = plt.gca()
        ax.set_autoscale_on(False)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m**0.5)))

# Plot the original image and the masks
fig, axs = plt.subplots(1, 2, figsize=(16, 16))
axs[0].imshow(image_array)
axs[1].imshow(image_array)
show_anns(masks, axs[1])
axs[0].axis('off')
axs[1].axis('off')
plt.show()
