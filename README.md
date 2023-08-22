# Segment Anything Model (SAM) Implementation
Leverage the power of Meta's Segment Anything Model (SAM) to easily segment objects in any image.

> **Recommended Environment:** This project is optimized for Google Colab. For best results, run the provided scripts within a Colab notebook, and replace any instance of Google Drive paths with your own paths.

## Overview

The Segment Anything Model (SAM) is a state-of-the-art model from Meta that has demonstrated remarkable capabilities in object segmentation. This repository offers a hassle-free way to test SAM on any image of your choice.

![Sample Segmentation](<https://user-images.githubusercontent.com/78195053/262208356-df430dde-6bbe-4848-ac79-c7846c143ae4.png>)

> **Example of SAM in action. Left: Original Image, Right: Segmented Image**

---

## Model Checkpoint

The SAM model requires a pre-trained checkpoint to run. Due to the large size of the checkpoint, it is not hosted directly in this repository. Instead, you can download it from Hugging Face.

### Steps to Download the Checkpoint:

1. Visit the SAM model page on Hugging Face [here](https://huggingface.co/spaces/facebook/ov-seg/blob/f9b1bcfebfafe86b45b0cf16a1797ca5663d81af/sam_vit_l_0b3195.pth).
2. Download the `.pth` checkpoint file.
3. Once downloaded, this can be uploaded to your Google Drive and then used in the same way as shown in the script (using your own path).
