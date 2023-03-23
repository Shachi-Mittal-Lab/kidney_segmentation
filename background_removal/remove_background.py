# import time
import os

# import shutil
import cv2
from cv2 import imread

import torch
import torchvision.transforms as T
import torch.utils.data as data
import torch.nn as nn

# import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np

from dataloader import ImageFolderWithPaths
from patch_utils import extract_tiles, stitch_img
from predict import predict

## Inputs ##

MODEL_DIRECTORY = (
    "/home/riware/Documents/Mittal_Lab/kidney_project/foreground_background/models"
)
model = "modelF8"
DATASET_DIRECTORY = (
    "/home/riware/Documents/Mittal_Lab/kidney_project/foreground_background/datasets"
)
dataset = "VF8_altered"
epoch = "3"
batchsize = 8

x_tile_size = 100  # trained on 100
y_tile_size = 100  # trained on 100
threshold = 60  # allows for 60% background
color_delta = 40  # defines distance from 255 considered background

# File structure required
"""
|-- models_location
|  |-- starting_model.pt
|  |-- model#
|  |  |-- train
|  |  |  |--model#_epoch#.pt 
|  |  |-- test
|-- dataset location
|  |-- datset#
|  |  |-- patches_for_fmodel_dataset#
|  |  |-- test_patches_for_fmodel_dataset#
|  |  |-- test_ROIs_for_fmodel_dataset#
"""

WEIGHTS_FILE = os.path.join(MODEL_DIRECTORY, f"{model}/train/{model}_epoch{epoch}.pt")
MODEL_FILE = os.path.join(MODEL_DIRECTORY, "starting_model.pt")
DATA_PATH = os.path.join(
    DATASET_DIRECTORY, f"{dataset}/test_patches_for_fmodel_{dataset}"
)
RESULTS_PATH = os.path.join(MODEL_DIRECTORY, f"{model}/test")

# Use gpu if available
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
    checkpoint = torch.load(WEIGHTS_FILE)
    trained_VGG16 = torch.load(MODEL_FILE)
    trained_VGG16.load_state_dict(checkpoint["model_state_dict"])
else:
    print("Using CPU")
    checkpoint = torch.load(WEIGHTS_FILE, map_location=torch.device("cpu"))
    trained_VGG16 = torch.load(MODEL_FILE, map_location=torch.device("cpu"))
    trained_VGG16.load_state_dict(checkpoint["model_state_dict"])

# Resize image, convert to tensor: values between 0-1
transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
    ]
)

# Initialize dataloader
test_data = ImageFolderWithPaths(root=DATA_PATH, transform=transform)
data_size = len(test_data)
class_names = test_data.classes
print(test_data.class_to_idx)
classes = [key for key in test_data.class_to_idx]

# Create ROIs folder if it does not exist
ROI_DIRECTORY = os.path.join(
    DATASET_DIRECTORY, dataset, f"test_ROIs_for_fmodel_{dataset}"
)
ROI_PREDS_DIRECTORY = os.path.join(MODEL_DIRECTORY, model, "test", "ROIs")

if os.path.exists(ROI_PREDS_DIRECTORY):
    pass
else:
    os.mkdir(ROI_PREDS_DIRECTORY)

for roi in os.listdir(ROI_DIRECTORY):
    print(roi)
    # Make directory for case in predictions directory if it does not exist
    case = os.path.basename(roi)
    CASE_PRED_DIRECTORY = os.path.join(ROI_PREDS_DIRECTORY, case)
    if os.path.exists(CASE_PRED_DIRECTORY):
        pass
    else:
        os.mkdir(CASE_PRED_DIRECTORY)

    # Read Image
    roi_path = os.path.join(ROI_DIRECTORY, roi)
    tiff_file = imread(roi_path)
    tiff_file = cv2.cvtColor(tiff_file, cv2.COLOR_BGR2RGB)
    image_array = np.asarray(tiff_file)
    roi_size_x = image_array.shape[1]
    roi_size_y = image_array.shape[0]

    # Create folder for ROI in case folder
    ROI_PRED_DIRECTORY = os.path.join(
        CASE_PRED_DIRECTORY, f"roi_patch{x_tile_size}x{y_tile_size}"
    )
    os.mkdir(ROI_PRED_DIRECTORY)
    IMG_DIRECTORY = os.path.join(ROI_PRED_DIRECTORY, "images")
    os.mkdir(IMG_DIRECTORY)

    # Break up ROI into tiles to give to model
    x_tile_num, y_tile_num = extract_tiles(
        case=case,
        image_array=image_array,
        IMG_DIRECTORY=IMG_DIRECTORY,
        x_tile_size=x_tile_size,
        y_tile_size=y_tile_size,
        threshold=threshold,
        color_delta=color_delta,
    )

    pred_data = ImageFolderWithPaths(root=ROI_PRED_DIRECTORY, transform=transform)
    data_size = len(pred_data)
    pred_loader = data.DataLoader(
        pred_data, batch_size=batchsize, shuffle=False, num_workers=2
    )

    print("Starting predict!")
    img_files, predictions = predict(trained_VGG16, pred_loader)
    print("Predicted!")

    stitch_img(
        x_tile_num,
        y_tile_num,
        x_tile_size,
        y_tile_size,
        classes,
        img_files,
        IMG_DIRECTORY,
        ROI_PRED_DIRECTORY,
        predictions,
    )
