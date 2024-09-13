from tkinter import CENTER
import numpy as np
import os

import tifffile as tiff
from sklearn.cluster import KMeans
from shutil import copy, rmtree
from tqdm import tqdm

from patch_utils import extract_tiles, extract_tiles_original, stitch_img
from predict import predict
from dataloader import ImageFolderWithPaths

from style import histoverlap, histmax

import cv2
from scipy.ndimage.measurements import label

import torch
import torchvision.transforms as T
import torch.utils.data as data
import torch.nn as nn

# Cluster Model Inputs
#PREDICT_DIR = "/home/riware/Desktop/fibrosis_score/10_AB_fibscores/ROIs_original"
#PREDICT_DIR = "/home/riware/Documents/MittalLab/fibrosis_score/10_AB_fibscores/manualstructureremoval/original_rois"
PREDICT_DIR = "/home/riware/Documents/MittalLab/kidney_project/fibrosis_score/fib_seg/clustering/regions_for_drsetty/finalrois"
#CENTERS_FILE = "/home/riware/Documents/MittalLab/Mittal_Lab_Repository/Mittal_Lab_Repository/kidney_project/ndpi_tools/cluster/average_centers_4.txt"
CENTERS_FILE = "/home/riware/Documents/MittalLab/kidney_project/fibrosis_score/fib_seg/superpatches/superpatches_v5/average_centers_4.txt"
DIRECTORY = "/home/riware/Documents/MittalLab/kidney_project/fibrosis_score/fib_seg/clustering/regions_for_drsetty/finalrois"
CENTERS_FILE_COLLAGEN = "/home/riware/Documents/MittalLab/kidney_project/fibrosis_score/10_AB_fibscores/clustering/collagen_cluster_centers_investigation/old_annotations/all_superpatches/average_centers_5_collagen.txt"

# VGG16 4 Class Model Inputs
MODEL_4CLASS_DIR = "/home/riware/Documents/MittalLab/kidney_project/VGG16_model_data"
model_4class = "model22"
DATASET_4CLASS_DIR = "/home/riware/Documents/MittalLab/kidney_project/VGG16_datasets"
dataset_4class = "V29"
epoch_4class = "22"
batchsize = 8
x_tile_size = 48
y_tile_size = 48
threshold = 60 #20
color_delta = 60

# VGG16 Fibrosis Model Inputs
MODEL_FIB_DIR = "/home/riware/Documents/MittalLab/kidney_project/fibrosis_score/VGG16_fib_models"
model_fib = "model26"
DATASET_FIB_DIR = "/home/riware/Documents/MittalLab/kidney_project/fibrosis_score/fibrosis_datasets"
dataset_fib = "V34_altered"
epoch_fib = "27"
batchsize = 8
x_tile_size = 48
y_tile_size = 48
threshold = 60 #20

# Get cluster centers
centers = np.loadtxt(CENTERS_FILE)
n_clusters = len(centers)

centers_collagen = np.loadtxt(CENTERS_FILE_COLLAGEN)
n_clusters_collagen = len(centers_collagen)

# Assign cluster centers
clustering = KMeans(n_clusters=n_clusters)
clustering.fit(centers)

clustering_collagen = KMeans(n_clusters=n_clusters_collagen)
clustering_collagen.fit(centers_collagen)

# Create image list
images = [
    os.path.join(PREDICT_DIR, img)
    for img in os.listdir(PREDICT_DIR)
    if img.lower().endswith((".tif", ".tiff"))
]

# Grab centers file name
centers_filename = os.path.splitext(os.path.basename(CENTERS_FILE))[0]

# Iterate through images
for img in tqdm(images):

    # Make new folder for images
    predict_filename = os.path.splitext(os.path.basename(img))[0]
    OUTPUT_DIR = os.path.join(DIRECTORY, f"cluster_{predict_filename}")
    CLUSTER_DIR = os.path.join(OUTPUT_DIR, f"{n_clusters}_clusters")
    if os.path.exists(OUTPUT_DIR):
        if os.path.exists(CLUSTER_DIR):
            print(f"Cluster of {predict_filename} previously performed.")
            pass
        else:
            os.mkdir(CLUSTER_DIR)
    else:
        os.mkdir(OUTPUT_DIR)
        os.mkdir(CLUSTER_DIR)

    # Copy centers file to output folder
    copy(CENTERS_FILE, CLUSTER_DIR)
    copy(CENTERS_FILE_COLLAGEN, CLUSTER_DIR)

    # Determine image style
    imageGBR = cv2.imread(img)
    imageRGB = cv2.cvtColor(imageGBR, cv2.COLOR_BGR2RGB)

    blue_color = cv2.calcHist([imageRGB], [0], None, [256], [0, 256])
    red_color = cv2.calcHist([imageRGB], [1], None, [256], [0, 256])
    rboverlap = histoverlap(red_color, blue_color)

    gray = cv2.cvtColor(imageGBR, cv2.COLOR_BGR2GRAY)
    grayhist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    graymax = histmax(grayhist, 200)

    # Calc darkness category
    if graymax < 50:
        staindarkness = "dark"
    elif graymax > 70:
        staindarkness = "light"
    else: 
        staindarkness = "medium"

    # Calc stain darkness category
    if rboverlap < 0.25:
        staincontrast = "high"
    elif rboverlap > 0.32:
        staincontrast= "low"
        adjusted = cv2.convertScaleAbs(imageGBR, alpha=1.5, beta=-50)
        imageRGB = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
        adjusted_img = os.path.join(CLUSTER_DIR, "adjusted_img.tiff")
        tiff.imwrite(adjusted_img, imageRGB, photometric="rgb")
    else: 
        staincontrast = "medium"

    # Predict
    predict_image_array = imageRGB
    flattened_predict_array = predict_image_array.reshape(-1, 3)
    prediction = clustering.predict(flattened_predict_array)

    # Order centers (order random by default)
    index = np.arange(0, n_clusters, 1).reshape(-1, 1)
    centers = np.hstack((index, clustering.cluster_centers_))
    ascending_centers = np.array(sorted(centers, key=lambda x: x[1]))
    ordered_centers = ascending_centers[::-1]
    new_order = ordered_centers[:, 0].astype(int).reshape(-1, 1)
    mapping = np.array(
        [x[1] for x in sorted(zip(new_order, index))], dtype=int
    ).reshape(-1, 1)
    prediction_labels = np.uint8(
        ordered_centers[:, 1:][mapping[prediction]].reshape(predict_image_array.shape)
    )

    # Cluster image
    tiff.imwrite(
        os.path.join(CLUSTER_DIR, "prediction.tiff"),
        prediction_labels,
        photometric="rgb",
    )

    for i in range(n_clusters):
        cluster = np.uint8(
            (mapping[prediction] == i).reshape(*predict_image_array.shape[:2], 1)
            * predict_image_array
        )
        tiff.imwrite(
            os.path.join(CLUSTER_DIR, f"cluster{i}.tiff"), cluster, photometric="rgb"
        )

    # collagen
    collagen = np.uint8(
        (mapping[prediction] == 1).reshape(*predict_image_array.shape[:2], 1)
        * predict_image_array
    ) + np.uint8(
        (mapping[prediction] == 2).reshape(*predict_image_array.shape[:2], 1)
        * predict_image_array
    )
    collagen_img = os.path.join(CLUSTER_DIR, f"collagen_{predict_filename}.tiff")
    tiff.imwrite(collagen_img, collagen, photometric="rgb")

    # size filter 
    kernel = np.ones((3, 3), dtype=int)

    # connected components
    collagen_label = (np.isin(mapping[prediction], [1,2])).reshape(*predict_image_array.shape[:2], 1)[:,:,0]
    labeled_components, ncomponents = label(collagen_label, kernel)
    unique, counts = np.unique(labeled_components, return_counts=True)
    for i, count in enumerate(counts): 
        if count < 200000: 
            collagen_label[labeled_components==i] = 0
        else:
            pass 
    collagen_filtered = np.zeros(collagen_label.shape)
    collagen_filtered = np.where(collagen_label > 0, 1, 0)
    collagen_filtered = np.uint8(collagen_filtered.reshape(*collagen_filtered.shape,1) * collagen)
    collagen_filtered_img = os.path.join(CLUSTER_DIR, f"collagen_filtered_{predict_filename}.tiff")
    tiff.imwrite(collagen_filtered_img, collagen_filtered, photometric="rgb")
    
    # further cluster collagen image
    # Predict
    predict_image_array = tiff.TiffFile(collagen_img).asarray()
    flattened_predict_array = predict_image_array.reshape(-1, 3)
    prediction_collagen = clustering_collagen.predict(flattened_predict_array)
    
    # Order centers (order random by default)
    index_collagen = np.arange(0, n_clusters_collagen, 1).reshape(-1, 1)
    centers_collagen = np.hstack((index_collagen, clustering_collagen.cluster_centers_))
    ordered_centers_collagen = np.array(sorted(centers_collagen, key=lambda x: x[1]))
    new_order_collagen = ordered_centers_collagen[:, 0].astype(int).reshape(-1, 1)
    mapping_collagen = np.array(
        [x[1] for x in sorted(zip(new_order_collagen, index_collagen))], dtype=int
    ).reshape(-1, 1)
    prediction_labels_collagen = np.uint8(
        ordered_centers_collagen[:, 1:][mapping_collagen[prediction_collagen]].reshape(predict_image_array.shape)
    )

    # Cluster image
    tiff.imwrite(
        os.path.join(CLUSTER_DIR, "prediction_collagen.tiff"),
        prediction_labels_collagen,
        photometric="rgb",
    )

    for i in range(n_clusters_collagen):
        cluster = np.uint8(
            (mapping_collagen[prediction_collagen] == i).reshape(*predict_image_array.shape[:2], 1)
            * predict_image_array
        )
        tiff.imwrite(
            os.path.join(CLUSTER_DIR, f"cluster_collagen{i}.tiff"), cluster, photometric="rgb"
        )
    
    # remove tubules (this feature doesn't work well)
    remove_tubules = np.uint8(
        (mapping_collagen[prediction_collagen] == 3).reshape(*predict_image_array.shape[:2], 1)
        * predict_image_array
    ) + np.uint8(
        (mapping_collagen[prediction_collagen] == 4).reshape(*predict_image_array.shape[:2], 1)
        * predict_image_array
    )

    no_tubules_img = os.path.join(CLUSTER_DIR, f"no_tubules_{predict_filename}.tiff")
    tiff.imwrite(no_tubules_img, remove_tubules, photometric="rgb")
    
    # predict on collagen image with four-class model_4class
    WEIGHTS_FILE_4CLASS = os.path.join(
        MODEL_4CLASS_DIR, f"{model_4class}/train/{model_4class}_epoch{epoch_4class}.pt"
    )
    WEIGHTS_FILE_FIB = os.path.join(
        MODEL_FIB_DIR, f"{model_fib}/train/{model_fib}_epoch{epoch_fib}.pt")

    MODEL_FILE = os.path.join(MODEL_4CLASS_DIR, "starting_model.pt")
    MODEL_FILE_FIB = os.path.join(MODEL_FIB_DIR, "starting_model.pt")

    DATA_PATH = os.path.join(
        DATASET_4CLASS_DIR, f"{dataset_4class}/test_patches_for_fmodel_{dataset_4class}"
    )

    DATA_PATH_FIB = os.path.join(
    DATASET_FIB_DIR, f"{dataset_fib}/test_patches_for_fmodel_{dataset_fib}"
    )

    ROI_PATCHES_DIR = os.path.join(CLUSTER_DIR, "patches")
    ROI_TILE_DIR = os.path.join(ROI_PATCHES_DIR, f"tiles_{x_tile_size}-{y_tile_size}")
    os.mkdir(ROI_PATCHES_DIR)
    os.mkdir(ROI_TILE_DIR)

    # Grab dimensions
    roi_size_x = predict_image_array.shape[1]
    roi_size_y = predict_image_array.shape[0]

    # Break up ROI into tiles to give to model_4class (used imageRGB before)
    x_tile_num, y_tile_num = extract_tiles(
        case=predict_filename,
        image_array=collagen_filtered,
        IMG_DIRECTORY=ROI_TILE_DIR,
        x_tile_size=x_tile_size,
        y_tile_size=y_tile_size,
        threshold=threshold,
        color_delta=color_delta
    )

    # Use gpu if available
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
        checkpoint_4class = torch.load(WEIGHTS_FILE_4CLASS)
        trained_VGG16_4class = torch.load(MODEL_FILE)
        trained_VGG16_4class.load_state_dict(checkpoint_4class["model_state_dict"])

        checkpoint_fib = torch.load(WEIGHTS_FILE_FIB)
        trained_VGG16_fib = torch.load(MODEL_FILE_FIB)
        trained_VGG16_fib.load_state_dict(checkpoint_fib["model_state_dict"])

    else:
        print("Using CPU")
        checkpoint_4class = torch.load(WEIGHTS_FILE_4CLASS, map_location=torch.device("cpu"))
        trained_VGG16_4class = torch.load(MODEL_FILE, map_location=torch.device("cpu"))
        trained_VGG16_4class.load_state_dict(checkpoint_4class["model_state_dict"])

        checkpoint_fib = torch.load(WEIGHTS_FILE_FIB, map_location=torch.device("cpu"))
        trained_VGG16_fib = torch.load(MODEL_FILE_FIB, map_location=torch.device("cpu"))
        trained_VGG16_fib.load_state_dict(checkpoint_fib["model_state_dict"])

    
    # Resize image, convert to tensor: values between 0-1
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )

    # Initialize dataloader
    pred_data = ImageFolderWithPaths(root=ROI_PATCHES_DIR, transform=transform)
    data_size = len(pred_data)
    class_names = pred_data.classes
    classes = [key for key in pred_data.class_to_idx]

    pred_loader = data.DataLoader(
        pred_data, batch_size=batchsize, shuffle=False, num_workers=2
    )
    """
  
    img_files, predictions_4class = predict(trained_VGG16_4class, pred_loader)
    print(" \n Predicted with 4 class model!")

    img_files1, predictions_fib = predict(trained_VGG16_fib, pred_loader)
    print(" \n Predicted with fibrosis model!")

    ROI_PRED_PATCHES_DIR = os.path.join(CLUSTER_DIR, "mask")

    # Create ROIs folder if it does not exist
    stitch_img(
        x_tile_num,
        y_tile_num,
        x_tile_size,
        y_tile_size,
        classes,
        img_files,
        CLUSTER_DIR,
        predictions_4class,
        model="4classmodel"
    )

    # make fibrosis mask
    stitch_img(
        x_tile_num,
        y_tile_num,
        x_tile_size,
        y_tile_size,
        classes,
        img_files,
        CLUSTER_DIR,
        predictions_fib,
        model="fibmodel"
    )   
    """
    # predict on entire image with background threshold
    # tile image_RGB

    # Break up ROI into tiles to give to model_4class (used imageRGB before)
    x_tile_num_original, y_tile_num_original = extract_tiles_original(
        case=predict_filename,
        image_array=imageRGB,
        IMG_DIRECTORY=ROI_TILE_DIR,
        x_tile_size=x_tile_size,
        y_tile_size=y_tile_size,
        threshold=threshold,
        color_delta=color_delta
    )

   # Initialize dataloader
    pred_data_original = ImageFolderWithPaths(root=ROI_PATCHES_DIR, transform=transform)
    data_size_original = len(pred_data_original)
    class_names_original = pred_data_original.classes
    classes_original = [key for key in pred_data_original.class_to_idx]

    pred_loader_original = data.DataLoader(
        pred_data_original, batch_size=batchsize, shuffle=False, num_workers=2
    )
  
    img_files, predictions_4class = predict(trained_VGG16_4class, pred_loader_original)
    print(" \n Predicted with 4 class model on full image!")

    model = "4classmodel_originalimage"
    stitch_img(
        x_tile_num_original,
        y_tile_num_original,
        x_tile_size,
        y_tile_size,
        classes_original,
        img_files,
        CLUSTER_DIR,
        predictions_4class,
        model=model
        )
    
    pred_mask = os.path.join(CLUSTER_DIR, f"mask_finished_{model}.tiff")
    full_pred_mask_GBR = cv2.imread(pred_mask)
    full_pred_mask_RGB = cv2.cvtColor(imageGBR, cv2.COLOR_BGR2RGB)
    
    just_gloms = np.zeros(full_pred_mask_RGB.shape)
    #just_gloms = full_pred_mask_RGB[]

