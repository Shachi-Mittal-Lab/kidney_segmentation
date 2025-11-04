# Repo Tools
from fibrosis_score.zarr_utils import ndpi_to_zarr, omniseg_to_zarr
from fibrosis_score.mask_utils import (
    prepare_mask,
    prepare_foreground_masks,
    prepare_clustering_masks,
    prepare_seg_masks,
    prepare_postprocessing_masks,
    prepare_patch_mask,
    generate_patchcount_mask,
    generate_grid_mask,
)
from fibrosis_score.patch_utils_dask import foreground_mask
from fibrosis_score.cluster_utils import prepare_clustering, grayscale_cluster
from fibrosis_score.pred_utils import pred_cortex
from fibrosis_score.processing_utils import fill_holes, erode, varied_vessel_dilation, print_gpu_usage
from fibrosis_score.daisy_blocks import (
    model_prediction,
    model_prediction_lsds,
    model_prediction_rgb,
    upsample,
    tissuemask_upsample,
    id_tbm,
    id_bc,
    downsample_vessel,
    calculate_fibscore
)

# Funke Lab Tools
import daisy
from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi

# Tools
import os
import subprocess
import sys
from pathlib import Path

import dask
import dask.array
from dask.diagnostics import ProgressBar
from dask.array import coarsen, mean
import zarr
import zarr.convenience

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from skimage.transform import integral_image
from skimage.morphology import (
    binary_erosion,
    binary_dilation,
    disk,
    remove_small_objects,
    remove_small_holes,
)
from skimage.filters import gaussian
from scipy.ndimage import label

import torch
import torchvision.transforms as T

def unet_models_pred(
        input_path: Path,
        threshold: int,
        offset: Coordinate,
        axis_names: list,
):

    # check there is a GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device}")

    ###########
    input_filename = input_path.stem
    input_file_ext = input_path.suffix
    print(input_filename)

    # convert file to zarr if not a zarr
    if input_file_ext == ".ndpi":
        # name zarr/home/amninder/Desktop/project/Folder_2/subfolder
        zarr_path = input_path.with_suffix(".zarr")
        # convert ndpi to zarr array levels 40x, 20x, 10x, 5x, and rechunk
        ndpi_to_zarr(input_path, zarr_path, offset, axis_names)
    elif input_file_ext == ".zarr":
        zarr_path = input_path
    else:
        print(f"File must be of format .ndpi or .zarr.  File extension is {input_file_ext}. Skipping {input_filename}")
        return
    
    # grab 5x, 10x levels & calculate foreground masks
    s3_array = open_ds(zarr_path / "raw" / "s3")
    s2_array = open_ds(zarr_path / "raw" / "s2")

    print("Generating Foreground Masks")
    fmask, filled_fmask, eroded_fmask = prepare_foreground_masks(zarr_path, s3_array)
    mask = foreground_mask(s3_array.data, threshold)
    store_fgbg = zarr.open(zarr_path / "mask" / "foreground")
    dask.array.store(mask, store_fgbg)

    # fill holes & erode foreground mask
    filled_fmask._source_data[:] = fill_holes(mask, filldisk=15, shrinkdisk=10)
    eroded_fmask._source_data[:] = erode(filled_fmask._source_data[:], shrinkdisk=40)

    # create integral mask of filled foreground mask and save
    imask = prepare_mask(zarr_path, s3_array, "integral_foreground")
    integral_mask = integral_image(filled_fmask._source_data[:])
    imask._source_data[:] = integral_mask

    # create patch count mask for Cortex model
    patch_shape = Coordinate(224, 224)
    patch_count_mask, patch_size = generate_patchcount_mask(
        imask, patch_shape, tissueallowance=0.30
    )
    # generate gridmask
    patch_spacing = Coordinate(112, 112)
    gridmask = generate_grid_mask(patch_count_mask, patch_spacing)

    # create patchmask
    patch_mask = prepare_patch_mask(zarr_path, s3_array, patch_count_mask, "patch_mask")
    patch_mask._source_data[:] = patch_count_mask * gridmask

    # Generate list of coordinates of patches
    coords = np.nonzero(patch_mask._source_data[:])
    offsets = [
        patch_mask.roi.offset + Coordinate(a, b) * patch_mask.voxel_size
        for a, b in zip(*coords)
    ]

    # apply VGG16 Cortex model at 5x (s3)
    s3_array = open_ds(zarr_path / "raw" / "s3")
    pred_mask = prepare_mask(zarr_path, s3_array, "cortex")
    pred_mask[:] = 3
    pred_cortex(s3_array, offsets, patch_size, patch_spacing, pred_mask, device)
    # shave off background
    pred_mask._source_data[:] = pred_mask._source_data * filled_fmask._source_data[:]

    # create integral mask of cortex mask and save
    integral_cortex_mask = integral_image(pred_mask._source_data[:])
    icmask = prepare_mask(zarr_path, s3_array, "integral_cortex")
    icmask._source_data[:] = integral_cortex_mask

    # create patch mask for Omni-Seg
    patch_shape_final = Coordinate(512, 512)  # Will convention: shape in voxels

    patch_count_mask_final, patch_size_final = generate_patchcount_mask(
        icmask, patch_shape_final, tissueallowance=0.10
    )

    # generate gridmask
    patch_spacing_final = Coordinate(256, 256)
    gridmaskfinal = generate_grid_mask(patch_count_mask_final, patch_spacing_final)

    # create patch mask
    patch_mask_final = prepare_patch_mask(
        zarr_path, s3_array, patch_count_mask_final, "patch_mask_final"
    )
    patch_mask_final._source_data[:] = gridmaskfinal * patch_count_mask_final

    # Generate list of coordinates of patches
    coords = np.nonzero(patch_mask_final._source_data[:])
    offsets_final = [
        patch_mask_final.roi.offset + Coordinate(a, b) * patch_mask_final.voxel_size
        for a, b in zip(*coords)
    ]

    # create visualization of regions pulled
    roi_mask = prepare_patch_mask(zarr_path, s3_array, patch_count_mask, "rois")

      # grab 40x level
    s0_array = open_ds(zarr_path / "raw" / "s0")

    print("Preparing Masks")
     # prepare histological segmentation mask locations in zarr (10x)
    pt_mask_10x, dt_mask_10x, vessel_mask_10x, cap_mask_10x = prepare_seg_masks(zarr_path, s2_array, "10x")
    # prepare histological segmentation mask locations in zarr (40x)
    pt_mask, dt_mask_10x, vessel_mask, cap_mask = prepare_seg_masks(zarr_path, s0_array, "40x")

    # prepare postprocessing masks in zarr (40x)
    tbm_mask, bc_mask, fincap_mask, finfib_mask, fincollagen_mask, fincollagen_exclusion, fininflamm_mask = (
        prepare_postprocessing_masks(zarr_path, s0_array)
    )
    # prep mask for cap at 10x
    cap_mask10x = prepare_mask(zarr_path, s2_array, "cap10x")
    # create text file for fibrosis score data
    txt_path = input_path.with_suffix(".txt")
    with open(txt_path, "w") as f:
        f.writelines("ROI fibrosis scores \n")

    # begin u-net preds: define preprocessing & patch size
    inp_transforms = T.Compose(
        [
            T.Grayscale(),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
        ]
    )
    patch_shape_final = Coordinate(1056, 1056)
    patch_size_final = patch_shape_final * s2_array.voxel_size  # size in nm
    """
    print("Predicting Gloms with U-Net")
    #### Use U-net to predict gloms ####

    # load glom model
    # model = torch.load("model_dataset1_flips_eric.pt", weights_only=False) before
    print_gpu_usage(device)

    model = torch.load("model_unet_dataset11Aug2025_glom_LSDs_gaussianelastic_longer_350.pt", weights_only=False)
    binary_head = torch.load("binaryhead_unet_dataset11Aug2025_glom_LSDs_gaussianelastic_longer_350.pt", weights_only=False)
    # predict
    print_gpu_usage(device)

    model_prediction_lsds(cap_mask_10x, s2_array, patch_size_final, model, binary_head, device, "Cap ID")

    # remove small objects from cap mask
    cap_mask_10x._source_data[:] = remove_small_objects(
        cap_mask_10x._source_data[:].astype(bool), min_size=2000
    )

    print("Predicting tubules with U-Net")
    #### Use U-net to predict tubules ####

    # remove previous model from gpu
    #del model 
    torch.cuda.empty_cache()

    # load model
    model = torch.load("model_unet_dataset0_dtpt0_200.pt", weights_only=False)
    # predict
    model_prediction(pt_mask_10x, s2_array, patch_size_final, model, device, "PT ID")
    """
    print("Predicting vessels with U-Net")

    #### Use U-net to predict vessel ####

    # remove previous model from gpu
    # del model 
    # load model
    model = torch.load("model_unet_dataset11_vessel0_LSDs_400.pt", weights_only=False)
    binary_head = torch.load("binaryhead_unet_dataset11_vessel0_LSDs_400.pt", weights_only=False)
    # predict
    print_gpu_usage(device)

    model_prediction_lsds(vessel_mask_10x, s2_array, patch_size_final, model, binary_head, device, "Vessel ID")

    # model = torch.load("model_unet_dataset10_vessel4evensmallerjitter_200.pt", weights_only=False)
    # predict
    # model_prediction(vessel_mask_10x, s2_array, patch_size_final, model, device, "Vessel ID")
    """
    # remove small objects from cap mask
    cap_mask_10x._source_data[:] = remove_small_objects(
        cap_mask_10x._source_data[:].astype(bool), min_size=2000
    )
    """
    upsampling_factor = cap_mask_10x.voxel_size / fincap_mask.voxel_size
    """
    # blockwise upsample from 10x to 40x
    upsample(cap_mask_10x, fincap_mask, upsampling_factor, s2_array, s0_array)
    upsample(pt_mask_10x, pt_mask, upsampling_factor, s2_array, s0_array)
    """
    upsample(vessel_mask_10x, vessel_mask, upsampling_factor, s2_array, s0_array)
    """
    # scale eroded foreground mask to 40x from 5x for final multiplications
    fg_eroded = open_ds(zarr_path / "mask" / "foreground_eroded")  # in s3
    upsampling_factor = fg_eroded.voxel_size / vessel_mask.voxel_size
    fg_eroded_s0 = prepare_ds(
        zarr_path / "mask" / "foreground_eroded_s0",
        shape=Coordinate(fg_eroded.shape) * upsampling_factor,
        offset=fg_eroded.offset,
        voxel_size=vessel_mask.voxel_size,
        axis_names=fg_eroded.axis_names,
        units=fg_eroded.units,
        mode="w",
        dtype=fg_eroded.dtype,
    )

    # upsample tissue mask from 5x to 40x
    tissuemask_upsample(fg_eroded, fg_eroded_s0, s0_array, upsampling_factor)
    """
    return
