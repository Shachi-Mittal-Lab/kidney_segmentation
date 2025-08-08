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
from fibrosis_score.processing_utils import fill_holes, erode, varied_vessel_dilation
from fibrosis_score.daisy_blocks import (
    model_prediction,
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

def cluster(
        input_path: Path,
        png_path: Path,
        threshold: int,
        offset: Coordinate,
        axis_names: list,
        gray_cluster_centers: Path,
):

    # check there is a GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device}")

    # Eric edit:no pngs before it was created
    #os.makedirs("/media/mrl/Data/pipeline_connection/pngs", exist_ok=True)

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

    # get cluster centers
    clustering, n_clusters = prepare_clustering(gray_cluster_centers)

    # grab 40x level
    s0_array = open_ds(zarr_path / "raw" / "s0")

    print("Preparing Masks for Clustering")
    # prepare clustering mask locations in zarr (40x)
    fibrosis1_mask, fibrosis2_mask, inflammation_mask, structuralcollagen_mask = (
        prepare_clustering_masks(zarr_path, s0_array)
    )

    print("Clustering")
    for offset in tqdm(offsets_final):
        # world units roi selection
        roi = Roi(offset, patch_size_final)  # in nm
        # visualize roi in mask
        voxel_mask_roi = (roi - roi_mask.offset) / roi_mask.voxel_size
        roi_mask._source_data[
            voxel_mask_roi.begin[0] : voxel_mask_roi.end[0],
            voxel_mask_roi.begin[1] : voxel_mask_roi.end[1],
        ] = 1

        # save image at diff pyramid levels for Omni-Seg input
        # name roi
        patchname = Path(f"{zarr_path.stem}_{offset}_pyr0.png")
        patchpath = png_path / patchname
        # select roi
        array = open_ds(zarr_path / "raw" / f"s0")

        # redefine roi on s0 array
        patch_shape_final = Coordinate(4096, 4096)  # omniseg input was 4096/8192
        patch_size_final = patch_shape_final * array.voxel_size  # nm
        roi = Roi(offset, patch_size_final)  # nm
        voxel_roi = (roi - array.offset) / array.voxel_size  # voxels
        patch_raw = array[
            voxel_roi.begin[0] : voxel_roi.end[0], voxel_roi.begin[1] : voxel_roi.end[1]
        ]

        # cluster 40x pyramid level
        # save png of image at current pyramid level
        grayscale_cluster(
            patchpath,
            patch_raw,
            voxel_roi,
            clustering,
            n_clusters,
            fibrosis1_mask,
            fibrosis2_mask,
            inflammation_mask,
            structuralcollagen_mask,
        )

input_path = Path("/home/riware/Desktop/loose_files/clustering_investigation_04Aug2025/BR22-2097-A-1-9-TRICHROME - 2022-11-11 17.54.03.ndpi")
png_path = Path("/home/riware/Desktop/loose_files/clustering_investigation_04Aug2025")

# background threshold
threshold = 50
# slide offset (zero)
offset = (0, 0)
# axis names
axis_names = [
    "x",
    "y",
    "c^",
]
grayscale_cluster_centers = Path("/home/riware/Documents/MittalLab/kidney_project/kidney_segmentation/fibrosis_score/average_centers_5.txt")
cluster(input_path, png_path, threshold, offset, axis_names, grayscale_cluster_centers)