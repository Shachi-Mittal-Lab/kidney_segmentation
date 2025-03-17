# Repo Tools
from zarr_utils import ndpi_to_zarr, omniseg_to_zarr
from mask_utils import (
    prepare_mask,
    prepare_foreground_masks,
    prepare_clustering_masks,
    prepare_omniseg_masks,
    prepare_postprocessing_masks,
    prepare_patch_mask,
    generate_patchcount_mask,
    generate_grid_mask,
)
from patch_utils_dask import foreground_mask
from cluster_utils import prepare_clustering, grayscale_cluster
from pred_utils import pred_cortex
from processing_utils import fill_holes, erode, varied_vessel_dilation
from daisy_blocks import (
    cap_prediction,
    cap_upsample,
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
from model import UNet, ConvBlock, Downsample, CropAndConcat, OutputConv

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

sys.path.append("/media/mrl/Data/pipeline_connection/kidney_segmentation")

# import inputs from inputs file
from directory_paths import (
    ndpi_path,
    zarr_path,
    png_path,
    txt_path,
    threshold,
    offset,
    axis_names,
    gray_cluster_centers,
    output_directory,
    omni_seg_path,
    subprocesswd,
)

# check there is a GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using {device}")

# Eric edit:no pngs before it was created
os.makedirs("/media/mrl/Data/pipeline_connection/pngs", exist_ok=True)

###########

# convert ndpi to zarr array levels 40x, 20x, 10x, 5x, and rechunk
# ndpi_to_zarr(ndpi_path, zarr_path, offset, axis_names)

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

print("Preparing Masks for Clustering and Omni-Seg")
# prepare clustering mask locations in zarr (40x)
fibrosis1_mask, fibrosis2_mask, inflammation_mask, structuralcollagen_mask = (
    prepare_clustering_masks(zarr_path, s0_array)
)
# prepare Omni-Seg mask locations in zarr (40x)
pt_mask, dt_mask, ptc_mask, vessel_mask = prepare_omniseg_masks(zarr_path, s0_array)

# prepare postprocessing masks in zarr (40x)
tbm_mask, bc_mask, fincap_mask, finfib_mask, fincollagen_mask, fininflamm_mask = (
    prepare_postprocessing_masks(zarr_path, s0_array)
)
# prep mask for cap at 10x
cap_mask10x = prepare_mask(zarr_path, s2_array, "cap10x")

# create text file for fibrosis score data
with open(txt_path, "w") as f:
    f.writelines("ROI fibrosis scores \n")

print("Clustering and Omn-Seg Predictions")
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

    # send 40x image to omni-seg to predict
    subprocess.run(["python3", omni_seg_path], cwd=subprocesswd, check=True)

    output_path = Path(output_directory) / "final_merge" / patchname.stem

    # grab omni-seg outputs and write to zarr
    omniseg_to_zarr(output_path / "0_1_dt" / "slice_pred_40X.npy", dt_mask, voxel_roi)
    omniseg_to_zarr(output_path / "1_1_pt" / "slice_pred_40X.npy", pt_mask, voxel_roi)
    omniseg_to_zarr(
        output_path / "4_1_vessel" / "slice_pred_40X.npy", vessel_mask, voxel_roi
    )
    omniseg_to_zarr(output_path / "5_3_ptc" / "slice_pred_40X.npy", ptc_mask, voxel_roi)

print("Predicting Gloms with U-Net")
# Use U-net to predict gloms
# load model
model = torch.load("model_dataset1_flips.pt", weights_only=False)
# preprocessing
inp_transforms = T.Compose(
    [
        T.Grayscale(),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
    ]
)

# define roi size
patch_shape_final = Coordinate(512, 512)
patch_size_final = patch_shape_final * s2_array.voxel_size  # size in nm

print("Predicting Glom Class")
cap_prediction(cap_mask10x, s2_array, patch_size_final, model, device)

# remove small objects from cap mask
cap_mask10x._source_data[:] = remove_small_objects(
    cap_mask10x._source_data[:].astype(bool), min_size=2000
)

upsampling_factor = cap_mask10x.voxel_size / fincap_mask.voxel_size

# blockwise upsample from 10x to 40x
cap_upsample(cap_mask10x, fincap_mask, upsampling_factor, s2_array, s0_array)

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

print("Identifying TBM")
# need to erode and dilate to generate a TBM class
id_tbm(dt_mask, pt_mask, structuralcollagen_mask, tbm_mask, s0_array, fg_eroded_s0)


print("Identifying Bowman's Capsules")
# need to erode and dilate to generate Bowman's Capsule class
id_bc(
    fincap_mask,
    structuralcollagen_mask,
    fibrosis1_mask,
    fibrosis2_mask,
    bc_mask,
    fg_eroded_s0,
    s0_array,
)


print("Identifying Structural Collagen around Vessels")
# create masks for vessel
vessel10x = prepare_mask(zarr_path, s2_array, "vessel10x")
vessel10xdilated = prepare_mask(zarr_path, s2_array, "vessel10xdilated")
vessel40xdilated = prepare_mask(zarr_path, s0_array, "vessel40xdilated")

# downsample 40x vessel predictions to 10x for dilation
downsample_vessel(vessel_mask, vessel10x)

# size of dilation of smallest object
varied_vessel_dilation(3, vessel10x, vessel10xdilated)

################################################
# create final fibrosis overlay: fibrosis1 + fibrosis2 from clustering - tuft, capsule, vessel, tubules
finfib_mask.data = (
    (fibrosis1_mask.data + fibrosis2_mask.data + structuralcollagen_mask.data)
    * (1 - vessel_mask.data)
    * (1 - vessel40xdilated.data)
    * (1 - dt_mask.data)
    * (1 - pt_mask.data)
    * (1 - fincap_mask.data)
    * (1 - tbm_mask.data)
    * fg_eroded_s0.data
)

dask.array.store(finfib_mask.data, finfib_mask._source_data)

# create final collagen overlay
fincollagen_mask.data = (
    (structuralcollagen_mask.data + vessel40xdilated.data + tbm_mask.data + bc_mask.data)
    * (1 - dt_mask.data)
    * (1 - pt_mask.data)
    * (1 - fincap_mask.data)
    * (1 - vessel_mask.data)
    * fg_eroded_s0.data
)

dask.array.store(fincollagen_mask.data, fincollagen_mask._source_data)

# create final inflammation overlay
fininflamm_mask.data = (
    inflammation_mask.data
    * ~vessel_mask.data.astype(bool)
    * ~dt_mask.data.astype(bool)
    * ~pt_mask.data.astype(bool)
    * ~fincap_mask.data.astype(bool)
    * fg_eroded_s0.data.astype(bool)
)
dask.array.store(fininflamm_mask.data, fininflamm_mask._source_data)

# calculate ROI fibrosis score & save to txt file
with open(Path(zarr_path.parent / "fibpx.txt"), "w") as f:
    f.writelines("# Fibrosis Pixels per Block \n")
with open(Path(zarr_path.parent / "tissuepx.txt"), "w") as f:
    f.writelines("# Tissue Pixels per Block \n")


# blockwise mask multiplications
calculate_fibscore(finfib_mask, fg_eroded_s0, vessel_mask, fincap_mask, zarr_path, s0_array)

fibpx = np.loadtxt(Path(zarr_path.parent / "fibpx.txt"), comments="#", dtype=int)
tissuepx = np.loadtxt(Path(zarr_path.parent / "tissuepx.txt", comments="#", dtype=int))

total_fibpx = np.sum(fibpx)
total_tissuepx = np.sum(tissuepx)
total_fibscore = (total_fibpx / total_tissuepx) * 100

with open(txt_path, "a") as f:
    f.writelines(f"Final score: {total_fibscore}")

# total_tissue = foregroundmask * ~capsule * ~tuft * ~vessel
# score = finfib_mask / total_tissue
# with open(os.path.join(MODEL_DATA_PATH, "test_cases.txt"), "w") as f:
#  f.writelines("\n".join(score))s
