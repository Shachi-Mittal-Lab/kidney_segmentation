import zarr.convenience
from patch_utils_dask import openndpi, foreground_mask
from cluster_utils import sigmoid_norm
import dask
import dask.array 
import zarr
import numpy as np
import cv2
from tqdm import tqdm
from skimage.transform import integral_image
from skimage.morphology import binary_erosion, binary_dilation, disk, remove_small_objects
from skimage.filters import gaussian
from scipy.ndimage import label
from skimage import measure
import daisy

from pathlib import Path
import matplotlib.pyplot as pltS
import torch
import torchvision.transforms as T
from sklearn.cluster import KMeans

from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi
from dask.diagnostics import ProgressBar
from dask.array import coarsen, mean
import os  # Eric edit
import subprocess  # Eric edit

from torchvision import transforms
from model import UNet, ConvBlock, Downsample, CropAndConcat, OutputConv
from PIL import Image

# Eric edit
import sys

sys.path.append("/media/mrl/Data/pipeline_connection/kidney_segmentation")
from directory_paths import (
    ndpi_path,
    zarr_path,
    png_path,
    txt_path,
    threshold,
    offset,
    axis_names,
    cortmed_weights_path,
    cortmed_starting_model_path,
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
# Eric edit:no pngs before it was created
os.makedirs("/media/mrl/Data/pipeline_connection/pngs", exist_ok=True)

###########

# convert ndpi to zarr array levels 40x, 20x, 10x, 5x

# open highest pyramid level
#dask_array0, x_res, y_res, units = openndpi(ndpi_path, 0)
#s0_shape = dask_array0.shape
#units = ("nm", "nm")
## grab resolution in cm, convert to nm, calculate for each pyramid level
#voxel_size0 = Coordinate(int(1 / x_res * 1e7), int(1 / y_res * 1e7))
## format data as funlib dataset
#raw = prepare_ds(
#    zarr_path / "raw" / "s0",
#    dask_array0.shape,
#    offset,
#    voxel_size0,
#    axis_names,
#    units,
#    mode="w",
#    dtype=np.uint8,
#)
## storage info
#store_rgb = zarr.open(zarr_path / "raw" / "s0")
#dask_array = dask_array0.rechunk(raw.data.chunksize)
#
#with ProgressBar():
#    dask.array.store(dask_array, store_rgb)
#
#for i in range(1, 4):
#    # open the ndpi with openslide for info and tifffile as zarr
#    dask_array, x_res, y_res, _ = openndpi(ndpi_path, i)
#    # grab resolution in cm, convert to nm, calculate for each pyramid level
#    voxel_size = Coordinate(int(1 / x_res * 1e7) * 2**i, int(1 / y_res * 1e7) * 2**i)
#    expected_shape = tuple((s0_shape[0] // 2**i, s0_shape[1] // 2**i,3))
#    print(f"expected shape: {expected_shape}")
#    print(f"actual shape: {dask_array.shape}")
#
#    # check shape is expected shape 
#    if dask_array.shape == expected_shape:
#        print("correct shape")
#        # format data as funlib dataset
#        raw = prepare_ds(
#            zarr_path / "raw" / f"s{i}",
#            dask_array.shape,
#            offset,
#            voxel_size,
#            axis_names,
#            units,
#            mode="w",
#            dtype=np.uint8,
#        )
#        # storage info
#        store_rgb = zarr.open(zarr_path / "raw" / f"s{i}")
#        dask_array = dask_array.rechunk(raw.data.chunksize)
#
#        with ProgressBar():
#            dask.array.store(dask_array, store_rgb)
#
#    else:
#        voxel_size = tuple((voxel_size0[0] * 2**i, voxel_size0[0] * 2**i))
#        # format data as funlib dataset
#        raw = prepare_ds(
#            zarr_path / "raw" / f"s{i}",
#            expected_shape,
#            offset,
#            voxel_size,
#            axis_names,
#            units,
#            mode="w",
#            dtype=np.uint8,
#        )
#        # storage info
#        store_rgb = zarr.open(zarr_path / "raw" / f"s{i}")
#        prev_layer = open_ds(zarr_path / "raw" / f"s{i-1}")
#        print(f"chunk shape: {prev_layer.chunk_shape}")
#
#        # mean downsampling
#        dask_array = coarsen(mean, prev_layer.data, {0: 2, 1: 2})
#
#        # save to zarr
#        with ProgressBar():
#            dask.array.store(dask_array, store_rgb)

# grab new zarr with reasonable chunks as dask array
s3_array = open_ds(zarr_path / "raw" / "s3")

# create and save foreground mask
mask = foreground_mask(s3_array.data, threshold)
fmask = prepare_ds(
    zarr_path / "mask" / "foreground",
    s3_array.shape[0:2],
    s3_array.offset,
    s3_array.voxel_size,
    s3_array.axis_names[0:2],
    s3_array.units,
    mode="w",
    dtype=np.uint8,
)
store_fgbg = zarr.open(zarr_path / "mask" / "foreground")
dask.array.store(mask, store_fgbg)

# fill holes in foreground mask and save as filled mask
fill_hole_kernel = disk(
    15, decomposition="sequence"
)  # sequence for computational efficiency

# fill holes
shrink_kernel = disk(10, decomposition="sequence")
mask_filled = binary_erosion(binary_dilation(mask, fill_hole_kernel), shrink_kernel)
mask_filled_array = prepare_ds(
    zarr_path / "mask" / "foreground_filled",
    s3_array.shape[0:2],
    s3_array.offset,
    s3_array.voxel_size,
    s3_array.axis_names[0:2],
    s3_array.units,
    mode="w",
    dtype=np.uint8,
)
mask_filled_array._source_data[:] = mask_filled

# erosion to shave away edge tissue
shrink_kernel = disk(
    40, decomposition="sequence"
)  # sequence for computational efficiency
mask_eroded = binary_erosion(binary_dilation(mask, fill_hole_kernel), shrink_kernel)
mask_eroded_array = prepare_ds(
    zarr_path / "mask" / "foreground_eroded",
    s3_array.shape[0:2],
    s3_array.offset,
    s3_array.voxel_size,
    s3_array.axis_names[0:2],
    s3_array.units,
    mode="w",
    dtype=np.uint8,
)
mask_eroded_array._source_data[:] = mask_eroded

# create integral mask of filled foreground mask and save
integral_mask = integral_image(mask_filled)
imask = prepare_ds(
    zarr_path / "mask" / "integral_foreground",
    s3_array.shape[0:2],
    s3_array.offset,
    s3_array.voxel_size,
    s3_array.axis_names[0:2],
    s3_array.units,
    mode="w",
    dtype=np.uint64,
)
imask._source_data[:] = integral_mask

# create patch mask: this marks every pixel that can be the starting pixel
# of a patch of the defined patch size that will encompass mostly tissue
patch_shape = Coordinate(224, 224)  # Will convention: shape in voxels started 150 x 150
patch_size = patch_shape * imask.voxel_size  # size in nm
patch_roi = Roi(imask.roi.offset, imask.roi.shape - patch_size)
patch_counts = (
    imask[patch_roi]  # indexing with a roi gives you a plain numpy array
    + imask[patch_roi + patch_size]
    - imask[patch_roi + patch_size * Coordinate(0, 1)]
    - imask[patch_roi + patch_size * Coordinate(1, 0)]
)
patch_count_mask = (
    patch_counts >= patch_shape[0] * patch_shape[1] * 0.30
)  # 30% of voxels are tissue

# make grid on s3
patch_spacing = Coordinate(112, 112)  # was 100x100

# very strange... why we give y shape then x shape... but it works
grid_x, grid_y = np.meshgrid(
    range(patch_count_mask.shape[1]), range(patch_count_mask.shape[0])
)
gridmask = grid_x % patch_spacing[0] + grid_y % patch_spacing[1]
gridmask = gridmask == 0

# create mask of pixels for possible starting pixels gien the ROI size and threshold
patch_mask = prepare_ds(
    zarr_path / "mask" / "patch_mask",
    patch_count_mask.shape,
    s3_array.offset,
    s3_array.voxel_size,
    s3_array.axis_names[0:2],
    s3_array.units,
    mode="w",
    dtype=np.uint8,
)
patch_mask._source_data[:] = patch_count_mask * gridmask
# dask.array.store(patch_count_mask, store_patch_mask)

coords = np.nonzero(patch_mask._source_data[:])
offsets = [
    patch_mask.roi.offset + Coordinate(a, b) * patch_mask.voxel_size
    for a, b in zip(*coords)
]

# apply foreground vs background model at 5x (s3)
s3_array = open_ds(zarr_path / "raw" / "s3")

with torch.no_grad():
    # load vgg16 weights
    checkpoint = torch.load(
        cortmed_weights_path, map_location=torch.device("cpu"), weights_only=False
    )
    # load vgg16 architecture
    vgg16: torch.nn.Module = torch.load(
        cortmed_starting_model_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    # load weights into architecture
    vgg16.load_state_dict(checkpoint["model_state_dict"])
    vgg16.to(device)

    # make dataset for mask
    pred_mask = prepare_ds(
        zarr_path / "mask" / "cortex",
        s3_array.shape[0:2],
        s3_array.offset,
        s3_array.voxel_size,
        s3_array.axis_names[0:2],
        s3_array.units,
        mode="w",
        dtype=np.uint8,
    )

    # default pixel value to 3
    pred_mask[:] = 3

    # for each patch, send to vgg16 and write label to mask
    for offset in tqdm(offsets):
        # world units roi selection
        roi = Roi(offset, patch_size)
        voxel_roi = (roi - s3_array.offset) / s3_array.voxel_size
        patch_raw = s3_array[
            voxel_roi.begin[0] : voxel_roi.end[0], voxel_roi.begin[1] : voxel_roi.end[1]
        ]
        # format roi for vgg16
        in_data = T.Resize((224, 224))(
            torch.from_numpy(patch_raw).float().permute(2, 0, 1).unsqueeze(0) / 255
        ).to(device)
        # predict
        pred = vgg16(in_data).cpu().squeeze(0).numpy().argmax()
        #  write to mask, accounting for roi overlap
        context = (patch_size - patch_spacing * s3_array.voxel_size) // 2
        slices = pred_mask._Array__slices(roi.grow(-context, -context))
        pred_mask._source_data[slices] = pred
        pred_mask._source_data[:] = pred_mask._source_data[:] == 1

# remove vgg16 from gpu memmory 
del vgg16

# multiply by tissue max to shave off background
pred_mask._source_data[:] = pred_mask._source_data * mask_filled_array._source_data[:]

# create integral mask of cortex mask and save
integral_cortex_mask = integral_image(pred_mask._source_data[:])
icmask = prepare_ds(
    zarr_path / "mask" / "integral_cortex",
    s3_array.shape[0:2],
    s3_array.offset,
    s3_array.voxel_size,
    s3_array.axis_names[0:2],
    s3_array.units,
    mode="w",
    dtype=np.uint64,
)
icmask._source_data[:] = integral_cortex_mask

# create patch mask: this marks every pixel that can be the starting pixel
# of a patch of the defined patch size that will encompass mostly tissue
patch_shape_final = Coordinate(
    512, 512
)  # Will convention: shape in voxels started 150 x 150 omni-seg input WAS 512/1024 
patch_size_final = patch_shape_final * icmask.voxel_size  # size in nm
patch_roi_final = Roi(icmask.roi.offset, icmask.roi.shape - patch_size_final)
patch_counts_final = (
    icmask[patch_roi_final]  # indexing with a roi gives you a plain numpy array
    + icmask[patch_roi_final + patch_size_final]
    - icmask[patch_roi_final + patch_size_final * Coordinate(0, 1)]
    - icmask[patch_roi_final + patch_size_final * Coordinate(1, 0)]
)
patch_count_mask_final = (
    patch_counts_final >= patch_shape_final[0] * patch_shape_final[1] * 0.10
)  # 10% of voxels are cortex

# make grid on s3
patch_spacing_final = Coordinate(256, 256)  # was 256x256/512 #TRY CHANGING THIS THIS TIME

# very strange... why we give y shape then x shape... but it works
grid_x_final, grid_y_final = np.meshgrid(
    range(patch_count_mask_final.shape[1]), range(patch_count_mask_final.shape[0])
)
gridmaskfinal = (
    grid_x_final % patch_spacing_final[0] + grid_y_final % patch_spacing_final[1]
)
gridmaskfinal = gridmaskfinal == 0

# create mask of pixels for possible starting pixels gien the ROI size and threshold
patch_mask_final = prepare_ds(
    zarr_path / "mask" / "patch_mask_final",
    patch_count_mask_final.shape,
    s3_array.offset,
    s3_array.voxel_size,
    s3_array.axis_names[0:2],
    s3_array.units,
    mode="w",
    dtype=np.uint8,
)
patch_mask_final._source_data[:] = gridmaskfinal * patch_count_mask_final

# dask.array.store(patch_count_mask, store_patch_mask)
coords = np.nonzero(patch_mask_final._source_data[:])
offsets_final = [
    patch_mask_final.roi.offset + Coordinate(a, b) * patch_mask_final.voxel_size
    for a, b in zip(*coords)
]

# create visualization of regions pulled
roi_mask = prepare_ds(
    zarr_path / "mask" / "rois",
    patch_count_mask.shape,
    s3_array.offset,
    s3_array.voxel_size,
    s3_array.axis_names[0:2],
    s3_array.units,
    mode="w",
    dtype=np.uint8,
)
roi_mask._source_data[:] = 0

# get cluster centers
centers = np.loadtxt(gray_cluster_centers)
centers = np.expand_dims(centers, axis=1)
n_clusters = len(centers)
# Assign cluster centers
clustering = KMeans(n_clusters=n_clusters)
clustering.fit(centers)

# create visualization of fibrosis clusters
s0_array = open_ds(zarr_path / "raw" / "s0")

fibrosis1_mask = prepare_ds(
    zarr_path / "mask" / "fibrosis1",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
fibrosis1_mask._source_data[:] = 0

fibrosis2_mask = prepare_ds(
    zarr_path / "mask" / "fibrosis2",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
fibrosis2_mask._source_data[:] = 0

# create visualization of inflammation cluster
inflammation_mask = prepare_ds(
    zarr_path / "mask" / "inflammation",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
inflammation_mask._source_data[:] = 0

# create visualization of tbm - mixing tubule dilation & structural collagen cluster
tbm_mask = prepare_ds(
    zarr_path / "mask" / "tbm",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
tbm_mask._source_data[:] = 0

# create visualization of tbm - mixing tubule dilation & structural collagen cluster
bc_mask = prepare_ds(
    zarr_path / "mask" / "bc",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
bc_mask._source_data[:] = 0

# create visualization of caps at 40x
fincap_mask = prepare_ds(
    zarr_path / "mask" / "fincap",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
fincap_mask._source_data[:] = 0

# create visualization of structural collagen cluster
structuralcollagen_mask = prepare_ds(
    zarr_path / "mask" / "structuralcollagen",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
structuralcollagen_mask._source_data[:] = 0

# create mask in zarr for proximal tubule class at 40x
pt_mask = prepare_ds(
    zarr_path / "mask" / "pt",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
pt_mask._source_data[:] = 0

# create mask in zarr for distal tubule class at 40x
dt_mask = prepare_ds(
    zarr_path / "mask" / "dt",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
dt_mask._source_data[:] = 0

# create mask in zarr for tuft class at 40x
tuft_mask = prepare_ds(
    zarr_path / "mask" / "tuft",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
tuft_mask._source_data[:] = 0

s2_array = open_ds(zarr_path / "raw" / "s2")

# create mask in zarr for capsule class at 10x
cap_mask10x = prepare_ds(
    zarr_path / "mask" / "cap10x",
    s2_array.shape[0:2],
    s2_array.offset,
    s2_array.voxel_size,
    s2_array.axis_names[0:2],
    s2_array.units,
    mode="w",
    dtype=np.uint8,
)
cap_mask10x._source_data[:] = 0

# create mask in zarr for peritubular capillaries class at 40x
ptc_mask = prepare_ds(
    zarr_path / "mask" / "ptc",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
ptc_mask._source_data[:] = 0

# create mask in zarr for vessel class at 40x
vessel_mask = prepare_ds(
    zarr_path / "mask" / "vessel",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
vessel_mask._source_data[:] = 0

# create mask for final fibrosis overlay at 40x
finfib_mask = prepare_ds(
    zarr_path / "mask" / "finfib",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
finfib_mask._source_data[:] = 0

# create mask for final structural collagen overlay at 40x
fincollagen_mask = prepare_ds(
    zarr_path / "mask" / "fincollagen",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
fincollagen_mask._source_data[:] = 0

# create mask for final inflammation overlay at 40x
fininflamm_mask = prepare_ds(
    zarr_path / "mask" / "fininflamm",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
fininflamm_mask._source_data[:] = 0

# create text file for fibrosis score data
with open(txt_path, "w") as f:
    f.writelines("ROI fibrosis scores \n")

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
    patch_shape_final = Coordinate(4096,4096) #omniseg input was 4096/8192
    patch_size_final = patch_shape_final * array.voxel_size # nm
    roi = Roi(offset, patch_size_final) # nm
    voxel_roi = (roi - array.offset) / array.voxel_size # voxels
    patch_raw = array[
        voxel_roi.begin[0] : voxel_roi.end[0], voxel_roi.begin[1] : voxel_roi.end[1]
    ]

    # cluster 40x pyramid level
    # save png of image at current pyramid level
    patch_raw = np.transpose(patch_raw, axes=(1, 0, 2))
    patch_raw = patch_raw[:, :, ::-1]  # flip to rgb
    cv2.imwrite(str(patchpath), patch_raw)
    # convert to grayscale & format for k means
    roi_gbr = cv2.imread(str(patchpath))
    roi_rgb = cv2.cvtColor(roi_gbr, cv2.COLOR_BGR2RGB)
    roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2GRAY)
    roi_gray = sigmoid_norm(roi_gray)
    roi_gray_flat = roi_gray.flatten()
    # k means predict
    roi_gray_flat = np.expand_dims(roi_gray_flat, axis=1)
    prediction = clustering.predict(roi_gray_flat)
    # order centers from clustering for ease of color reassignment
    index = np.arange(0, n_clusters, 1).reshape(-1, 1)
    centers = np.hstack((index, clustering.cluster_centers_))
    ascending_centers = np.array(sorted(centers, key=lambda x: x[1]))
    ordered_centers = ascending_centers[::-1]
    new_order = ordered_centers[:, 0].astype(int).reshape(-1, 1)
    mapping = np.array(
        [x[1] for x in sorted(zip(new_order, index))], dtype=int
    ).reshape(-1, 1)
    # updated labels
    class_labels = mapping[prediction].reshape(roi_gray.shape)
    # save centers
    final_centers = np.uint8(ordered_centers[:, 1:])
    # save inflammation 4
    inflammation = class_labels == 4
    # visualize roi in mask, transpose to align
    inflammation_mask._source_data[
        voxel_roi.begin[0] : voxel_roi.end[0],
        voxel_roi.begin[1] : voxel_roi.end[1],
    ] = inflammation.T
    # save fibrosis 1
    fibrosis1 = class_labels == 1
    # connected components fib1
    structure = np.ones((3, 3), dtype=np.int8)  # allows any connection
    labels, ncomponents = label(fibrosis1, structure)
    unique, counts = np.unique(labels, return_counts=True)
    small_obj = counts <= 50
    to_remove = unique[small_obj]
    small_obj_mask = np.isin(labels, to_remove)
    fibrosis1[small_obj_mask] = 0
    
    # visualize roi in mask, transpose to align
    fibrosis1_mask._source_data[
        voxel_roi.begin[0] : voxel_roi.end[0],
        voxel_roi.begin[1] : voxel_roi.end[1],
    ] = fibrosis1.T
    # save fibrosis 2
    fibrosis2 = class_labels == 2
    # connected components fib2
    structure = np.ones((3, 3), dtype=np.int8)  # allows any connectio
    labels, ncomponents = label(fibrosis2, structure)
    unique, counts = np.unique(labels, return_counts=True)
    small_obj = counts <= 50
    to_remove = unique[small_obj]
    small_obj_mask = np.isin(labels, to_remove)
    fibrosis2[small_obj_mask] = 0
    fibrosis2_mask._source_data[
        voxel_roi.begin[0] : voxel_roi.end[0],
        voxel_roi.begin[1] : voxel_roi.end[1],
    ] = fibrosis2.T

    # save structural collagen 3
    structural_collagen = class_labels == 3
    structuralcollagen_mask._source_data[
        voxel_roi.begin[0] : voxel_roi.end[0],
        voxel_roi.begin[1] : voxel_roi.end[1],
    ] = structural_collagen.T

    # send 40x image to omni-seg to predict
    subprocess.run(["python3", omni_seg_path], cwd=subprocesswd, check=True)

    output_path = Path(output_directory) / "final_merge" / patchname.stem

    # grab omni-seg output and write to zarr
    dt_roimask = np.load(output_path / "0_1_dt" / "slice_pred_40X.npy")
    dt_roimask = dt_roimask[:,:,0]
    dt_roimask = dt_roimask > 0
    dt_mask._source_data[
        voxel_roi.begin[0] : voxel_roi.end[0],
        voxel_roi.begin[1] : voxel_roi.end[1],
    ] = dt_roimask.T

    pt_roimask = np.load(output_path / "1_1_pt" / "slice_pred_40X.npy")
    pt_roimask = pt_roimask[:,:,0]
    pt_roimask = pt_roimask > 0
    pt_mask._source_data[
        voxel_roi.begin[0] : voxel_roi.end[0],
        voxel_roi.begin[1] : voxel_roi.end[1],
    ] = pt_roimask.T

    vessel_roimask = np.load(output_path / "4_1_vessel" / "slice_pred_40X.npy")
    vessel_roimask = vessel_roimask[:,:,0]
    vessel_roimask = vessel_roimask > 0
    vessel_mask._source_data[
        voxel_roi.begin[0] : voxel_roi.end[0],
        voxel_roi.begin[1] : voxel_roi.end[1],
    ] = vessel_roimask.T

    ptc_roimask = np.load(output_path / "5_3_ptc" / "slice_pred_40X.npy")
    ptc_roimask = ptc_roimask[:,:,0]
    ptc_roimask = ptc_roimask > 0
    ptc_mask._source_data[
        voxel_roi.begin[0] : voxel_roi.end[0],
        voxel_roi.begin[1] : voxel_roi.end[1],
    ] = ptc_roimask.T

# Use U-net to predict gloms
# load model
model = torch.load("model_dataset1_flips.pt", weights_only=False)
# preprocessing
inp_transforms = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
    ]
)

# define roi size 
patch_shape_final = Coordinate(512,512)
patch_size_final = patch_shape_final * s2_array.voxel_size  # size in nm

# blockwise predictions
def cap_id_block(block: daisy.Block):
    # in data slice
    inslices = s2_array._Array__slices(block.read_roi)
    # it was [:, 0:512, 0:512]
    # we want [0:512, 0:512, :]
    inslices = (inslices[1], inslices[2], inslices[0])
    img = Image.fromarray(s2_array[inslices])
    input = inp_transforms(img).unsqueeze(1).to(device)
    with torch.no_grad():
        preds = model(input)
        preds = preds.squeeze().squeeze().cpu().detach().numpy()
        preds = preds > 0.5
    cap_mask10x[block.write_roi] = preds

cap_id_task = daisy.Task(
    "Cap ID",
    total_roi=s2_array.roi,
    read_roi=Roi((0,0), patch_size_final), # (offset, shape)
    write_roi=Roi((0,0), patch_size_final),
    read_write_conflict=False,
    num_workers=2,
    process_function=cap_id_block
)
daisy.run_blockwise(tasks=[cap_id_task], multiprocessing=False)

# remove small objects
cap_mask10x._source_data[:] = remove_small_objects(cap_mask10x._source_data[:].astype(bool), min_size=2000)

# blockwise upsample to 40x
# upsample 5x eroded tissue mask to use to filter predictions
def upsample_block(block: daisy.Block):
    s2_data = cap_mask10x[block.read_roi]
    s0_data = s2_data
    for axis, reps in enumerate(upsampling_factor):
        s0_data = np.repeat(s0_data, reps, axis=axis)
    fincap[block.write_roi] = s0_data

block_roi = Roi((0, 0), (1000, 1000)) * s0_array.voxel_size
upsample_task = daisy.Task(
    "blockwise_upsample",
    total_roi=s2_array.roi,
    read_roi=block_roi,
    write_roi=block_roi,
    read_write_conflict=False,
    num_workers=2,
    process_function=upsample_block,
)
daisy.run_blockwise(tasks=[upsample_task], multiprocessing=False)

# grab arrays that were already calc - TEMP SECTION
vessel_roimask = open_ds(zarr_path / "mask" / "vessel")
dt_roimask = open_ds(zarr_path / "mask" / "dt")
pt_roimask = open_ds(zarr_path / "mask" / "pt")
cap_roimask = open_ds(zarr_path / "mask" / "cap")
inflammation = open_ds(zarr_path / "mask" / "inflammation")
fibrosis1 = open_ds(zarr_path / "mask" / "fibrosis1")
fibrosis2 = open_ds(zarr_path / "mask" / "fibrosis2")
finfib = open_ds(zarr_path / "mask" / "finfib", "r+")
fininflamm = open_ds(zarr_path / "mask" / "fininflamm", "r+")
structural_collagen = open_ds(zarr_path / "mask" / "structuralcollagen")
fincollagen = open_ds(zarr_path / "mask" / "fincollagen", "r+")
fincap = open_ds(zarr_path / "mask" / "fincap", "r+")
bc = open_ds(zarr_path / "mask" / "bc", "r+")

# scale eroded foreground mask to 40x from 5x for final multiplications
fg_eroded = open_ds(zarr_path / "mask" / "foreground_eroded")  # in s3
upsampling_factor = fg_eroded.voxel_size / vessel_roimask.voxel_size
fg_eroded_s0 = prepare_ds(
    zarr_path / "mask" / "foreground_eroded_s0",
    shape=Coordinate(fg_eroded.shape) * upsampling_factor,
    offset=fg_eroded.offset,
    voxel_size=vessel_roimask.voxel_size,
    axis_names=fg_eroded.axis_names,
    units=fg_eroded.units,
    mode="w",
    dtype=fg_eroded.dtype,
)

# upsample 5x eroded tissue mask to use to filter predictions
def upsample_block(block: daisy.Block):
    s3_data = fg_eroded[block.read_roi]
    s0_data = s3_data
    for axis, reps in enumerate(upsampling_factor):
        s0_data = np.repeat(s0_data, reps, axis=axis)
    fg_eroded_s0[block.write_roi] = s0_data

block_roi = Roi((0, 0), (1000, 1000)) * fg_eroded_s0.voxel_size
upsample_task = daisy.Task(
    "blockwise_upsample",
    total_roi=s0_array.roi,
    read_roi=block_roi,
    write_roi=block_roi,
    read_write_conflict=False,
    num_workers=2,
    process_function=upsample_block,
)
daisy.run_blockwise(tasks=[upsample_task], multiprocessing=False)

fg_eroded_s0 = open_ds(zarr_path / "mask" / "foreground_eroded_s0")  # in s0

# need to erode and dilate to generate a TBM class
erode_kernel = disk(
    2, decomposition="sequence"
)
dilate_kernel = disk(
    15, decomposition="sequence"
)  # sequence for computational efficiency

def tbm_block(block: daisy.Block):
    pt_data = dt_roimask[block.read_roi]
    dt_data = pt_roimask[block.read_roi]
    cluster = structural_collagen[block.read_roi]
    tubules = pt_data + dt_data
    eroded_tubules = binary_erosion(tubules, erode_kernel)
    tbm_tubule = binary_dilation(eroded_tubules, dilate_kernel)
    tbm = (tbm_tubule * ~eroded_tubules.astype(bool)) * cluster
    tbm = gaussian(tbm.astype(float), sigma=2.0) > 0.5
    tbm_mask[block.write_roi] = tbm

block_roi = Roi((0, 0), (1000, 1000)) * fg_eroded_s0.voxel_size
tbm_task = daisy.Task(
    "ID TBM",
    total_roi=s0_array.roi,
    read_roi=block_roi,
    write_roi=block_roi,
    read_write_conflict=False,
    num_workers=2,
    process_function=tbm_block,
)
daisy.run_blockwise(tasks=[tbm_task], multiprocessing=False)

tbm = open_ds(zarr_path / "mask" / "tbm")

structure = np.ones((3, 3), dtype=np.int8)  # allows any connection

# clean cap class
structure = np.ones((3, 3), dtype=np.int8)  # allows any connection
erode_kernel = disk(
    40, decomposition="sequence"
)
dilate_kernel = disk(
    17, decomposition="sequence"
)  # sequence for computational efficiency

def filter_circular_objects(mask, circularity_threshold=0.2):
    labeled_mask = measure.label(mask)
    output_mask = np.zeros(mask.shape, dtype = np.uint8)

    for region in measure.regionprops(labeled_mask):
        
        perimeter = region.perimeter if region.perimeter > 0 else 1
        circularity = (4 * np.pi * region.area) / (perimeter ** 2)

        if circularity >= circularity_threshold:
            output_mask[labeled_mask == region.label] = True

    return output_mask


################################################
# create final fibrosis overlay: fibrosis1 + fibrosis2 from clustering - tuft, capsule, vessel, tubules
finfib.data = (
    (fibrosis1.data
    + fibrosis2.data
    + structural_collagen.data)
    * (1 - vessel_roimask.data)
    * (1 - dt_roimask.data)
    * (1 - pt_roimask.data)
    * (1 - fincap.data)
    * (1 - tbm.data)
    * fg_eroded_s0.data.astype(bool)
)

dask.array.store(finfib.data, finfib._source_data)

# create final collagen overlay
fincollagen.data = (
    structural_collagen.data
    * ~dt_roimask.data.astype(bool)
    * ~pt_roimask.data.astype(bool)
    * fg_eroded_s0.data.astype(bool)
)

dask.array.store(fincollagen.data, fincollagen._source_data)

# create final inflammation overlay
fininflamm.data = (
    inflammation.data
    * ~vessel_roimask.data.astype(bool)
    * ~dt_roimask.data.astype(bool)
    * ~pt_roimask.data.astype(bool)
    * ~fincap.data.astype(bool)
    * fg_eroded_s0.data.astype(bool)
)
dask.array.store(fininflamm.data, fininflamm._source_data)

# calculate ROI fibrosis score & save to txt file
with open(Path(zarr_path.parent / "fibpx.txt"), "w") as f:
    f.writelines("# Fibrosis Pixels per Block \n")
with open(Path(zarr_path.parent / "tissuepx.txt"), "w") as f:
    f.writelines("# Tissue Pixels per Block \n")

# blockwise mask multiplications
def fibscore_block(block: daisy.Block):
    # in data
    fib = finfib[block.read_roi]
    fg_er = fg_eroded_s0[block.read_roi]
    vessel = vessel_roimask[block.read_roi]
    cap = fincap[block.read_roi]
    # calc positive pixels in mask
    fibpx = np.count_nonzero(fib)
    fgpx = fg_er * ~vessel.astype(bool) * ~cap.astype(bool)
    fgpx = np.count_nonzero(fgpx)

    #write to text file
    # create text file
    with open(Path(zarr_path.parent / "fibpx.txt"), "a") as f:
        f.writelines(f"{fibpx} \n")
    with open(Path(zarr_path.parent / "tissuepx.txt"), "a") as f:
        f.writelines(f"{fgpx} \n")

fibscore_task = daisy.Task(
    "fibscore calc",
    total_roi=s0_array.roi,
    read_roi=Roi((0,0),(1000000,1000000)),
    write_roi=Roi((0,0),(1000000,1000000)),
    read_write_conflict=False,
    num_workers=2,
    process_function=fibscore_block
)
daisy.run_blockwise(tasks=[fibscore_task], multiprocessing=False)

fibpx = np.loadtxt(Path(zarr_path.parent / "fibpx.txt"), comments="#", dtype=int)
tissuepx = np.loadtxt(Path(zarr_path.parent / "tissuepx.txt", comments="#", dtype=int))

total_fibpx = np.sum(fibpx)
total_tissuepx = np.sum(tissuepx)
total_fibscore = (total_fibpx / total_tissuepx) * 100

with open(txt_path, "a") as f:
    f.writelines(f"Final score: {total_fibscore}")

# total_tissue = foregroundmask * ~capsule * ~tuft * ~vessel
# score = finfib / total_tissue
# with open(os.path.join(MODEL_DATA_PATH, "test_cases.txt"), "w") as f:
#  f.writelines("\n".join(score))s
