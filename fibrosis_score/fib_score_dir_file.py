# Repo Tools
from zarr_utils import ndpi_to_zarr, omniseg_to_zarr
from mask_utils import (
    prepare_mask,
    prepare_foreground_masks,
    prepare_clustering_masks,
    prepare_omniseg_masks,
    prepare_postprocessing_masks,
    prepare_patch_mask,
)
from patch_utils_dask import foreground_mask
from cluster_utils import sigmoid_norm

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
from sklearn.cluster import KMeans

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

print(f"Using {device}")

# Eric edit:no pngs before it was created
os.makedirs("/media/mrl/Data/pipeline_connection/pngs", exist_ok=True)

###########

# convert ndpi to zarr array levels 40x, 20x, 10x, 5x
#ndpi_to_zarr(ndpi_path, zarr_path, offset, axis_names)

# grab new zarr with reasonable chunks as dask array
s3_array = open_ds(zarr_path / "raw" / "s3")
s2_array = open_ds(zarr_path / "raw" / "s2")

# prep foreground mask locations in zarr
fmask, filled_fmask, eroded_fmask = prepare_foreground_masks(zarr_path, s3_array)

# generate and save foreground mask
print("Generating Foreground Masks")
mask = foreground_mask(s3_array.data, threshold)
store_fgbg = zarr.open(zarr_path / "mask" / "foreground")
dask.array.store(mask, store_fgbg)

# fill holes in foreground mask and save as filled mask, use sequence for computational efficiency
fill_hole_kernel = disk(
    15, decomposition="sequence"
)
shrink_kernel = disk(10, decomposition="sequence")
mask_filled = binary_erosion(binary_dilation(mask, fill_hole_kernel), shrink_kernel)
filled_fmask._source_data[:] = mask_filled

# erosion to shave away edge tissue, use sequence for computational efficiency
shrink_kernel = disk(
    40, decomposition="sequence"
)
mask_eroded = binary_erosion(binary_dilation(mask, fill_hole_kernel), shrink_kernel)
eroded_fmask._source_data[:] = mask_eroded

# create integral mask of filled foreground mask and save
imask = prepare_mask(zarr_path, s3_array, "integral_foreground")
integral_mask = integral_image(mask_filled)
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
patch_mask = prepare_patch_mask(zarr_path, s3_array, patch_count_mask, "patch_mask")
patch_mask._source_data[:] = patch_count_mask * gridmask

coords = np.nonzero(patch_mask._source_data[:])
offsets = [
    patch_mask.roi.offset + Coordinate(a, b) * patch_mask.voxel_size
    for a, b in zip(*coords)
]

# apply foreground vs background model at 5x (s3)
s3_array = open_ds(zarr_path / "raw" / "s3")

with torch.no_grad():
    print("Identifying Cortex")
    # load vgg16 weights
    checkpoint = torch.load(
        cortmed_weights_path, map_location=torch.device("cpu"), weights_only=False
    )
    # load vgg16 architecture
    vgg16: torch.nn.Module = torch.load(
        cortmed_starting_model_path,
        map_location=device,
        weights_only=False,
    )
    # load weights into architecture
    vgg16.load_state_dict(checkpoint["model_state_dict"])
    vgg16.to(device)

    # make dataset for mask
    pred_mask = prepare_mask(zarr_path, s3_array, "cortex")

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

print("Preparing Masks for Clustering and Omni-Seg")

# multiply by tissue max to shave off background
pred_mask._source_data[:] = pred_mask._source_data * filled_fmask._source_data[:]

# create integral mask of cortex mask and save
integral_cortex_mask = integral_image(pred_mask._source_data[:])
icmask = prepare_mask(zarr_path, s3_array, "integral_cortex")
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
patch_spacing_final = Coordinate(
    256, 256
)  # was 256x256/512 #TRY CHANGING THIS THIS TIME

# very strange... why we give y shape then x shape... but it works
grid_x_final, grid_y_final = np.meshgrid(
    range(patch_count_mask_final.shape[1]), range(patch_count_mask_final.shape[0])
)
gridmaskfinal = (
    grid_x_final % patch_spacing_final[0] + grid_y_final % patch_spacing_final[1]
)
gridmaskfinal = gridmaskfinal == 0

# create mask of pixels for possible starting pixels gien the ROI size and threshold
patch_mask_final = prepare_patch_mask(
    zarr_path, s3_array, patch_count_mask_final, "patch_mask_final"
)
patch_mask_final._source_data[:] = gridmaskfinal * patch_count_mask_final

# dask.array.store(patch_count_mask, store_patch_mask)
coords = np.nonzero(patch_mask_final._source_data[:])
offsets_final = [
    patch_mask_final.roi.offset + Coordinate(a, b) * patch_mask_final.voxel_size
    for a, b in zip(*coords)
]

# create visualization of regions pulled
roi_mask = prepare_patch_mask(zarr_path, s3_array, patch_count_mask, "rois")

# get cluster centers
centers = np.loadtxt(gray_cluster_centers)
centers = np.expand_dims(centers, axis=1)
n_clusters = len(centers)
# Assign cluster centers
clustering = KMeans(n_clusters=n_clusters)
clustering.fit(centers)

# create visualization of fibrosis clusters
s0_array = open_ds(zarr_path / "raw" / "s0")

# prepare clustering mask locations in zarr (40x)
fibrosis1_mask, fibrosis2_mask, inflammation_mask, structuralcollagen_mask = (
    prepare_clustering_masks(zarr_path, s0_array)
)

# prepare Omni-Seg mask locations in zarr (40x)
pt_mask, dt_mask, ptc_mask, vessel_mask = prepare_omniseg_masks(zarr_path, s0_array)

# prepare masks
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
    fibrosis1 = remove_small_objects(
    fibrosis1.astype(bool), min_size=50)
       # visualize roi in mask, transpose to align
    fibrosis1_mask._source_data[
        voxel_roi.begin[0] : voxel_roi.end[0],
        voxel_roi.begin[1] : voxel_roi.end[1],
    ] = fibrosis1.T
    # save fibrosis 2
    fibrosis2 = class_labels == 2
    # connected components fib2
    fibrosis2 = remove_small_objects(
    fibrosis2.astype(bool), min_size=50)
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

    """
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
    """

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

print("Cleaning Glom Class")
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
    read_roi=Roi((0, 0), patch_size_final),  # (offset, shape)
    write_roi=Roi((0, 0), patch_size_final),
    read_write_conflict=False,
    num_workers=2,
    process_function=cap_id_block,
)
daisy.run_blockwise(tasks=[cap_id_task], multiprocessing=False)

# remove small objects
cap_mask10x._source_data[:] = remove_small_objects(
    cap_mask10x._source_data[:].astype(bool), min_size=2000
)

upsampling_factor = cap_mask10x.voxel_size / fincap_mask.voxel_size
# blockwise upsample to 40x
# upsample 5x eroded tissue mask to use to filter predictions
def upsample_block(block: daisy.Block):
    s2_data = cap_mask10x[block.read_roi]
    s0_data = s2_data
    for axis, reps in enumerate(upsampling_factor):
        s0_data = np.repeat(s0_data, reps, axis=axis)
    fincap_mask[block.write_roi] = s0_data

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

print("Identifying TBM")
# need to erode and dilate to generate a TBM class
erode_kernel = disk(2, decomposition="sequence")
dilate_kernel = disk(
    15, decomposition="sequence"
)  # sequence for computational efficiency

def tbm_block(block: daisy.Block):
    pt_data = dt_mask[block.read_roi]
    dt_data = pt_mask[block.read_roi]
    cluster = structuralcollagen_mask[block.read_roi]
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

print("Identifying Bowman's Capsules")
erode_kernel = disk(14, decomposition="sequence")
dilate_kernel = disk(
    9, decomposition="sequence"
)  # sequence for computational efficiency

def bc_block(block: daisy.Block):
    cap = fincap_mask[block.read_roi]
    cluster0 = structuralcollagen_mask[block.read_roi]
    cluster1 = fibrosis1_mask[block.read_roi]
    cluster2 = fibrosis2_mask[block.read_roi]
    cluster = cluster0 + cluster1 + cluster2
    eroded_cap = binary_erosion(cap, erode_kernel)
    dilated_cap = binary_dilation(cap, dilate_kernel)
    bc = (dilated_cap * (1 - eroded_cap)) * cluster
    bc = gaussian(bc.astype(float), sigma=2.0) > 0.5
    bc_mask[block.write_roi] = bc

block_roi = Roi((0, 0), (1000, 1000)) * fg_eroded_s0.voxel_size
bc_task = daisy.Task(
    "ID Bowman's Capsule",
    total_roi=s0_array.roi,
    read_roi=block_roi,
    write_roi=block_roi,
    read_write_conflict=False,
    num_workers=2,
    process_function=bc_block,
)
daisy.run_blockwise(tasks=[bc_task], multiprocessing=False)

print("Identifying Structural Collagen around Vessels")
# create masks for vessel
vessel10x = prepare_mask(zarr_path, s2_array, "vessel10x")
vessel10xdilated = prepare_mask(zarr_path, s2_array, "vessel10xdilated")
vessel40xdilated = prepare_mask(zarr_path, s0_array, "vessel40xdilated")

# downsample 40x vessel predictions to 10x for dilation
downsample_factor = 4
def downsample_block(block: daisy.Block):
    s0_data = vessel_mask[block.read_roi]
    new_shape = (
        s0_data.shape[0] // downsample_factor,
        downsample_factor,
        s0_data.shape[1] // downsample_factor,
        downsample_factor,
    )
    downsampled = s0_data.reshape(new_shape).mean(
        axis=(1, 3)
    )  # Average over grouped blocks
    vessel10x[block.write_roi] = downsampled


block_roi = Roi((0, 0), (1000, 1000)) * vessel_mask.voxel_size

downsample_task = daisy.Task(
    "blockwise_downsample",
    total_roi=vessel_mask.roi,
    read_roi=block_roi,
    write_roi=block_roi,
    read_write_conflict=False,
    num_workers=2,
    process_function=downsample_block,
)
daisy.run_blockwise(tasks=[downsample_task], multiprocessing=False)

# size of dilation of smallest object
base_size = 3
# Label connected components
filtered_array = remove_small_objects(
    vessel10x._source_data[:].astype(bool), min_size=2000
)
vessel10x._source_data[:] = remove_small_holes(filtered_array, area_threshold=5000)

labeled_array, num_features = label(vessel10x._source_data[:])

# Process each object separately
for obj_label in tqdm(range(1, labeled_array.max() + 1)):
    # Isolate object
    obj_mask = labeled_array == obj_label
    # Compute object size
    object_size = np.sum(obj_mask)
    # Define dilation size (larger for bigger objects)
    dilation_radius = base_size + int(np.log1p(object_size))  # Log-based scaling
    # Create structuring element (disk for circular dilation)
    struct_elem = disk(dilation_radius, decomposition="sequence")
    # Dilate the object and store in the output array
    vessel10xdilated._source_data[:] |= binary_dilation(obj_mask, struct_elem)

vessel10xdilated._source_data[:] = dask.array.clip(
    vessel10xdilated._source_data[:] - vessel10x._source_data[:], 0, 1
)

################################################
# create final fibrosis overlay: fibrosis1 + fibrosis2 from clustering - tuft, capsule, vessel, tubules
finfib_mask.data = (
    (fibrosis1_mask.data + fibrosis2_mask.data + structuralcollagen_mask.data)
    * (1 - vessel_mask.data)
    * (1 - vessel40xdilated.data)
    * (1 - dt_mask.data)
    * (1 - pt_mask.data)
    * (1 - fincap_mask.data)
    * (1 - tbm.data)
    * fg_eroded_s0.data
)

dask.array.store(finfib_mask.data, finfib_mask._source_data)

# create final collagen overlay
fincollagen_mask.data = (
    (structuralcollagen_mask.data
    + vessel40xdilated.data
    + tbm.data
    + bc_mask.data)
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
def fibscore_block(block: daisy.Block):
    # in data
    fib = finfib_mask[block.read_roi]
    fg_er = fg_eroded_s0[block.read_roi]
    vessel = vessel_mask[block.read_roi]
    cap = fincap_mask[block.read_roi]
    # calc positive pixels in mask
    fibpx = np.count_nonzero(fib)
    fgpx = fg_er * ~vessel.astype(bool) * ~cap.astype(bool)
    fgpx = np.count_nonzero(fgpx)

    # write to text file
    # create text file
    with open(Path(zarr_path.parent / "fibpx.txt"), "a") as f:
        f.writelines(f"{fibpx} \n")
    with open(Path(zarr_path.parent / "tissuepx.txt"), "a") as f:
        f.writelines(f"{fgpx} \n")

fibscore_task = daisy.Task(
    "fibscore calc",
    total_roi=s0_array.roi,
    read_roi=Roi((0, 0), (1000000, 1000000)),
    write_roi=Roi((0, 0), (1000000, 1000000)),
    read_write_conflict=False,
    num_workers=2,
    process_function=fibscore_block,
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
# score = finfib_mask / total_tissue
# with open(os.path.join(MODEL_DATA_PATH, "test_cases.txt"), "w") as f:
#  f.writelines("\n".join(score))s
