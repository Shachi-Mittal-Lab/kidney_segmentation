import zarr.convenience
from patch_utils_dask import openndpi, foreground_mask

# from patch_utils import foreground_mask
import dask
import dask.array
import zarr
import numpy as np
import cv2
from tqdm import tqdm
from skimage.transform import integral_image
from skimage.morphology import binary_erosion, binary_dilation, disk
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi


# inputs #

# file inputs
ndpi_path = Path(
    "/home/riware/Desktop/loose_files/pathmltest/BR21-2074-B-A-1-9-TRICHROME - 2022-11-09 17.38.50.ndpi"
)
zarr_path = Path("/home/riware/Desktop/loose_files/pathmltest/test.zarr")
zarr_float_path = Path("/home/riware/Desktop/loose_files/pathmltest/test_floats.zarr")

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
# cortex vs medulla model info
starting_model_path = "/home/riware/Documents/MittalLab/kidney_project/fibrosis_score/omniseg/cortexvsmedulla/starting_model.pt"
weights_path = "/home/riware/Documents/MittalLab/kidney_project/fibrosis_score/omniseg/cortexvsmedulla/model9_epoch2.pt"

###########

# convert ndpi to zarr array levels 40x, 20x, 10x, 5x
for i in range(0, 4):
    # open the ndpi with openslide for info and tifffile as zarr
    dask_array, x_res, y_res, units = openndpi(ndpi_path,i)
    # define units in nm since native is cm and then you get a (0,0) res
    units = ("nm", "nm")
    # grab resolution in cm, convert to nm, calculate for each pyramid level
    voxel_size = Coordinate(int(1/x_res * 1e7) * 2 ** i , int(1/y_res * 1e7) * 2 ** i)
    # format data as funlib dataset
    raw = prepare_ds( zarr_path / "raw" / f"s{i}", dask_array.shape, offset, voxel_size, axis_names, units, dtype=np.uint8)
    # storage info
    store_rgb = zarr.open(zarr_path / "raw" / f"s{i}")
    # store info with storage info
    dask.array.store(dask_array, store_rgb)
    # open pyramid level and assign appropriate voxel size and units to the dataset
    ds = zarr.open(zarr_path / "raw" / f"s{i}")
    ds.attrs["voxel_size"] = voxel_size
    ds.attrs["units"] = units

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
shrink_kernel = disk(5, decomposition="sequence")
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
mask_filled_array[:] = mask_filled

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
imask[:] = integral_mask

## random grab from 10x for vgg16 - 300x300 10x
# zarr_array0 = zarr.open(zarr_path / "raw" / "s0")
# voxel_size = Coordinate(int(1/ 44114.0 * 2 ** 0 * 1e7) , int(1/ 44114.0 * 2 ** 0 * 1e7))
# roi = Roi((3,8), (30000,30000)) * voxel_size # asking for 100 x 100 pixels in s2

# create patch mask: this marks every pixel that can be the starting pixel
# of a patch of the defined patch size that will encompass mostly tissue
patch_shape = Coordinate(150, 150)  # Will convention: shape in voxels
patch_size = patch_shape * imask.voxel_size  # size in nm
patch_roi = Roi(imask.roi.offset, imask.roi.shape - patch_size)
patch_counts = (
    imask[patch_roi]  # indexing with a roi gives you a plain numpy array
    + imask[patch_roi + patch_size]
    - imask[patch_roi + patch_size * Coordinate(0, 1)]
    - imask[patch_roi + patch_size * Coordinate(1, 0)]
)
patch_count_mask = (
    patch_counts >= patch_shape[0] * patch_shape[1] * 0.50
)  # 50% of voxels are tissue

# make grid on s3
patch_spacing = Coordinate(100, 100)

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
patch_mask[:] = patch_count_mask * gridmask
# dask.array.store(patch_count_mask, store_patch_mask)

coords = np.nonzero(patch_mask[:])
offsets = [
    patch_mask.roi.offset + Coordinate(a, b) * patch_mask.voxel_size
    for a, b in zip(*coords)
]

s2_array = open_ds(zarr_path / "raw" / "s2")

import time

with torch.no_grad():
    # load vgg16 weights
    checkpoint = torch.load(
        weights_path, map_location=torch.device("cpu"), weights_only=False
    )
    # load vgg16 architecture
    vgg16: torch.nn.Module = torch.load(
        starting_model_path, map_location=torch.device("cpu"), weights_only=False
    )
    # load weights into architecture
    vgg16.load_state_dict(checkpoint["model_state_dict"])

    # make dataset for mask 
    pred_mask = prepare_ds(
        zarr_path / "mask" / "cortexmedulla",
        s2_array.shape[0:2],
        s2_array.offset,
        s2_array.voxel_size,
        s2_array.axis_names[0:2],
        s2_array.units,
        mode="w",
        dtype=np.uint8,
    )
    # default pixel value to 2
    pred_mask[:] = 2

    # for each patch, send to vgg16 and write label to mask
    for offset in tqdm(offsets):
        t1 = time.time()
        # world units roi selection
        roi = Roi(offset, patch_size)
        voxel_roi = (roi - s2_array.offset) / s2_array.voxel_size
        patch_raw = s2_array[
            voxel_roi.begin[0] : voxel_roi.end[0], voxel_roi.begin[1] : voxel_roi.end[1]
        ]
        t2 = time.time()
        # format roi for vgg16
        in_data = T.Resize((224, 224))(
            torch.from_numpy(patch_raw).float().permute(2, 0, 1).unsqueeze(0) / 255
        )
        t3 = time.time()
        # predict
        pred = vgg16(in_data).squeeze(0).numpy().argmax()
        t4 = time.time()

        #  write to mask, accounting for roi overlap
        context = (patch_size - patch_spacing * s3_array.voxel_size) // 2
        slices = pred_mask._Array__slices(roi.grow(-context, -context))
        pred_mask._source_data[slices] = pred

        # pred_mask[roi.grow(-context, -context)] = pred

        t5 = time.time()
        print(f"{t2-t1:.3}, {t3-t2:.3}, {t4-t3:.3}, {t5-t4:.3}")

    # pred_mask[:] = pred_mask_tmp[:]
