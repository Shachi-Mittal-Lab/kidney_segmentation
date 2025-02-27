from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
import daisy
import dask
from dask.diagnostics import ProgressBar
from dask.array import coarsen, mean
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation, disk
from skimage.filters import gaussian
from scipy.ndimage import label
from skimage import measure
from patch_utils_dask import openndpi
import zarr
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from model import UNet, ConvBlock, Downsample, CropAndConcat, OutputConv
from PIL import Image

from pathlib import Path

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#ndpi_path = Path("/media/mrl/Data/pipeline_connection/ndpis/predict/BR22-2072-A-1-9-TRI - 2022-08-08 14.51.30.ndpi")
zarr_path = Path("/media/mrl/Data/pipeline_connection/ndpis/predicted/unet_testing/21-2002_region_model0_dataset0.zarr")
offset = (0, 0)
axis_names = [
    "x",
    "y",
    "c^",
]

s2_array = open_ds(zarr_path / "raw" / "s2")

model = torch.load("model_dataset1_flips.pt", weights_only=False)

# create mask for final inflammation overlay at 40x
caps2 = prepare_ds(
    zarr_path / "mask" / "caps3",
    s2_array.shape[0:2],
    s2_array.offset,
    s2_array.voxel_size,
    s2_array.axis_names[0:2],
    s2_array.units,
    mode="w",
    dtype=np.uint8,
)
caps2._source_data[:] = 0

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

# blockwise mask multiplications
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
    caps2[block.write_roi] = preds

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

#roi = Roi((0, 0), (7232000, 22216704))  # s0_array.roi
"""
# open highest pyramid level
dask_array0, x_res, y_res, units = openndpi(ndpi_path, 0)
s0_shape = dask_array0.shape
units = ("nm", "nm")
# grab resolution in cm, convert to nm, calculate for each pyramid level
voxel_size0 = Coordinate(int(1 / x_res * 1e7), int(1 / y_res * 1e7))
# format data as funlib dataset
raw = prepare_ds(
    zarr_path / "raw" / "s0",
    dask_array0.shape,
    offset,
    voxel_size0,
    axis_names,
    units,
    mode="w",
    dtype=np.uint8,
)
# storage info
store_rgb = zarr.open(zarr_path / "raw" / "s0")
dask_array = dask_array0.rechunk(raw.data.chunksize)

with ProgressBar():
    dask.array.store(dask_array, store_rgb)

for i in range(1, 4):
    # open the ndpi with openslide for info and tifffile as zarr
    dask_array, x_res, y_res, _ = openndpi(ndpi_path, i)
    # grab resolution in cm, convert to nm, calculate for each pyramid level
    voxel_size = Coordinate(int(1 / x_res * 1e7) * 2**i, int(1 / y_res * 1e7) * 2**i)
    expected_shape = tuple((s0_shape[0] // 2**i, s0_shape[1] // 2**i,3))
    print(f"expected shape: {expected_shape}")
    print(f"actual shape: {dask_array.shape}")

    # check shape is expected shape 
    if dask_array.shape == expected_shape:
        print("correct shape")
        # format data as funlib dataset
        raw = prepare_ds(
            zarr_path / "raw" / f"s{i}",
            dask_array.shape,
            offset,
            voxel_size,
            axis_names,
            units,
            mode="w",
            dtype=np.uint8,
        )
        # storage info
        store_rgb = zarr.open(zarr_path / "raw" / f"s{i}")
        dask_array = dask_array.rechunk(raw.data.chunksize)

        with ProgressBar():
            dask.array.store(dask_array, store_rgb)

        # open pyramid level and assign appropriate voxel size and units to the dataset
        #ds = zarr.open(zarr_path / "raw" / f"s{i}")
        #ds.attrs["voxel_size"] = voxel_size
        #ds.attrs["units"] = units
    else:
        voxel_size = tuple((voxel_size0[0] * 2**i, voxel_size0[0] * 2**i))
        # format data as funlib dataset
        raw = prepare_ds(
            zarr_path / "raw" / f"s{i}",
            expected_shape,
            offset,
            voxel_size,
            axis_names,
            units,
            mode="w",
            dtype=np.uint8,
        )
        # storage info
        store_rgb = zarr.open(zarr_path / "raw" / f"s{i}")
        prev_layer = open_ds(zarr_path / "raw" / f"s{i-1}")
        print(f"chunk shape: {prev_layer.chunk_shape}")

        # mean downsampling
        dask_array = coarsen(mean, prev_layer.data, {0: 2, 1: 2})

        # save to zarr
        with ProgressBar():
            dask.array.store(dask_array, store_rgb)



s0_array = open_ds(zarr_path / "raw" / "s0")

vessel_roimask = open_ds(zarr_path / "mask" / "vessel")
dt_roimask = open_ds(zarr_path / "mask" / "dt")
pt_roimask = open_ds(zarr_path / "mask" / "pt")
cap_roimask = open_ds(zarr_path / "mask" / "cap")
tuft_roimask = open_ds(zarr_path / "mask" / "tuft")
inflammation = open_ds(zarr_path / "mask" / "inflammation")
fibrosis1 = open_ds(zarr_path / "mask" / "fibrosis1")
fibrosis2 = open_ds(zarr_path / "mask" / "fibrosis2")
finfib = open_ds(zarr_path / "mask" / "finfib", "r+")
fininflamm = open_ds(zarr_path / "mask" / "fininflamm", "r+")
structural_collagen = open_ds(zarr_path / "mask" / "structuralcollagen")
fincollagen = open_ds(zarr_path / "mask" / "fincollagen", "r+")
fg_eroded_s0 = open_ds(zarr_path / "mask" / "foreground_eroded_s0")  # in s0
fincap = open_ds(zarr_path / "mask" / "fincap", "r+")
bc = open_ds(zarr_path / "mask" / "bc", "r+")
tbm = open_ds(zarr_path / "mask" / "tbm")


# create mask for final inflammation overlay at 40x
vesselcollagen = prepare_ds(
    zarr_path / "mask" / "vesselcollagen",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
vesselcollagen._source_data[:] = 0

vessel_collagen = open_ds(zarr_path / "mask" / "vesselcollagen", "r+")

structure = np.ones((3, 3), dtype=np.int8)  # allows any connection
erode_kernel = disk(
    40, decomposition="sequence"
)
dilate_kernel = disk(
    17, decomposition="sequence"
)  # sequence for computational efficiency

def vessel_block(block: daisy.Block):
    vessel_data = vessel_roimask[block.read_roi]
    # connected components capsule
    labels, ncomponents = label(vessel_data, structure)
    unique, counts = np.unique(labels, return_counts=True)
    small_obj = counts <= 150000 # was 150000
    to_remove = unique[small_obj]
    small_obj_mask = np.isin(labels, to_remove)
    vessel_data[small_obj_mask] = 0
    # make donut
    eroded_vessel = binary_erosion(vessel_data, erode_kernel)
    dilated_vessel = binary_dilation(vessel_data, dilate_kernel)
    donut = dilated_vessel * (1-eroded_vessel)
    # find Bowman's Capsule w/ cap & structural collagen overlap
    cluster = structural_collagen[block.read_roi]
    tbm_collagen = tbm[block.read_roi]
    vesselcollagen_id = donut * (1 - tbm_collagen) * cluster
    vessel_smooth = gaussian(vesselcollagen_id.astype(float), sigma=2.0) > 0.5
    vesselcollagen[block.write_roi] = vessel_smooth


block_roi = Roi((0, 0), (4000, 4000)) * fg_eroded_s0.voxel_size
vessel_task = daisy.Task(
    "ID vessel structural collagen",
    total_roi=s0_array.roi,
    read_roi=block_roi,
    write_roi=block_roi,
    read_write_conflict=False,
    num_workers=2,
    process_function=vessel_block,
)
daisy.run_blockwise(tasks=[vessel_task], multiprocessing=False)


"""