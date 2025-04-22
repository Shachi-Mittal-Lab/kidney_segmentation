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
zarr_path = Path("/media/mrl/Data/pipeline_connection/ndpis/21-2001_region.zarr")
#zarr_path = Path("/media/mrl/Data/pipeline_connection/ndpis/full_pipeline_preds_originalunet/21-2007_region/21-2007_region.zarr")
offset = (0, 0)
axis_names = [
    "x",
    "y",
    "c^",
]

s2_array = open_ds(zarr_path / "raw" / "s2")
s0_array = open_ds(zarr_path / "raw" / "s0")

model = torch.load("model_dataset4_dt0_400.pt", weights_only=False)

# create mask for final dt overlay at 40x
dt_unet = prepare_ds(
    zarr_path / "mask" / "dt_unet_2_400",
    s2_array.shape[0:2],
    s2_array.offset,
    s2_array.voxel_size,
    s2_array.axis_names[0:2],
    s2_array.units,
    mode="w",
    dtype=np.uint8,
)
dt_unet._source_data[:] = 0

# create mask for final dt overlay at 40x
dt_unet_40x = prepare_ds(
    zarr_path / "mask" / "dt_unet_40x_2_400",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
dt_unet_40x._source_data[:] = 0

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
def dt_id_block(block: daisy.Block):
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
    dt_unet[block.write_roi] = preds

dt_id_task = daisy.Task(
    "DT ID",
    total_roi=s2_array.roi,
    read_roi=Roi((0,0), patch_size_final), # (offset, shape)
    write_roi=Roi((0,0), patch_size_final),
    read_write_conflict=False,
    num_workers=2,
    process_function=dt_id_block
)
daisy.run_blockwise(tasks=[dt_id_task], multiprocessing=False)

# blockwise upsample to 40x
# upsample 5x eroded tissue mask to use to filter predictions
upsampling_factor = (4,4)
def upsample_block(block: daisy.Block):
    s2_data = dt_unet[block.read_roi]
    s0_data = s2_data
    for axis, reps in enumerate(upsampling_factor):
        s0_data = np.repeat(s0_data, reps, axis=axis)
    dt_unet_40x[block.write_roi] = s0_data

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