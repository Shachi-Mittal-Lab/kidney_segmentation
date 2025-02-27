import zarr.convenience
from patch_utils_dask import openndpi, foreground_mask
import dask
import dask.array 
import zarr
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dask.array import coarsen, mean
from pathlib import Path



from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi
from dask.diagnostics import ProgressBar
import openslide
from matplotlib import pyplot as plt

ndpi_path = Path("/media/mrl/Data/pipeline_connection/ndpis/predict/BR22-2091-A-1-9-TRICHROME - 2022-11-11 16.49.25.ndpi")
zarr_path = Path("/media/mrl/Data/pipeline_connection/ndpis/predict/22-2091_region.zarr")

def openndpi(ndpi_path, pyramid_level):
    slide = openslide.OpenSlide(str(ndpi_path)) #Eric edit
    x_res = float(slide.properties["tiff.XResolution"])
    y_res = float(slide.properties["tiff.YResolution"])
    units = slide.properties["tiff.ResolutionUnit"]
    return x_res, y_res, units

# open highest pyramid level
x_res, y_res, units = openndpi(ndpi_path, 0)

# read region & convert to RGB from RGBA and np to dask
slide = openslide.OpenSlide(str(ndpi_path))
region = slide.read_region(location=(5000,19000),level=0,size=(40000,14000))
plt.imshow(region)
plt.show()
s0_array = np.array(region)[:,:,:3]
dask_array = dask.array.from_array(s0_array, chunks="auto")
s0_shape = s0_array.shape

# grab resolution in cm, convert to nm, calculate for each pyramid level
units = ("nm", "nm")
offset = (0, 0)
axis_names = [
    "x",
    "y",
    "c^",
]
voxel_size0 = Coordinate(int(1 / x_res * 1e7), int(1 / y_res * 1e7))

# format data as funlib dataset
raw = prepare_ds(
    zarr_path / "raw" / "s0",
    s0_shape,
    offset,
    voxel_size0,
    axis_names,
    units,
    mode="w",
    dtype=np.uint8,
)
# storage info
store_rgb = zarr.open(zarr_path / "raw" / "s0")

with ProgressBar():
    dask.array.store(dask_array, store_rgb)

for i in range(1, 4):
    expected_shape = tuple((s0_shape[0] // 2**i, s0_shape[1] // 2**i,3))
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
    # mean downsampling
    dask_array = coarsen(mean, prev_layer.data, {0: 2, 1: 2})
 # save to zarr
    with ProgressBar():
        dask.array.store(dask_array, store_rgb)
