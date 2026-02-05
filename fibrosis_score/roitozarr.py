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

## inputs ##
input_path = Path("/mnt/49c617b2-221d-486c-9542-4ff0dc297048/kidney/verylargeimages/BR22-2096-A-1-9-TRICHROME - 2022-11-11 17.42.10.ndpi")
roi_zarr = input_path.parent / f"{input_path.stem}_region.zarr"

#############
input_file_ext = input_path.suffix

# grab resolution from image 
def openndpi(ndpi_path):
    slide = openslide.OpenSlide(str(ndpi_path))
    # these resolutions are px/cm and will convert to nm/px later
    x_res = float(slide.properties["tiff.XResolution"])
    y_res = float(slide.properties["tiff.YResolution"])
    return x_res, y_res, units

def opensvs(svs_path):
    slide = openslide.OpenSlide(svs_path)
    # these resolutions are in micrometer/px and I convert to nm/px here
    x_res = float(slide.properties["openslide.mpp-x"]) * 1000
    y_res = float(slide.properties["openslide.mpp-y"]) * 1000
    return x_res, y_res, units

if __name__ == "__main__": 
    # open highest pyramid level
    if input_file_ext == ".ndpi":
        x_res, y_res = openndpi(input_path)
        # convert to nm/px
        voxel_size0 = Coordinate(int(1 / x_res * 1e7), int(1 / y_res * 1e7))
    if input_file_ext == ".svs": 
        x_res, y_res = opensvs(input_path)
        # no conversion needed
        voxel_size0 = Coordinate(int(x_res), int(y_res))

    slide = openslide.OpenSlide(str(input_path))
    max_width, max_height = slide.dimensions
    print(f"width: {max_width}, height: {max_height}")

    # preview slide 
    preview_level = 4
    max_prev_width, max_prev_height = slide.level_dimensions[preview_level]
    preview_region = slide.read_region(location=(0,0), level=preview_level, size=(max_prev_width, max_prev_height))

    plt.imshow(preview_region)
    plt.show()

    # DEFINE REGION #
    # left/right , up/down
    region = slide.read_region(location=(0, 25000),level=0,size=(77000, 25000))
    #################

    # Show Region #
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

    # format data as funlib dataset
    raw = prepare_ds(
        roi_zarr / "raw" / "s0",
        s0_shape,
        offset,
        voxel_size0,
        axis_names,
        units,
        mode="w",
        dtype=np.uint8,
    )
    # storage info
    store_rgb = zarr.open(roi_zarr / "raw" / "s0")

    with ProgressBar():
        dask.array.store(dask_array, store_rgb)

    for i in range(1, 4):
        expected_shape = tuple((s0_shape[0] // 2**i, s0_shape[1] // 2**i,3))
        voxel_size = tuple((voxel_size0[0] * 2**i, voxel_size0[0] * 2**i))
        # format data as funlib dataset
        raw = prepare_ds(
            roi_zarr / "raw" / f"s{i}",
            expected_shape,
            offset,
            voxel_size,
            axis_names,
            units,
            mode="w",
            dtype=np.uint8,
        )
        # storage info
        store_rgb = zarr.open(roi_zarr / "raw" / f"s{i}")
        prev_layer = open_ds(roi_zarr / "raw" / f"s{i-1}")
        # mean downsampling
        dask_array = coarsen(mean, prev_layer.data, {0: 2, 1: 2})
    # save to zarr
        with ProgressBar():
            dask.array.store(dask_array, store_rgb)
