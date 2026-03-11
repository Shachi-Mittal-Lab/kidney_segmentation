# Repo Tools
from fibrosis_score.zarr_utils import ndpi_to_zarr, svs_to_zarr

# Funke Lab Tools
from funlib.persistence import open_ds, Array
from funlib.geometry import Coordinate, Roi

# Tools
from pathlib import Path
from dask.diagnostics import ProgressBar
import numpy as np
import tifffile


def get_5x_tif(
        input_path: Path,
        offset: Coordinate,
        axis_names: list,
):

    # identify input file
    input_filename = input_path.stem
    input_file_ext = input_path.suffix
    print(input_filename)

    # convert file to zarr if not a zarr
    if input_file_ext == ".ndpi":
        # name zarr/home/amninder/Desktop/project/Folder_2/subfolder
        zarr_path = input_path.with_suffix(".zarr")
        # convert ndpi to zarr array levels 40x, 20x, 10x, 5x, and rechunk
        ndpi_to_zarr(input_path, zarr_path, offset, axis_names)

    elif input_file_ext == ".svs": 
        # name zarr/home/amninder/Desktop/project/Folder_2/subfolder
        zarr_path = input_path.with_suffix(".zarr")
        # convert ndpi to zarr array levels 40x, 20x, 10x, 5x, and rechunk
        svs_to_zarr(input_path, zarr_path, offset, axis_names)

    elif input_file_ext == ".zarr":
        zarr_path = input_path

    else:
        print(f"File must be of format .ndpi, .svs, or .zarr.  File extension is {input_file_ext}. Skipping {input_filename}")
        return

    # grab 5x
    s3_array = open_ds(zarr_path / "raw" / "s3")

    # save as tiff
    # Convert to numpy array and save as TIFF
    data = s3_array.data # Load into memory
    tiff_path = input_path.parent / Path(f"{input_filename}_5x.tif")
    tifffile.imwrite(tiff_path, data)

# slide offset (zero)
offset = (0, 0)
# axis names
axis_names = [
    "x",
    "y",
    "c^",
]

input_dir = Path("/home/riware/Desktop/mittal_lab/imgs_for_ann")

input_files = [x for x in input_dir.glob("*")]
print(f"input files: {input_files}")

for file in input_files:
    get_5x_tif(file, offset, axis_names)
