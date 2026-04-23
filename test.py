import tifffile
import numpy as np
from pathlib import Path

from fibrosis_score.daisy_blocks_newcortexmodel import calculate_fibscore
# Funke Lab Tools
from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi
from fibrosis_score.daisy_blocks_newcortexmodel import upsample
from fibrosis_score.mask_utils_newcortexmodel import prepare_mask

zarr_path = Path("/media/mrl/Data/pipeline_connection/ndpis/test_new_pipeline_21Apr2026/21-2004 A-1-9 Trich - 2021-03-22 12.58.20.zarr")
input_filename = zarr_path.stem


s2_array = open_ds(zarr_path / "raw" / "s2", mode='a')
s3_array = open_ds(zarr_path / "raw" / "s3", mode='a')

eroded_cortex_5x = open_ds(zarr_path / "mask" / "cortex_eroded", mode='a')
eroded_cortex_10x = prepare_mask(zarr_path, s2_array, "eroded_cortex_10x_new") 


upsampling_factor = s3_array.voxel_size / s2_array.voxel_size

upsample(eroded_cortex_5x, eroded_cortex_10x, upsampling_factor, s3_array, s2_array)