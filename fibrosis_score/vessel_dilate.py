from pathlib import Path
import zarr
import numpy as np

from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
import daisy
import dask

from scipy.ndimage import label
from skimage.morphology import binary_erosion, binary_dilation, disk, remove_small_objects, remove_small_holes
from tqdm import tqdm

zarr_path = Path("/media/mrl/Data/pipeline_connection/ndpis/predict/21-2002_region.zarr")
offset = (0, 0)
axis_names = [
    "x",
    "y",
    "c^",
]

s0_array = open_ds(zarr_path / "raw" / "s0")
s2_array = open_ds(zarr_path / "raw" / "s2")
vessel_mask = open_ds(zarr_path / "mask" / "vessel")


# create mask for downsampled vessel overlay at 10x
vessel10x = prepare_ds(
    zarr_path / "mask" / "vessel10x",
    s2_array.shape[0:2],
    s2_array.offset,
    s2_array.voxel_size,
    s2_array.axis_names[0:2],
    s2_array.units,
    mode="w",
    dtype=np.uint8,
)
vessel10x._source_data[:] = 0

# create mask for dilated vessel overlay at 10x
vessel10xdilated = prepare_ds(
    zarr_path / "mask" / "vessel10xdilated",
    s2_array.shape[0:2],
    s2_array.offset,
    s2_array.voxel_size,
    s2_array.axis_names[0:2],
    s2_array.units,
    mode="w",
    dtype=np.uint8,
)
vessel10xdilated._source_data[:] = 0

# create mask for dilated vessel overlay at 40x
vessel40xdilated = prepare_ds(
    zarr_path / "mask" / "vessel40xdilated",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
vessel40xdilated._source_data[:] = 0

downsample_factor = 4

# downsample 40x vessel predictions to 10x for dilation
def downsample_block(block: daisy.Block):
    s0_data = vessel_mask[block.read_roi]
    new_shape = (s0_data.shape[0] // downsample_factor, downsample_factor, s0_data.shape[1] // downsample_factor, downsample_factor)
    downsampled = s0_data.reshape(new_shape).mean(axis=(1, 3))  # Average over grouped blocks
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
filtered_array = remove_small_objects(vessel10x._source_data[:].astype(bool), min_size=2000)
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


vessel10xdilated._source_data[:] = dask.array.clip(vessel10xdilated._source_data[:] - vessel10x._source_data[:], 0, 1)
