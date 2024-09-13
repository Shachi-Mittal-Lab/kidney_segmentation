# create a new conda environment with python>=3.11
# pip install funlib.show.neuroglancer
# pip install funlib.persistence

from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi
import numpy as np

# get your data from somewhere
test_array1 = np.arange(0,300000, 1).reshape(3, 500,200) / 300000
voxel_size = (20,50)
offset = (0,0)
# we give non physical dimensions a little "^" in the axis name
axis_names = ["c^", "x", "y"]
units = ["nm", "nm"]

raw_s0 = prepare_ds("test_data.zarr/raw/s0", test_array1.shape, offset, voxel_size, axis_names, units)
# this is convenient for writing our data, not the best for big data
# for big data you will want to use something like `dask.array.store("path/to/data.zarr/array")`
# after running `prepare_ds` to initialize array with metadata
raw_s0[:] = test_array1

# make a second levvel in pyramid downsampled by (2, 2)
test_array1 = np.random.randn(3, 250,100)
voxel_size = (40,100)
raw_s1 = prepare_ds("test_data.zarr/raw/s1", test_array1.shape, offset, voxel_size, axis_names, units)
raw_s1[:] = test_array1


# you want to select data from the same region of two different scale levels
# region you're interested in:
roi = Roi((0,0), (100,100)) * raw_s0.voxel_size # asking for 100 x 100 pixels in s0
assert raw_s0[roi].shape == (3,100,100)
roi = Roi((0,0), (100,100)) * raw_s1.voxel_size # asking for 100 x 100 pixels in s1
assert raw_s0[roi].shape == (3,200,200)

# this data will line up perfectly once accounting for voxel size
roi = Roi((0,0), (100,100)) * raw_s0.voxel_size # asking for 100 x 100 pixels in s0
assert raw_s0[roi].shape == (3,100,100)
assert raw_s1[roi].shape == (3,50,50)

# if not multiplied by voxel size, it is still interpreted as nanometers
# Roi(offset, shape)
roi = Roi((0,0), (100,100)) # directly asking for 100 nm x 100 nm
assert raw_s0[roi].shape == (3,5,2)
try:
    # breaks because you can't have 2.5 pixels
    assert raw_s1[roi].shape == (3,2.5,1)
except AssertionError:
    print("This should have failed")


# if we run this, we can then run from the command line: `neuroglancer -d test_data.zarr/raw/s0`
# to visualize
# exit()

# get your data from somewhere
test_array2 = np.arange(0, 750, 1).reshape(3,25,10) / 750
voxel_size = (20,50)
offset = (500,500)
# we give non physical dimensions a little "^" in the axis name
axis_names = ["c^", "x", "y"]
units = ["nm", "nm"]

raw_s0 = prepare_ds("test_data.zarr/raw2/s0", test_array2.shape, offset, voxel_size, axis_names, units)
# this is convenient for writing our data, not the best for big data
# for big data you will want to use something like `dask.array.store("path/to/data.zarr/array")`
# after running `prepare_ds` to initialize array with metadata
raw_s0[:] = test_array2

# world units. The usual interface for our data
print(f"roi: {raw_s0.roi}")

# units per voxel
print(f"voxel_size: {raw_s0.voxel_size}")

# shape of your underlying data (voxel units). we don't use this much
print(f"shape: {raw_s0.shape}")

# any mathematical operation on a roi applies to both offset and shape
# plus/minus: shift, multiplication/division: scale
print(f"roi in voxel units: {raw_s0.roi / raw_s0.voxel_size}")
