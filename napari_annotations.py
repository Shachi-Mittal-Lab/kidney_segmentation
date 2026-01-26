import napari
from pathlib import Path
from funlib.persistence import open_ds, prepare_ds
from fibrosis_score.mask_utils import prepare_mask
from fibrosis_score.daisy_blocks import tissuemask_upsample
import numpy as np
import tifffile
from PIL import Image
import dask
from dask.diagnostics import ProgressBar
from dask.array import coarsen, mean

# inputs #
zarr = Path("/home/riware/Desktop/mittal_lab/file_conversions/21-2013 A-1-9 Trich - 2021-03-22 15.05.44.zarr")
tiff_5x = zarr.parent / f"{zarr.stem}_mask.tiff"
desired_tissue_zarr = zarr.parent / f"{zarr.stem}_desiredtissue.zarr"
mask_5x = "desired_tissue_5x"
mask_40x = "desired_tissue_40x"

# Grab Zarr 
zarr_s0 = zarr / "raw" / "s0"
zarr_s3 = zarr / "raw" / "s3"
s0_array = open_ds(zarr_s0)
s3_array = open_ds(zarr_s3)

# visualize image to annotate
viewer = napari.Viewer()
viewer.add_image(s3_array.data, multiscale=False)
napari.run()  # generate annotations and save as tiff_5x

# convert tiff to zarr mask
# If already a tif, just read and resize; otherwise convert first
if tiff_5x.suffix.lower() in [".tif", ".tiff"]:
    image = tifffile.imread(tiff_5x)
else:
    image = np.array(Image.open(tiff_5x))

# prep zarr masks
# prepare_mask(zarr, s3_array, mask_5x)
mask_5x_array = prepare_ds(
    zarr / "mask" / f"{mask_5x}",
    s3_array.shape[0:2],
    s3_array.offset,
    s3_array.voxel_size,
    s3_array.axis_names[0:2],
    s3_array.units,
    mode="w",
    dtype=np.uint8,
)
mask_5x_array._source_data[:] = image

mask_40x_array = prepare_ds(
    zarr / "mask" / f"{mask_40x}",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
mask_40x_array._source_data[:] = 0

viewer = napari.Viewer()
viewer.add_labels(mask_5x_array.data, multiscale=False)
napari.run()  # check to be sure the annotations are there

upsampling_factor = mask_5x_array.voxel_size / mask_40x_array.voxel_size

tissuemask_upsample(mask_5x_array, mask_40x_array, s0_array, upsampling_factor)

# apply tissue mask to zarr and then create new zarr with mask applied
# format new zarr
units = ("nm", "nm")
desired_tissue = prepare_ds(
        desired_tissue_zarr / "raw" / "s0",
        s0_array.shape,
        s0_array.offset,
        s0_array.voxel_size,
        s0_array.axis_names,
        units,
        mode="w",
        dtype=np.uint8,
    )
desired_tissue._source_data[:] = 0

# Apply mask and save 
mask_40x_array_expanded = mask_40x_array.data[:,:,None]
print("Broadcasting mask")
mask_broadcasted = dask.array.broadcast_to(mask_40x_array_expanded, s0_array.shape)
print("Multiplying Image with Mask")
multiplication = (1 - mask_40x_array_expanded) * s0_array.data
store_multiplication = dask.array.store(multiplication, desired_tissue._source_data, execute=False)
print("Saving new Zarr")
dask.compute(store_multiplication)  

# replace all black pixels with white ones to not mess up the foreground/background

# Create a mask for black pixels
black_mask = dask.array.all(desired_tissue.data == 0, axis=-1)
# Replace black pixels with white
white_pixel = dask.array.full((3,), 255, dtype=np.uint8)
processed_image = dask.array.where(black_mask[..., None], white_pixel, desired_tissue._source_data)

# Compute the result
white_background_img = dask.array.store(processed_image, desired_tissue._source_data, execute=False)
dask.compute(white_background_img)

# Generate 20x, 10x, and 5x layers 
for i in range(1, 4):
    voxel_size = tuple((s0_array.voxel_size[0] * 2**i, s0_array.voxel_size[0] * 2**i))
    shape = tuple((s0_array.shape[0] // 2**i, s0_array.shape[1] // 2**i,3))
    # format data as funlib dataset
    raw = prepare_ds(
        desired_tissue_zarr / "raw" / f"s{i}",
        shape,
        s0_array.offset,
        voxel_size,
        s0_array.axis_names,
        units,
        mode="w",
        dtype=np.uint8,
    )
    # storage info
    #store_rgb = zarr.open(mask_applied_zarr / "raw" / f"s{i}", mode="w")
    prev_layer = open_ds(desired_tissue_zarr / "raw" / f"s{i-1}")

    # mean downsampling
    dask_array = coarsen(mean, prev_layer.data, {0: 2, 1: 2})

    # save to zarr
    with ProgressBar():
        dask.array.store(dask_array, raw._source_data)