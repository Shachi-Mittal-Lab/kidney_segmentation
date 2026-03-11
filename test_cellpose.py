# Funke Lab Tools
import daisy
from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi

# Cellpose Tools
from cellpose import models, io
from cellpose.io import imread

# General Tools
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

# inputs 
zarr_path = Path("/media/mrl/Data/pipeline_connection/ndpis/nuclei_cellpose_testing/21-2008 A-1-9 Trich - 2021-03-22 13.47.31.zarr")
cellpose_model = Path("CP_20260223_084158")
patch_shape_final = Coordinate(2240, 2240)

# open model and raw data 
s0_array = open_ds(zarr_path / "raw" / "s0")
mask = open_ds(zarr_path / "mask" / "nuclei_40x", mode="a")
model = models.CellposeModel(gpu=True, pretrained_model=str(cellpose_model))
patch_size_final = patch_shape_final * s0_array.voxel_size  # size in nm

# list of files
# PUT PATH TO YOUR FILES HERE!
"""
file_paths = [p for p in files.iterdir() if p.is_file()]

imgs = [imread(f) for f in file_paths]

nimg = len(imgs)

for img in imgs: 
    print(dir(img))

imgs = np.array(imgs)
arr_zcyx = imgs.transpose(0, 3, 2, 1)

masks, flows, styles = model.eval(arr_zcyx)

print(masks.shape)

plt.imshow(masks)
plt.show()
"""

def model_prediction_cellpose(
    mask: Array,
    s0_array: Array,
    patch_size_final: Array,
    model,
    task: str,
):

    def process_block(block: daisy.Block):
        # in data slice
        inslices = s0_array._Array__slices(block.read_roi)
        # grayscale and invert img
        img = Image.fromarray(s0_array[inslices])
        img = ImageOps.invert(img.convert("L"))
        arr = np.asarray(img) # X,Y,C
        masks, flows, styles = model.eval(
            arr, 
            channels=[0, 0],        # single-channel grayscale
            channel_axis=-1,        # last axis holds channels
            z_axis=None,            # 2D image
            normalize=True,
        )
        masks = masks >> 0
        mask[block.write_roi] = masks

    pred_task = daisy.Task(
        task,
        total_roi=s0_array.roi,
        read_roi=Roi((0, 0), patch_size_final),  # (offset, shape)
        write_roi=Roi((0, 0), patch_size_final),
        read_write_conflict=False,
        num_workers=2,
        process_function=process_block,
    )
    daisy.run_blockwise(tasks=[pred_task], multiprocessing=False)
    return

model_prediction_cellpose(mask, s0_array,patch_size_final, "cellpose prediction")