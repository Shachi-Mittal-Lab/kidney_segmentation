# Repo Tools
from fibrosis_score.zarr_utils import ndpi_to_zarr_padding, svs_to_zarr_padding
from fibrosis_score.mask_utils import (
    prepare_mask,
    prepare_foreground_masks,
    prepare_seg_masks,
    prepare_postprocessing_masks,
    prepare_patch_mask,
    generate_patchcount_mask,
    generate_grid_mask,
)
from fibrosis_score.patch_utils_dask import foreground_mask
from fibrosis_score.cluster_utils import prepare_clustering
from fibrosis_score.pred_utils import pred_cortex
from fibrosis_score.processing_utils import fill_holes, erode, varied_vessel_dilation, print_gpu_usage
from fibrosis_score.daisy_blocks_nocluster import (
    remove_bg,
    model_prediction_lsds,
    upsample,
    tissuemask_upsample,
    id_tbm,
    id_bc,
    downsample_fibrosis_20x,
    clean_visualization,
    calculate_fibscore,
    calculate_inflammscore,
    remove_small_fib,
    background_cluster,
)

# Funke Lab Tools
from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi

# Tools
from pathlib import Path
import time
from tqdm import tqdm

import dask
import dask.array
from dask.array import coarsen, mean
import zarr
# import zarr.convenience
import numpy as np

from skimage.transform import integral_image
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
)

from skimage.filters import gaussian
from skimage.measure import label
from scipy.ndimage import gaussian_filter

import torch
import torchvision.transforms as T

def run_full_pipeline(
        input_path: Path,
        offset: Coordinate,
        axis_names: list,
):

    # check there is a GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device}")

    # identify input file
    input_filename = input_path.stem
    input_file_ext = input_path.suffix
    print(input_filename)
    start_time = time.time()

    # define image padding for u-net predictions
    padding = (5000, 5000, 0)

    # convert file to zarr if not a zarr
    if input_file_ext == ".ndpi":
        # name zarr/home/amninder/Desktop/project/Folder_2/subfolder
        zarr_path = input_path.with_suffix(".zarr")
        # convert ndpi to zarr array levels 40x, 20x, 10x, 5x, 2.5x, and rechunk
        ndpi_to_zarr_padding(input_path, zarr_path, offset, axis_names, padding)

    elif input_file_ext == ".svs": 
        # name zarr/home/amninder/Desktop/project/Folder_2/subfolder
        zarr_path = input_path.with_suffix(".zarr")
        # convert ndpi to zarr array levels 40x, 20x, 10x, 5x, 2.5x, and rechunk
        svs_to_zarr_padding(input_path, zarr_path, offset, axis_names, padding)

    elif input_file_ext == ".zarr":
        zarr_path = input_path

    else:
        print(f"File must be of format .ndpi, .svs, or .zarr.  File extension is {input_file_ext}. Skipping {input_filename}")
        return

    # grab 5x
    s3_array = open_ds(zarr_path / "raw" / "s3") # 5x

    print("Generating Cortex Mask")
    cortex_mask = prepare_mask(zarr_path, s3_array, "desired_cortex") 

    # begin u-net preds: define preprocessing & patch size
    inp_transforms = T.Compose(
        [
            T.Grayscale(),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
        ]
    )

    # size to feed to 5x u-net in pixels
    patch_shape_final = Coordinate(1408, 1408)
    # size to feed to 5x u-net in nm
    patch_size_final = patch_shape_final * s3_array.voxel_size  # size in nm
    # calculate size affected by padding of the 1056 x 1056 block
    padding_affected_shape = Coordinate(280,280)
    padding_affected_size = padding_affected_shape * s3_array.voxel_size  # size in nm

    print("Predicting Cortex")
    # load model
    print_gpu_usage(device)
    model = torch.load("model_rachel_14mar26_epoch2754.pt", weights_only=False)
    binary_head = torch.load("binaryhead_unet_cortexdataset1_5x_LSDs_2754.pt", weights_only=False)
    print_gpu_usage(device)
    model_prediction_lsds(cortex_mask, s3_array, patch_size_final, padding_affected_size, model, binary_head, device, "ID Cortex")
