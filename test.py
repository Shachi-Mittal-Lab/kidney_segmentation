from fibrosis_score.daisy_blocks import model_prediction_lsds
from fibrosis_score.mask_utils import prepare_seg_masks
from fibrosis_score.processing_utils import print_gpu_usage

# Funke Lab Tools
import daisy
from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi

from pathlib import Path
import torch

# check there is a GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using {device}")

zarr_path = Path("/home/riware/Desktop/mittal_lab/testing/21-2002 A-1-9 Trich - 2021-03-22 12.36.47.zarr")

s2_array = open_ds(zarr_path / "raw" / "s2")

tubule_mask_10x, vessel_mask_10x, cap_mask_10x = prepare_seg_masks(zarr_path, s2_array, "10x")

# size to feed to u-nets in pixels
patch_shape_final = Coordinate(1056, 1056)
# size to feed to u-nets in nm
patch_size_final = patch_shape_final * s2_array.voxel_size  # size in nm
# calculate size affected by padding of the 1056 x 1056 block
padding_affected_shape = Coordinate(208, 208)
padding_affected_size = padding_affected_shape * s2_array.voxel_size  # size in nm

# load glom model
print_gpu_usage(device)
model = torch.load("model_unet_dataset5_01Jan2026_glom0_LSDs_final.pt", weights_only=False)
binary_head = torch.load("binaryhead_unet_dataset5_01Jan2026_glom0_LSDs_final.pt", weights_only=False)
print_gpu_usage(device)

model_prediction_lsds(cap_mask_10x, s2_array, patch_size_final, padding_affected_size, model, binary_head, device, "Cap ID" )