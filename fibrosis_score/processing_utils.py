# Funke Lab Tools
from funlib.persistence import Array

# Tools
from skimage.morphology import (
    binary_erosion,
    binary_dilation,
    disk,
    remove_small_objects,
    remove_small_holes,
)
from scipy.ndimage import label
import numpy as np
from tqdm import tqdm
import dask
import torch


def fill_holes(inmask, filldisk, shrinkdisk):
    """
    Fill holes in mask by dilation followed by erosion.
    Uses "sequence" decomposition for computational efficiency.
    
    """
    fill_hole_kernel = disk(
        filldisk, decomposition="sequence"
    )
    shrink_kernel = disk(shrinkdisk, decomposition="sequence")
    mask_filled = binary_erosion(binary_dilation(inmask, fill_hole_kernel), shrink_kernel)
    return mask_filled

def erode(inmask, shrinkdisk): 
    """
    Erode mask.  Uses "sequence" decomposition for computational
    efficiency.
    """
    shrink_kernel = disk(shrinkdisk, decomposition="sequence")
    mask_eroded = binary_erosion(inmask, shrink_kernel)
    return mask_eroded

def varied_vessel_dilation(
        base_size: int,
        vessel10x: Array,
        vessel10xdilated: Array,
):
    # size of dilation of smallest object
    base_size = 5
    # Label connected components
    filtered_array = remove_small_objects(
        vessel10x._source_data[:].astype(bool), min_size=2000
    )
    vessel10x._source_data[:] = remove_small_holes(filtered_array, area_threshold=5000)

    labeled_array, num_features = label(vessel10x._source_data[:])

    # Process each object separately
    for obj_label in tqdm(range(1, labeled_array.max() + 1)):
        # Isolate object
        obj_mask = labeled_array == obj_label
        # Compute object size
        object_size = np.sum(obj_mask)
        # Define dilation size (larger for bigger objects)
        dilation_radius = base_size + int(np.log1p(5000 * object_size))  # Log-based scaling
        # Create structuring element (disk for circular dilation)
        struct_elem = disk(dilation_radius, decomposition="sequence")
        # Dilate the object and store in the output array
        vessel10xdilated._source_data[:] |= binary_dilation(obj_mask, struct_elem)

    vessel10xdilated._source_data[:] = dask.array.clip(
        vessel10xdilated._source_data[:] - vessel10x._source_data[:], 0, 1
    )
    return

def print_gpu_usage(device):
    print(f"Using device: {device}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    total = torch.cuda.get_device_properties(device).total_memory

    # Get current memory allocated by tensors
    allocated_memory = torch.cuda.memory_allocated(0)
    print(f"Allocated Memory: {allocated_memory / (1024**3):.2f} GB")

    # Get current memory managed by the caching allocator
    reserved_memory = torch.cuda.memory_reserved(0)
    print(f"Cached Memory: {reserved_memory / (1024**3):.2f} GB")

    # Get free space 
    free = reserved_memory - allocated_memory
    available = total - reserved_memory

    # Get maximum memory allocated (peak usage)
    max_allocated_memory = torch.cuda.max_memory_allocated(0)
    print(f"Max Allocated Memory (Peak): {max_allocated_memory / (1024**3):.2f} GB")

    # Get maximum memory reserved by the caching allocator
    max_reserved_memory = torch.cuda.max_memory_reserved(0)
    print(f"Max Reserved Memory (Peak): {max_reserved_memory / (1024**3):.2f} GB")

    # Print free and available memory
    print(f"Free Memory: {free / (1024**3):.2f} GB")
    print(f"Available Memory: {available / (1024**3):.2f} GB")


    # Print a comprehensive summary of CUDA memory statistics
    print("\nCUDA Memory Summary:")
    print(torch.cuda.memory_summary(device=None, abbreviated=False)) 
