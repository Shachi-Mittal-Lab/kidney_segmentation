from funlib.geometry import Coordinate
from funlib.persistence import prepare_ds
from funlib.geometry import Coordinate, Roi

import numpy as np


def prepare_mask(zarr_path, ref_array, mask_name):
    """
    This function prepares a binary zarr mask file
    with the dims of the reference array, discounting
    the channel dimensions and puts the mask in write
    mode so that the mask can be written to.  The mask
    is then filled with zeros to begin.
    """

    mask = prepare_ds(
        zarr_path / "mask" / f"{mask_name}",
        ref_array.shape[0:2],
        ref_array.offset,
        ref_array.voxel_size,
        ref_array.axis_names[0:2],
        ref_array.units,
        mode="w",
        dtype=np.uint64,
    )
    mask._source_data[:] = 0
    return mask

def prepare_patch_mask(zarr_path, ref_array, patch_count_mask, mask_name):
    """
    This function prepares a binary zarr mask file
    with the shape of the patch_count_mask array and the 
    offset, voxel_size, axis names, and units, of the ref_array.
    This puts the mask in write mode so that the mask can be
    written to.  The mask is then filled with zeros to begin.
    """
    patch_mask = prepare_ds(
        zarr_path / "mask" / f"{mask_name}",
        patch_count_mask.shape,
        ref_array.offset,
        ref_array.voxel_size,
        ref_array.axis_names[0:2],
        ref_array.units,
        mode="w",
        dtype=np.uint8,
    )
    patch_mask._source_data[:] = 0 
    return patch_mask


def prepare_foreground_masks(zarr_path, ref_array):
    """
    This function prepares foreground masks in 5x.
    The mask is filled with zeros to begin.
    """
    fmask = prepare_mask(zarr_path, ref_array, "foreground")
    filled_fmask = prepare_mask(zarr_path, ref_array, "foreground_filled")
    eroded_fmask = prepare_mask(zarr_path, ref_array, "foreground_eroded")

    return fmask, filled_fmask, eroded_fmask


def prepare_clustering_masks(zarr_path, ref_array):
    """
    This function prepares clustering masks in 40x.
    The mask is filled with zeros to begin.
    """
    fibrosis1_mask = prepare_mask(zarr_path, ref_array, "fibrosis1")
    fibrosis2_mask = prepare_mask(zarr_path, ref_array, "fibrosis2")
    inflammation_mask = prepare_mask(zarr_path, ref_array, "inflammation")
    structuralcollagen_mask = prepare_mask(zarr_path, ref_array, "structuralcollagen")

    return fibrosis1_mask, fibrosis2_mask, inflammation_mask, structuralcollagen_mask

def prepare_seg_masks(zarr_path, ref_array, magnification):
    """
    This function prepares segmentation masks in 10x
    for the outputs of binary u-nets.
    The mask is filled with zeros to begin.
    """
    tubule_mask = prepare_mask(zarr_path, ref_array, f"tubule_{magnification}")
    vessel_mask = prepare_mask(zarr_path, ref_array, f"vessel_{magnification}")
    cap_mask = prepare_mask(zarr_path, ref_array, f"cap_{magnification}")
    return tubule_mask, vessel_mask, cap_mask

def prepare_omniseg_masks(zarr_path, ref_array):
    """
    This function prepares segmentation masks in 40x
    for the outputs of Omni-Seg.
    The mask is filled with zeros to begin.
    """
    tubule_mask = prepare_mask(zarr_path, ref_array, "tubule")
    ptc_mask = prepare_mask(zarr_path, ref_array, "ptc")
    vessel_mask = prepare_mask(zarr_path, ref_array, "vessel")
    return tubule_mask, ptc_mask, vessel_mask


def prepare_postprocessing_masks(zarr_path, ref_array):
    """
    This function prepares segmentation masks in 40x
    for final masks after post-processing.
    The mask is filled with zeros to begin.
    """
    tbm_mask = prepare_mask(zarr_path, ref_array, "tbm")
    bc_mask = prepare_mask(zarr_path, ref_array, "bc")
    fincap_mask = prepare_mask(zarr_path, ref_array, "fincap")
    finfib_mask = prepare_mask(zarr_path, ref_array, "finfib")
    fincollagen_mask = prepare_mask(zarr_path, ref_array, "fincollagen")
    fincollagen_exclusion_mask = prepare_mask(zarr_path, ref_array, "fincollagen_exclusion")
    fininflamm_mask = prepare_mask(zarr_path, ref_array, "fininflamm")

    return (
        tbm_mask,
        bc_mask,
        fincap_mask,
        finfib_mask,
        fincollagen_mask,
        fincollagen_exclusion_mask,
        fininflamm_mask,
    )

def generate_patchcount_mask(integralmask, patchshape, tissueallowance):
    """
    This function generates a patch count mask: this marks every pixel that 
    can be the starting pixel of a patch of the defined patch size that will 
    encompass mostly tissue.  This intakes the integral mask calculated in
    the main script, the shape of the patch in pixels, and the allowance of 
    tissue pixels.  

    Args:
    integralmask: this is the integral mask that is used to calculate the 
    number of positive tissue pixels in a patch

    patchshape: this is the shape of the desired patch, as a Coordinate in
    pixels

    tissueallowance: This is the minimal percentage of tissue pixels required
    of a patch to be considered a patch to perform an operation on. E.g. 0.30 
    would be a minimum of 30% pixels tissue pixels in each patch.
    """

    patch_size = patchshape * integralmask.voxel_size  # size in nm
    patch_roi = Roi(integralmask.roi.offset, integralmask.roi.shape - patch_size)
    patch_counts = (
         integralmask[patch_roi]  # indexing with a roi gives you a plain numpy array
         + integralmask[patch_roi + patch_size]
         - integralmask[patch_roi + patch_size * Coordinate(0, 1)]
         - integralmask[patch_roi + patch_size * Coordinate(1, 0)]
         )
    patch_count_mask = (
         patch_counts >= patchshape[0] * patchshape[1] * tissueallowance
         )  # % of voxels are tissue
    return patch_count_mask, patch_size

def generate_grid_mask(patch_count_mask, patchspacing):
    """
    This function generates a grid mask using the patch spacing 
    as a Coordinate in pixels to determine the distances between 
    the positive pixels on the grid and the 
    patch_count_mask for dimensions of the mask.
    """
    # Give y shape then x shape - strange but works
    grid_x, grid_y = np.meshgrid(
        range(patch_count_mask.shape[1]), range(patch_count_mask.shape[0])
    )
    gridmask = grid_x % patchspacing[0] + grid_y % patchspacing[1]
    gridmask = gridmask == 0
    return gridmask
