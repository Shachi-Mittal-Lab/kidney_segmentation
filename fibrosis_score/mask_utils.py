from funlib.geometry import Coordinate
from funlib.persistence import open_ds, prepare_ds

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
    fmask = prepare_mask(zarr_path, ref_array, "foreground")
    filled_fmask = prepare_mask(zarr_path, ref_array, "foreground_filled")
    eroded_fmask = prepare_mask(zarr_path, ref_array, "foreground_eroded")

    return fmask, filled_fmask, eroded_fmask


def prepare_clustering_masks(zarr_path, ref_array):
    fibrosis1_mask = prepare_mask(zarr_path, ref_array, "fibrosis1")
    fibrosis2_mask = prepare_mask(zarr_path, ref_array, "fibrosis2")
    inflammation_mask = prepare_mask(zarr_path, ref_array, "inflammation")
    structuralcollagen_mask = prepare_mask(zarr_path, ref_array, "structuralcollagen")

    return fibrosis1_mask, fibrosis2_mask, inflammation_mask, structuralcollagen_mask


def prepare_omniseg_masks(zarr_path, ref_array):
    pt_mask = prepare_mask(zarr_path, ref_array, "pt")
    dt_mask = prepare_mask(zarr_path, ref_array, "dt")
    ptc_mask = prepare_mask(zarr_path, ref_array, "ptc")
    vessel_mask = prepare_mask(zarr_path, ref_array, "vessel")
    return pt_mask, dt_mask, ptc_mask, vessel_mask


def prepare_postprocessing_masks(zarr_path, ref_array):
    tbm_mask = prepare_mask(zarr_path, ref_array, "tbm")
    bc_mask = prepare_mask(zarr_path, ref_array, "bc")
    fincap_mask = prepare_mask(zarr_path, ref_array, "fincap")
    finfib_mask = prepare_mask(zarr_path, ref_array, "finfib")
    fincollagen_mask = prepare_mask(zarr_path, ref_array, "fincollagen")
    fininflamm_mask = prepare_mask(zarr_path, ref_array, "fininflamm")

    return (
        tbm_mask,
        bc_mask,
        fincap_mask,
        finfib_mask,
        fincollagen_mask,
        fininflamm_mask,
    )



