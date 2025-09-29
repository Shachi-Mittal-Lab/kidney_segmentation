from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi
from pathlib import Path
import numpy as np
import dask.array as da

input_path = Path("/media/mrl/Data/pipeline_connection/ndpis/full_pipeline_preds_DrSetty_04Sep2025/bleh")

input_files = [x for x in input_path.glob("*")]
print(f"input files: {input_files}")

for zarr_path in input_files:
    # raw data and metadata
    s0_array = open_ds(zarr_path / "raw" / "s0")
    s0_shape = s0_array.shape
    offset = s0_array.offset
    voxel_size0 = s0_array.voxel_size
    axis_names = s0_array.axis_names
    units = s0_array.units
    """
    ### CREATE TUBULE RGB VISUALIZATION ###
    # format image of tubules
    tubules_in_color = prepare_ds(
        zarr_path / "mask" / "tubules_in_color",
        s0_shape,
        offset,
        voxel_size0,
        axis_names,
        units,
        mode="w",
        dtype=np.uint8,
    )

    # save image of just tubule class 
    tubules_40x =  open_ds(zarr_path / "mask" / "pt_40x")
    tubules_40x_expanded = tubules_40x.data[:,:,None]
    mask_broadcasted = da.broadcast_to(tubules_40x_expanded, s0_array.shape)
    tubules_in_color.data = tubules_40x_expanded * s0_array.data
    da.store(tubules_in_color.data, tubules_in_color._source_data)

    ### CREATE TBM RGB VISUALIZATION ###
    # format image of tbm
    tbm_in_color = prepare_ds(
        zarr_path / "mask" / "tbm_in_color",
        s0_shape,
        offset,
        voxel_size0,
        axis_names,
        units,
        mode="w",
        dtype=np.uint8,
    )

    # save image of just tbm 
    tbm =  open_ds(zarr_path / "mask" / "tbm")
    tbm_expanded = tbm.data[:,:,None]
    mask_broadcasted = da.broadcast_to(tbm_expanded, s0_array.shape)
    tbm_in_color.data = tbm_expanded * s0_array.data
    da.store(tbm_in_color.data, tbm_in_color._source_data)
    """

    ### CREATE FIBROSIS RGB VISUALIZATION ###
    # format image of finfib
    finfib_in_color = prepare_ds(
        zarr_path / "mask" / "finfib_in_color",
        s0_shape,
        offset,
        voxel_size0,
        axis_names,
        units,
        mode="w",
        dtype=np.uint8,
    )

    # save image of just fibrosis class 
    finfib =  open_ds(zarr_path / "mask" / "finfib")
    finfib_expanded = finfib.data[:,:,None]
    mask_broadcasted = da.broadcast_to(finfib_expanded, s0_array.shape)
    finfib_in_color.data = finfib_expanded * s0_array.data
    da.store(finfib_in_color.data, finfib_in_color._source_data)

    """
    ### CREATE STRUCTURAL COLLAGEN RGB VISUALIZATION ###
    # format image of structural collagen
    fincollagen_in_color = prepare_ds(
        zarr_path / "mask" / "fincollagen_in_color",
        s0_shape,
        offset,
        voxel_size0,
        axis_names,
        units,
        mode="w",
        dtype=np.uint8,
    )

    # save image of just fin collagen
    fincollagen =  open_ds(zarr_path / "mask" / "fincollagen")
    fincollagen.data = fincollagen.data > 0
    fincollagen_expanded = fincollagen.data[:,:,None]
    mask_broadcasted = da.broadcast_to(fincollagen_expanded, s0_array.shape)
    fincollagen_in_color.data = fincollagen_expanded * s0_array.data
    da.store(fincollagen_in_color.data, fincollagen_in_color._source_data)
    """

    ### CREATE STRUCTURAL COLLAGEN RGB VISUALIZATION ###
    # format image of structural collagen
    fincollagen_exclusion_in_color = prepare_ds(
        zarr_path / "mask" / "fincollagen_exclusion_in_color",
        s0_shape,
        offset,
        voxel_size0,
        axis_names,
        units,
        mode="w",
        dtype=np.uint8,
    )

    # save image of just fin collagen exclusion 
    fincollagen_exclusion =  open_ds(zarr_path / "mask" / "fincollagen_exclusion")
    fincollagen_exclusion.data = fincollagen_exclusion.data > 0
    fincollagen_exclusion_expanded = fincollagen_exclusion.data[:,:,None]
    mask_broadcasted = da.broadcast_to(fincollagen_exclusion_expanded, s0_array.shape)
    fincollagen_exclusion_in_color.data = fincollagen_exclusion_expanded * s0_array.data
    da.store(fincollagen_exclusion_in_color.data, fincollagen_exclusion_in_color._source_data)

    """
    ### CREATE HISTOLOGICAL SEGMENTATION RGB VISUALIZATION ###
    # format image of hist mask
    histseg_in_color = prepare_ds(
        zarr_path / "mask" / "histseg_in_color",
        s0_shape,
        offset,
        voxel_size0,
        axis_names,
        units,
        mode="w",
        dtype=np.uint8,
    )

    # histological segmentation removed
    tubules_40x =  open_ds(zarr_path / "mask" / "pt_40x")
    fincap_40x =  open_ds(zarr_path / "mask" / "fincap")
    vessels_40x =  open_ds(zarr_path / "mask" / "vessel_40x")
    hist_mask = tubules_40x.data + fincap_40x.data + vessels_40x.data
    hist_mask = hist_mask > 0
    histmask = 1 - hist_mask

    # save image of just histological segmentation class 
    hist_mask_expanded = hist_mask[:,:,None]
    mask_broadcasted = da.broadcast_to(hist_mask_expanded, s0_array.shape)
    histseg_in_color.data = mask_broadcasted * s0_array.data
    da.store(histseg_in_color.data, histseg_in_color._source_data)


    ### CREATE HISTOLOGICAL SEGMENTATION + INFLAMMATION RGB VISUALIZATION ###

    # format image of hist mask
    histseginflamm_in_color = prepare_ds(
        zarr_path / "mask" / "histseginflamm_in_color",
        s0_shape,
        offset,
        voxel_size0,
        axis_names,
        units,
        mode="w",
        dtype=np.uint8,
    )

    # histological segmentation removed + inflammation
    inflammation_40x =  open_ds(zarr_path / "mask" / "fininflamm")
    histinflamm_mask = tubules_40x.data + fincap_40x.data + vessels_40x.data + inflammation_40x.data
    histinflamm_mask = histinflamm_mask > 0
    histinflamm_mask = 1 - histinflamm_mask

    # save image of just histological segmentation class 
    histseginflamm_expanded = histinflamm_mask[:,:,None]
    mask_broadcasted = da.broadcast_to(histseginflamm_expanded, s0_array.shape)
    histseginflamm_in_color.data = mask_broadcasted * s0_array.data
    da.store(histseginflamm_in_color.data, histseginflamm_in_color._source_data)
    """
    print(f"Finished {zarr_path}")