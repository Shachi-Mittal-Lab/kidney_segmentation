# Repo Tools
from fibrosis_score.patch_utils_dask import openndpi, opensvs

# Funke Tools
from funlib.geometry import Coordinate
from funlib.persistence import open_ds, prepare_ds

# Tools
import numpy as np
import zarr
import dask
from dask.array import coarsen, mean
from dask.diagnostics import ProgressBar


def ndpi_to_zarr(ndpi_path, zarr_path, offset, axis_names):
    # open highest pyramid level
    print("Opening NDPI")
    dask_array0, x_res, y_res, units = openndpi(ndpi_path, 0)
    print("NDPI Opened")
    s0_shape = dask_array0.shape
    # native units are listed as "centimeter" but are 
    # px/cm so manually reassigning to nm
    units = ("nm", "nm")
    # grab resolution in px/cm, convert to nm/px, calculate for each pyramid level
    voxel_size0 = Coordinate(int(1 / x_res * 1e7), int(1 / y_res * 1e7))
    # format data as funlib dataset
    raw = prepare_ds(
        zarr_path / "raw" / "s0",
        dask_array0.shape,
        offset,
        voxel_size0,
        axis_names,
        units,
        mode="w",
        dtype=np.uint8,
    )
    # storage info
    dask_array = dask_array0.rechunk(raw.data.chunksize)
    store_raw = dask.array.store(dask_array, raw._source_data, compute=False)
    
    print("Rechunking & Saving")
    with ProgressBar():
        dask.compute(store_raw)

    for i in range(1, 4):
        # open the ndpi with openslide for info and tifffile as zarr
        try:
            dask_array, x_res, y_res, _ = openndpi(ndpi_path, i)
            # grab resolution in px/cm, convert to nm/px, calculate for each pyramid level
            voxel_size = Coordinate(int(1 / x_res * 1e7) * 2**i, int(1 / y_res * 1e7) * 2**i)
            expected_shape = tuple((s0_shape[0] // 2**i, s0_shape[1] // 2**i,3))
            print(f"expected shape: {expected_shape}")
            print(f"actual shape: {dask_array.shape}")

            # check shape is expected shape 
            if dask_array.shape == expected_shape:
                print("correct shape")
                # format data as funlib dataset
                raw = prepare_ds(
                    zarr_path / "raw" / f"s{i}",
                    dask_array.shape,
                    offset,
                    voxel_size,
                    axis_names,
                    units,
                    mode="w",
                    dtype=np.uint8,
                )
                # storage info
                print("Rechunking")
                dask_array = dask_array.rechunk(raw.data.chunksize)
                print("Saving")
                store_raw = dask.array.store(dask_array, raw._source_data, compute=False)
                with ProgressBar():
                    dask.compute(store_raw)

        
            else:
                voxel_size = tuple((voxel_size0[0] * 2**i, voxel_size0[0] * 2**i))
                # format data as funlib dataset
                raw = prepare_ds(
                    zarr_path / "raw" / f"s{i}",
                    expected_shape,
                    offset,
                    voxel_size,
                    axis_names,
                    units,
                    mode="w",
                    dtype=np.uint8,
                )
                # storage info
                prev_layer = open_ds(zarr_path / "raw" / f"s{i-1}")
                print(f"chunk shape: {prev_layer.chunk_shape}")

                # mean downsampling
                print("Downsampling Previous Layer")
                dask_array = coarsen(mean, prev_layer.data, {0: 2, 1: 2})

                # save to zarr
                print("Saving")
                store_raw = dask.array.store(dask_array, raw._source_data, compute=False)
                with ProgressBar():
                    dask.compute(store_raw)

        except TypeError:
            print(f"Layer {i} does not exist.  Generating")
            dask_array, x_res, y_res, _ = openndpi(ndpi_path, i-1)
            # grab resolution in px/cm, convert to nm/px, calculate for each pyramid level
            voxel_size = Coordinate(int(1 / x_res * 1e7) * 2**i, int(1 / y_res * 1e7) * 2**i)
            expected_shape = tuple((s0_shape[0] // 2**i, s0_shape[1] // 2**i,3))
            print(f"expected shape: {expected_shape}")

            # format data as funlib dataset
            raw = prepare_ds(
                zarr_path / "raw" / f"s{i}",
                expected_shape,
                offset,
                voxel_size,
                axis_names,
                units,
                mode="w",
                dtype=np.uint8,
            )
            # storage info
            store_rgb = zarr.open(zarr_path / "raw" / f"s{i}")
            prev_layer = open_ds(zarr_path / "raw" / f"s{i-1}")
            print(f"chunk shape: {prev_layer.chunk_shape}")
            # mean downsampling
            dask_array = coarsen(mean, prev_layer.data, {0: 2, 1: 2})
            # save to zarr
            store_raw = dask.array.store(dask_array, raw._source_data, compute=False)
            with ProgressBar():
                dask.compute(store_raw)
    return print("npdi conversion complete")

def ndpi_to_zarr_padding(ndpi_path, zarr_path, offset, axis_names, padding):
    # open highest pyramid level
    print("Opening NDPI")
    dask_array0, x_res, y_res, units = openndpi(ndpi_path, 0)
    print("NDPI Opened")
    offset_plus_channels = offset + (0,)
    s0_shape = dask_array0.shape
    s0_shape_padded = Coordinate(dask_array0.shape) + Coordinate(padding) + Coordinate(offset_plus_channels)
    # native units are listed as "centimeter" but are 
    # px/cm so manually reassigning to nm
    units = ("nm", "nm")
    # grab resolution in px/cm, convert to nm/px, calculate for each pyramid level
    voxel_size0 = Coordinate(int(1 / x_res * 1e7), int(1 / y_res * 1e7))
    # format data as funlib dataset
    raw = prepare_ds(
        zarr_path / "raw" / "s0",
        s0_shape_padded,
        offset,
        voxel_size0,
        axis_names,
        units,
        mode="w",
        dtype=np.uint8,
        fill_value=255
    )

    # storage info
    dask_array = dask_array0.rechunk(raw.data.chunksize)
    store_raw = dask.array.store(dask_array, raw._source_data, compute=False)
    
    print("Rechunking & Saving")
    with ProgressBar():
        dask.compute(store_raw)

    for i in range(1, 4):
        # open the ndpi with openslide for info and tifffile as zarr
        try:
            dask_array, x_res, y_res, _ = openndpi(ndpi_path, i)
            # grab resolution in px/cm, convert to nm/px, calculate for each pyramid level
            voxel_size = Coordinate(int(1 / x_res * 1e7) * 2**i, int(1 / y_res * 1e7) * 2**i)
            expected_padded_shape = tuple((s0_shape_padded[0] // 2**i, s0_shape_padded[1] // 2**i,3))
            expected_shape = tuple((s0_shape[0] // 2**i, s0_shape[1] // 2**i,3))
            print(f"expected image shape: {expected_shape}")
            print(f"expected padded shape: {expected_padded_shape}")
            print(f"actual shape: {dask_array.shape}")

            # check shape is expected shape 
            if dask_array.shape == expected_shape:
                print("correct shape")
                # format data as funlib  with padding

                raw = prepare_ds(
                    zarr_path / "raw" / f"s{i}",
                    expected_padded_shape,
                    offset,
                    voxel_size,
                    axis_names,
                    units,
                    mode="w",
                    dtype=np.uint8,
                    fill_value=255

                )
                # storage info
                print("Rechunking")
                dask_array = dask_array.rechunk(raw.data.chunksize)
                print("Saving")
                store_raw = dask.array.store(dask_array, raw._source_data, compute=False)
                with ProgressBar():
                    dask.compute(store_raw)

            else:
                voxel_size = tuple((voxel_size0[0] * 2**i, voxel_size0[0] * 2**i))
                # format data as funlib dataset
                raw = prepare_ds(
                    zarr_path / "raw" / f"s{i}",
                    expected_padded_shape,
                    offset,
                    voxel_size,
                    axis_names,
                    units,
                    mode="w",
                    dtype=np.uint8,
                    fill_value=255

                )
                # storage info
                prev_layer = open_ds(zarr_path / "raw" / f"s{i-1}")
                print(f"chunk shape: {prev_layer.chunk_shape}")

                # mean downsampling
                print("Downsampling Previous Layer")
                dask_array = coarsen(mean, prev_layer.data, {0: 2, 1: 2})

                # save to zarr
                print("Saving")
                store_raw = dask.array.store(dask_array, raw._source_data, compute=False)
                with ProgressBar():
                    dask.compute(store_raw)

        except TypeError:
            print(f"Layer {i} does not exist.  Generating")
            dask_array, x_res, y_res, _ = openndpi(ndpi_path, i-1)
            # grab resolution in px/cm, convert to nm/px, calculate for each pyramid level
            voxel_size = Coordinate(int(1 / x_res * 1e7) * 2**i, int(1 / y_res * 1e7) * 2**i)
            expected_padded_shape = tuple((s0_shape_padded[0] // 2**i, s0_shape_padded[1] // 2**i,3))
            print(f"expected padded shape: {expected_padded_shape}")

            # format data as funlib dataset
            raw = prepare_ds(
                zarr_path / "raw" / f"s{i}",
                expected_padded_shape,
                offset,
                voxel_size,
                axis_names,
                units,
                mode="w",
                dtype=np.uint8,
                fill_value=255

            )
            # storage info
            store_rgb = zarr.open(zarr_path / "raw" / f"s{i}")
            prev_layer = open_ds(zarr_path / "raw" / f"s{i-1}")
            print(f"chunk shape: {prev_layer.chunk_shape}")
            # mean downsampling
            dask_array = coarsen(mean, prev_layer.data, {0: 2, 1: 2})
            # save to zarr
            store_raw = dask.array.store(dask_array, raw._source_data, compute=False)
            with ProgressBar():
                dask.compute(store_raw)
    return print("npdi conversion complete")

def svs_to_zarr(svs_path, zarr_path, offset, axis_names):
    # open highest pyramid level
    dask_array0, x_res, y_res, units = opensvs(svs_path, 0)
    s0_shape = dask_array0.shape
    # units are natively ("nm", "nm") so no need to convert to get voxel size
    # convert to integer and calculate for each pyramid level
    voxel_size0 = Coordinate(int(x_res), int(y_res))
    # format data as funlib dataset
    raw = prepare_ds(
        zarr_path / "raw" / "s0",
        dask_array0.shape,
        offset,
        voxel_size0,
        axis_names,
        units,
        mode="w",
        dtype=np.uint8,
    )
    # storage info
    store_rgb = zarr.open(zarr_path / "raw" / "s0")
    dask_array = dask_array0.rechunk(raw.data.chunksize)

    with ProgressBar():
        dask.array.store(dask_array, store_rgb)

    for i in range(1, 4):
        # open the ndpi with openslide for info and tifffile as zarr
        try:
            dask_array, x_res, y_res, _ = opensvs(svs_path, i)
            # units are natively ("nm", "nm") so no need to convert to get voxel size
            # convert to integer and calculate for each pyramid level
            voxel_size0 = Coordinate(int(x_res), int(y_res))
            expected_shape = tuple((s0_shape[0] // 2**i, s0_shape[1] // 2**i, 3))
            print(f"expected shape: {expected_shape}")
            print(f"actual shape: {dask_array.shape}")

            # check shape is expected shape
            if dask_array.shape == expected_shape:
                print("correct shape")
                # format data as funlib dataset
                raw = prepare_ds(
                    zarr_path / "raw" / f"s{i}",
                    dask_array.shape,
                    offset,
                    voxel_size,
                    axis_names,
                    units,
                    mode="w",
                    dtype=np.uint8,
                )
                # storage info
                store_rgb = zarr.open(zarr_path / "raw" / f"s{i}")
                dask_array = dask_array.rechunk(raw.data.chunksize)

                with ProgressBar():
                    dask.array.store(dask_array, store_rgb)

            else:
                voxel_size = tuple((voxel_size0[0] * 2**i, voxel_size0[0] * 2**i))
                # format data as funlib dataset
                raw = prepare_ds(
                    zarr_path / "raw" / f"s{i}",
                    expected_shape,
                    offset,
                    voxel_size,
                    axis_names,
                    units,
                    mode="w",
                    dtype=np.uint8,
                )
                # storage info
                store_rgb = zarr.open(zarr_path / "raw" / f"s{i}")
                prev_layer = open_ds(zarr_path / "raw" / f"s{i-1}")
                print(f"chunk shape: {prev_layer.chunk_shape}")

                # mean downsampling
                factors = {0: 2, 1: 2}
                try:
                    dask_array = coarsen(mean, prev_layer.data, factors)
                except ValueError as e:
                    new_shape = tuple(
                        (
                            (prev_layer.data.shape[i] // factors[i]) * factors[i]
                            if i in factors
                            else prev_layer.data.shape[i]
                        )
                        for i in range(prev_layer.data.ndim)
                    )
                    dask_array_cropped = prev_layer.data[:new_shape[0], :new_shape[1], :new_shape[2]]
                    dask_array = coarsen(mean, dask_array_cropped, factors)
                # save to zarr
                with ProgressBar():
                    dask.array.store(dask_array, store_rgb)
        except TypeError as e:
            print(f"for layer {i}: {e}")
            print("Generating layer")
            prev_layer = open_ds(zarr_path / "raw" / f"s{i-1}")
            voxel_size = tuple((voxel_size0[0] * 2**i, voxel_size0[0] * 2**i))
            # format data as funlib dataset
            raw = prepare_ds(
                zarr_path / "raw" / f"s{i}",
                expected_shape,
                offset,
                voxel_size,
                axis_names,
                units,
                mode="w",
                dtype=np.uint8,
            )
            # storage info
            store_rgb = zarr.open(zarr_path / "raw" / f"s{i}")
            prev_layer = open_ds(zarr_path / "raw" / f"s{i-1}")
            print(f"chunk shape: {prev_layer.chunk_shape}")
            # mean downsampling
            factors = {0: 2, 1: 2}
            try:
                dask_array = coarsen(mean, prev_layer.data, factors)
            except ValueError as e:
                new_shape = tuple(
                    (
                        (prev_layer.data.shape[i] // factors[i]) * factors[i]
                        if i in factors
                        else prev_layer.data.shape[i]
                    )
                    for i in range(prev_layer.data.ndim)
                )
                dask_array_cropped = prev_layer.data[:new_shape[0], :new_shape[1], :new_shape[2]]
                dask_array = coarsen(mean, dask_array_cropped, factors)
            # save to zarr
            with ProgressBar():
                dask.array.store(dask_array, store_rgb)
    return print("svs conversion complete")


def omniseg_to_zarr(omniseg_path, zarr_mask, voxel_roi):
    roimask = np.load(omniseg_path)
    roimask = roimask[:, :, 0]
    roimask = roimask > 0
    zarr_mask._source_data[
        voxel_roi.begin[0] : voxel_roi.end[0],
        voxel_roi.begin[1] : voxel_roi.end[1],
    ] = roimask.T
