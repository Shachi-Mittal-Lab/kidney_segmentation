# Repo Tools
from fibrosis_score.zarr_utils import ndpi_to_zarr, omniseg_to_zarr
from fibrosis_score.mask_utils import (
    prepare_mask,
    prepare_foreground_masks,
    prepare_clustering_masks,
    prepare_seg_masks,
    prepare_postprocessing_masks,
    prepare_patch_mask,
    generate_patchcount_mask,
    generate_grid_mask,
)
from fibrosis_score.patch_utils_dask import foreground_mask
from fibrosis_score.cluster_utils import prepare_clustering, grayscale_cluster
from fibrosis_score.pred_utils import pred_cortex
from fibrosis_score.processing_utils import fill_holes, erode, varied_vessel_dilation, print_gpu_usage
from fibrosis_score.daisy_blocks import (
    model_prediction,
    model_prediction_rgb,
    model_prediction_lsds,
    upsample,
    tissuemask_upsample,
    id_tbm,
    id_bc,
    downsample_vessel,
    calculate_fibscore,
    calculate_inflammscore,
    remove_small_fib,
)

# Funke Lab Tools
import daisy
from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi

# Tools
import os
import subprocess
import sys
from pathlib import Path

import dask
import dask.array
from dask.diagnostics import ProgressBar
from dask.array import coarsen, mean
import zarr
import zarr.convenience

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from skimage.transform import integral_image
from skimage.morphology import (
    binary_erosion,
    binary_dilation,
    disk,
    remove_small_objects,
    remove_small_holes,
)
from skimage.filters import gaussian
from scipy.ndimage import label

import torch
import torchvision.transforms as T

def run_full_pipeline(
        input_path: Path,
        png_path: Path,
        threshold: int,
        offset: Coordinate,
        axis_names: list,
        gray_cluster_centers: Path,
        output_directory: Path,
        omni_seg_path: Path,
        subprocesswd: Path,
):

    # check there is a GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device}")

    # Eric edit:no pngs before it was created
    #os.makedirs("/media/mrl/Data/pipeline_connection/pngs", exist_ok=True)

    ###########
    input_filename = input_path.stem
    input_file_ext = input_path.suffix
    print(input_filename)

    # convert file to zarr if not a zarr
    if input_file_ext == ".ndpi":
        # name zarr/home/amninder/Desktop/project/Folder_2/subfolder
        zarr_path = input_path.with_suffix(".zarr")
        # convert ndpi to zarr array levels 40x, 20x, 10x, 5x, and rechunk
        ndpi_to_zarr(input_path, zarr_path, offset, axis_names)
    elif input_file_ext == ".zarr":
        zarr_path = input_path
    else:
        print(f"File must be of format .ndpi or .zarr.  File extension is {input_file_ext}. Skipping {input_filename}")
        return
    
    # grab 5x, 10x levels & calculate foreground masks
    s3_array = open_ds(zarr_path / "raw" / "s3")
    s2_array = open_ds(zarr_path / "raw" / "s2")

    print("Generating Foreground Masks")
    fmask, filled_fmask, eroded_fmask = prepare_foreground_masks(zarr_path, s3_array)
    mask = foreground_mask(s3_array.data, threshold)
    store_fgbg = zarr.open(zarr_path / "mask" / "foreground")
    dask.array.store(mask, store_fgbg)

    # fill holes & erode foreground mask
    filled_fmask._source_data[:] = fill_holes(mask, filldisk=31, shrinkdisk=28)
    eroded_fmask._source_data[:] = erode(filled_fmask._source_data[:], shrinkdisk=40)

    # create integral mask of filled foreground mask and save
    imask = prepare_mask(zarr_path, s3_array, "integral_foreground")
    integral_mask = integral_image(filled_fmask._source_data[:])
    imask._source_data[:] = integral_mask

    # create patch count mask for Cortex model
    patch_shape = Coordinate(224, 224)
    patch_count_mask, patch_size = generate_patchcount_mask(
        imask, patch_shape, tissueallowance=0.30
    )
    # generate gridmask
    patch_spacing = Coordinate(112, 112)
    gridmask = generate_grid_mask(patch_count_mask, patch_spacing)

    # create patchmask
    patch_mask = prepare_patch_mask(zarr_path, s3_array, patch_count_mask, "patch_mask")
    patch_mask._source_data[:] = patch_count_mask * gridmask

    # Generate list of coordinates of patches
    coords = np.nonzero(patch_mask._source_data[:])
    offsets = [
        patch_mask.roi.offset + Coordinate(a, b) * patch_mask.voxel_size
        for a, b in zip(*coords)
    ]

    # apply VGG16 Cortex model at 5x (s3)
    s3_array = open_ds(zarr_path / "raw" / "s3")
    pred_mask = prepare_mask(zarr_path, s3_array, "cortex")
    pred_mask[:] = 3
    pred_cortex(s3_array, offsets, patch_size, patch_spacing, pred_mask, device)
    # shave off background
    pred_mask._source_data[:] = pred_mask._source_data * filled_fmask._source_data[:]

    # create integral mask of cortex mask and save
    integral_cortex_mask = integral_image(pred_mask._source_data[:])
    icmask = prepare_mask(zarr_path, s3_array, "integral_cortex")
    icmask._source_data[:] = integral_cortex_mask

    # create patch mask for Omni-Seg
    patch_shape_final = Coordinate(512, 512)  # Will convention: shape in voxels

    patch_count_mask_final, patch_size_final = generate_patchcount_mask(
        icmask, patch_shape_final, tissueallowance=0.10
    )

    # generate gridmask
    patch_spacing_final = Coordinate(256, 256)
    gridmaskfinal = generate_grid_mask(patch_count_mask_final, patch_spacing_final)

    # create patch mask
    patch_mask_final = prepare_patch_mask(
        zarr_path, s3_array, patch_count_mask_final, "patch_mask_final"
    )
    patch_mask_final._source_data[:] = gridmaskfinal * patch_count_mask_final

    # Generate list of coordinates of patches
    coords = np.nonzero(patch_mask_final._source_data[:])
    offsets_final = [
        patch_mask_final.roi.offset + Coordinate(a, b) * patch_mask_final.voxel_size
        for a, b in zip(*coords)
    ]

    # create visualization of regions pulled
    roi_mask = prepare_patch_mask(zarr_path, s3_array, patch_count_mask, "rois")

    # get cluster centers
    clustering, n_clusters = prepare_clustering(gray_cluster_centers)

    # grab 40x level
    s0_array = open_ds(zarr_path / "raw" / "s0")

    print("Preparing Masks for Clustering")
    # prepare clustering mask locations in zarr (40x)
    fibrosis1_mask, fibrosis2_mask, inflammation_mask, structuralcollagen_mask = (
        prepare_clustering_masks(zarr_path, s0_array)
    )

    # prepare histological segmentation mask locations in zarr (10x)
    pt_mask_10x, dt_mask_10x, vessel_mask_10x, cap_mask_10x = prepare_seg_masks(zarr_path, s2_array, "10x")
    # prepare histological segmentation mask locations in zarr (40x)
    pt_mask, dt_mask, vessel_mask, cap_mask = prepare_seg_masks(zarr_path, s0_array, "40x")

    # prepare postprocessing masks in zarr (40x)
    tbm_mask, bc_mask, fincap_mask, finfib_mask, fincollagen_mask, fincollagen_exclusion_mask, fininflamm_mask = (
        prepare_postprocessing_masks(zarr_path, s0_array)
    )
    # prep mask for cap at 10x
    cap_mask10x = prepare_mask(zarr_path, s2_array, "cap10x")
    # create text file for fibrosis score data
    txt_path = input_path.with_suffix(".txt")
    with open(txt_path, "w") as f:
        f.writelines("ROI fibrosis scores \n")

    print("Clustering")
    for offset in tqdm(offsets_final):
        # world units roi selection
        roi = Roi(offset, patch_size_final)  # in nm
        # visualize roi in mask
        voxel_mask_roi = (roi - roi_mask.offset) / roi_mask.voxel_size
        roi_mask._source_data[
            voxel_mask_roi.begin[0] : voxel_mask_roi.end[0],
            voxel_mask_roi.begin[1] : voxel_mask_roi.end[1],
        ] = 1

        # save image at diff pyramid levels for Omni-Seg input
        # name roi
        patchname = Path(f"{zarr_path.stem}_{offset}_pyr0.png")
        patchpath = png_path / patchname
        # select roi
        array = open_ds(zarr_path / "raw" / f"s0")

        # redefine roi on s0 array
        patch_shape_final = Coordinate(4096, 4096)  # omniseg input was 4096/8192
        patch_size_final = patch_shape_final * array.voxel_size  # nm
        roi = Roi(offset, patch_size_final)  # nm
        voxel_roi = (roi - array.offset) / array.voxel_size  # voxels
        patch_raw = array[
            voxel_roi.begin[0] : voxel_roi.end[0], voxel_roi.begin[1] : voxel_roi.end[1]
        ]

        # cluster 40x pyramid level
        # save png of image at current pyramid level
        grayscale_cluster(
            patchpath,
            patch_raw,
            voxel_roi,
            clustering,
            n_clusters,
            fibrosis1_mask,
            fibrosis2_mask,
            inflammation_mask,
            structuralcollagen_mask,
        )

    # begin u-net preds: define preprocessing & patch size
    inp_transforms = T.Compose(
        [
            T.Grayscale(),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
        ]
    )
    patch_shape_final = Coordinate(1056, 1056)
    patch_size_final = patch_shape_final * s2_array.voxel_size  # size in nm

    print("Predicting Gloms with U-Net")
    #### Use U-net to predict gloms ####

    # load glom model
    # model = torch.load("model_dataset1_flips_eric.pt", weights_only=False) before
    print_gpu_usage(device)

    model = torch.load("model_unet_dataset11Aug2025_glom_LSDs_gaussianelastic_longer_350.pt", weights_only=False)
    binary_head = torch.load("binaryhead_unet_dataset11Aug2025_glom_LSDs_gaussianelastic_longer_350.pt", weights_only=False)
    # predict
    print_gpu_usage(device)

    model_prediction_lsds(cap_mask_10x, s2_array, patch_size_final, model, binary_head, device, "Cap ID")

    # remove small objects from cap mask
    cap_mask_10x._source_data[:] = remove_small_objects(
        cap_mask_10x._source_data[:].astype(bool), min_size=2000
    )

    cap_mask_10x._source_data[:] = remove_small_holes(
        cap_mask_10x._source_data[:].astype(bool), area_threshold=3000
    )

    print("Predicting proximal tubules with U-Net")
    #### Use U-net to predict pt ####
    patch_size_final = patch_shape_final * s2_array.voxel_size  # size in nm

    # remove previous model from gpu
    #del model 
    torch.cuda.empty_cache()

    # load model
    model = torch.load("model_unet_dataset0_dtpt0_200.pt", weights_only=False)
    # predict
    model_prediction(pt_mask_10x, s2_array, patch_size_final, model, device, "PT ID")

    print("Predicting distal tubules with U-Net")

    #### Use U-net to predict dt ####

    # remove previous model from gpu
    del model 
    # load model
    model = torch.load("model_unet_dataset0_dtpt0_200.pt", weights_only=False)
    # predict
    model_prediction(dt_mask_10x, s2_array, patch_size_final, model, device, "DT ID")
    
    # assign pixels that were postive in both the pt and dt masks to pt for simplicity 
    dt_mask_10x.data = dask.array.where((pt_mask_10x.data == 1) & (dt_mask_10x.data == 1), 0, dt_mask_10x.data)

    #### Use U-net to predict vessel ####

    # remove previous model from gpu
    del model 
    # load model
    model = torch.load("model_unet_dataset3_vessel0_final.pt", weights_only=False)
    # predict
    model_prediction(vessel_mask_10x, s2_array, patch_size_final, model, device, "Vessel ID")
    # remove small objects from vessel mask
    cap_mask_10x._source_data[:] = remove_small_objects(
        cap_mask_10x._source_data[:].astype(bool), min_size=2000
    )

    upsampling_factor = cap_mask_10x.voxel_size / fincap_mask.voxel_size

    # blockwise upsample from 10x to 40x
    upsample(cap_mask_10x, fincap_mask, upsampling_factor, s2_array, s0_array)
    upsample(pt_mask_10x, pt_mask, upsampling_factor, s2_array, s0_array)
    upsample(dt_mask_10x, dt_mask, upsampling_factor, s2_array, s0_array)
    upsample(vessel_mask_10x, vessel_mask, upsampling_factor, s2_array, s0_array)

    # scale eroded foreground mask to 40x from 5x for final multiplications
    fg_eroded = open_ds(zarr_path / "mask" / "foreground_eroded")  # in s3
    upsampling_factor = fg_eroded.voxel_size / vessel_mask.voxel_size
    fg_eroded_s0 = prepare_ds(
        zarr_path / "mask" / "foreground_eroded_s0",
        shape=Coordinate(fg_eroded.shape) * upsampling_factor,
        offset=fg_eroded.offset,
        voxel_size=vessel_mask.voxel_size,
        axis_names=fg_eroded.axis_names,
        units=fg_eroded.units,
        mode="w",
        dtype=fg_eroded.dtype,
    )

    # upsample tissue mask from 5x to 40x
    tissuemask_upsample(fg_eroded, fg_eroded_s0, s0_array, upsampling_factor)

    print("Identifying TBM")
    # need to erode and dilate to generate a TBM class
    id_tbm(dt_mask, pt_mask, structuralcollagen_mask, fibrosis1_mask, fibrosis2_mask, tbm_mask, s0_array, fg_eroded_s0)

    print("Identifying Bowman's Capsules")
    # need to erode and dilate to generate Bowman's Capsule class
    id_bc(
        fincap_mask,
        structuralcollagen_mask,
        fibrosis1_mask,
        fibrosis2_mask,
        bc_mask,
        fg_eroded_s0,
        s0_array,
    )

    print("Identifying Structural Collagen around Vessels")
    # create masks for vessel
    vessel10xdilated = prepare_mask(zarr_path, s2_array, "vessel10xdilated")
    vessel40xdilated = prepare_mask(zarr_path, s0_array, "vessel40xdilated")

    # size of dilation of smallest object
    print("Dilating Vessels")
    varied_vessel_dilation(3, vessel_mask_10x, vessel10xdilated)
    print("Saving Dilation")
    upsampling_factor = vessel10xdilated.voxel_size / vessel40xdilated.voxel_size
    print("Upsampling Dilation")
    upsample(vessel10xdilated, vessel40xdilated, upsampling_factor, s2_array, s0_array)
    #vessel_40x_dilated = open_ds(zarr_path / "mask" / "vessel40xdilated", mode="a")
    print("Saving Upsample")
    vessel40xdilated.data = (
        vessel40xdilated.data
        * (1 - inflammation_mask.data)
        * (1 - dt_mask.data)
        * (1 - pt_mask.data)
        * (1 - fincap_mask.data)
        * (1 - tbm_mask.data)
        * (1 - bc_mask.data)
        * (fibrosis1_mask.data + fibrosis2_mask.data + structuralcollagen_mask.data)
        * (fg_eroded_s0.data)
    )
    vessel_store = dask.array.store(vessel40xdilated.data, vessel40xdilated._source_data, execute=False)
    dask.compute(vessel_store)
    print("Completed Saving Upsampling")

    ################################################
    print("Calculating Fibrosis Mask")
    # create final fibrosis overlay: fibrosis1 + fibrosis2 from clustering - tuft, capsule, vessel, tubules
    finfib_mask.data = (
        (fibrosis1_mask.data + fibrosis2_mask.data + structuralcollagen_mask.data)
        * (1 - vessel_mask.data)
        * (1 - vessel40xdilated.data)
        * (1 - dt_mask.data)
        * (1 - pt_mask.data)
        * (1 - fincap_mask.data)
        * (1 - tbm_mask.data)
        * (1 - bc_mask.data)
        * fg_eroded_s0.data
    )
    print("Saving Fibrosis Mask")
    # execute multiplication
    finfib_store = dask.array.store(finfib_mask.data, finfib_mask._source_data, compute=False)
    # save blockwise
    dask.compute(finfib_store)

    print("Removing Small Fibrosis Objects")
    remove_small_fib(finfib_mask, s0_array)

    print("Completed Saving Fibrosis Mask")
    print("Calculating Collagen Overlay")
    # create final collagen overlay
    fincollagen_mask.data = (
        (structuralcollagen_mask.data + vessel40xdilated.data + tbm_mask.data + bc_mask.data)
        * (1 - dt_mask.data)
        * (1 - pt_mask.data)
        * (1 - fincap_mask.data)
        * (1 - vessel_mask.data)
        * fg_eroded_s0.data
    )
    fincollagen_mask.data = fincollagen_mask.data >> 0
    print("Saving Fibrosis Overlay")
    # execute multiplication
    fincollagen_store = dask.array.store(fincollagen_mask.data, fincollagen_mask._source_data, compute=False)
    # store blockwise
    dask.compute(fincollagen_store)

    print("Compeleted Saving Fibrosis Overlay")
    print("Calculating Collagen Mask with Exclusion")
    # create final collagen exclusion overlay
    fincollagen_exclusion_mask.data = (
        (vessel40xdilated.data + tbm_mask.data + bc_mask.data)
        * (1 - dt_mask.data)
        * (1 - pt_mask.data)
        * (1 - fincap_mask.data)
        * (1 - vessel_mask.data)
        * fg_eroded_s0.data
    )
    
    print("Saving Collagen Mask with Exclusion")
    # execulte multiplication
    fincollagen_exclusion_store = dask.array.store(fincollagen_exclusion_mask.data, fincollagen_exclusion_mask._source_data, compute=False)
    # store blockwise
    dask.compute(fincollagen_exclusion_store)
    print("Completed Saving Collagen Mask with Exclusion")

    print("Calculating Inflammation Mask")
    # create final inflammation overlay
    fininflamm_mask.data = (
        inflammation_mask.data
        * (1 - vessel_mask.data)
        * (1 - dt_mask.data)
        * (1 - pt_mask.data)
        * (1 - fincap_mask.data)
        * (1 - vessel_mask.data)
        * fg_eroded_s0.data
    )
    print("Saving Inflammation Mask")
    # execute multiplication
    inflamm_store = dask.array.store(fininflamm_mask.data, fininflamm_mask._source_data, compute=False)
    # store blockwise
    dask.compute(inflamm_store)
    print("Completed Saving Inflammation Mask")

    # apply edge tissue mask to remaining masks
    print("Applying Foreground Mask to Masks")
    fincap_mask.data = fincap_mask.data * fg_eroded_s0.data
    fincap_store = dask.array.store(fincap_mask.data, fincap_mask._source_data, execute=False)
    dask.compute(fincap_store)
    pt_mask.data = pt_mask.data * fg_eroded_s0.data
    pt_store = dask.array.store(pt_mask.data, pt_mask._source_data, execute=False)
    dask.compute(pt_store)
    dt_mask.data = dt_mask.data * fg_eroded_s0.data
    dt_store = dask.array.store(dt_mask.data, dt_mask._source_data, execute=False)
    dask.compute(dt_store)
    vessel_mask.data = vessel_mask.data * fg_eroded_s0.data
    fin_vessel_store = dask.array.store(vessel_mask.data, vessel_mask._source_data, execute=False)
    dask.compute(fin_vessel_store)

    # calculate ROI fibrosis score & save to txt file
    with open(Path(zarr_path.parent / "fibpx.txt"), "w") as f:
        f.writelines("# Fibrosis Pixels per Block \n")
    with open(Path(zarr_path.parent / "tissuepx.txt"), "w") as f:
        f.writelines("# Tissue Pixels per Block \n")
    with open(Path(zarr_path.parent / "inflammpx.txt"), "w") as f:
        f.writelines("# Inflammation Pixels per Block \n")
    with open(Path(zarr_path.parent / "interstitiumpx.txt"), "w") as f:
        f.writelines("# Interstitium Pixels per Block \n")

    # blockwise mask multiplications
    print("Calculating Fibrosis Score")
    calculate_fibscore(finfib_mask, fg_eroded_s0, vessel_mask, fincap_mask, zarr_path, s0_array)
    fibpx = np.loadtxt(Path(zarr_path.parent / "fibpx.txt"), comments="#", dtype=int)
    tissuepx = np.loadtxt(Path(zarr_path.parent / "tissuepx.txt", comments="#", dtype=int))

    print("Calculating Inflammation Score")
    calculate_inflammscore(fininflamm_mask, fincollagen_exclusion_mask, finfib_mask, zarr_path, s0_array)
    inflammpx = np.loadtxt(Path(zarr_path.parent / "inflammpx.txt"), comments="#", dtype=int)
    interstitiumpx = np.loadtxt(Path(zarr_path.parent / "interstitiumpx.txt", comments="#", dtype=int))

    total_fibpx = np.sum(fibpx)
    total_inflammpx = np.sum(inflammpx)
    total_tissuepx = np.sum(tissuepx)
    total_interstitiumpx = np.sum(interstitiumpx)
    total_fibscore = (total_fibpx / total_tissuepx) * 100
    total_inflammscore = (total_inflammpx / total_interstitiumpx) * 100 

    print("Fibrosis Score Calculated")
    print("Inflammation Score Calculated")

    with open(txt_path, "a") as f:
        f.writelines(f"Final Fibscore: {total_fibscore} \nFinal Inflammscore: {total_inflammscore}")
        
    return
