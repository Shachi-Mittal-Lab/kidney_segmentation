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
    downsample_fibrosis,
    clean_visualization,
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
from skimage.measure import label
from scipy.ndimage import gaussian_filter

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

    # identify input file
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
    s0_array = open_ds(zarr_path / "raw" / "s0")

    print("Generating Foreground Masks")
    fmask, filled_fmask, eroded_fmask, abnormaltissue_mask = prepare_foreground_masks(zarr_path, s3_array)
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

    # create eroded foreground mask in 40x to apply to segmentations
    # and use as denominator in fibrosis calculations
    eroded_cortex_mask_5x = prepare_mask(zarr_path, s3_array, "eroded_cortex_5x")
    eroded_cortex_mask_40x = prepare_mask(zarr_path, s0_array, "eroded_cortex_40x")
    cortex_erosion_multiplication = eroded_fmask.data * pred_mask.data
    eroded_cortex_5x = dask.array.store(cortex_erosion_multiplication, eroded_cortex_mask_5x._source_data, execute=False)
    dask.compute(eroded_cortex_5x)
    upsampling_factor = eroded_cortex_mask_5x.voxel_size / eroded_cortex_mask_40x.voxel_size
    upsample(eroded_cortex_mask_5x, eroded_cortex_mask_40x, upsampling_factor, s3_array, s0_array)

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

    print("Preparing Masks for Clustering")
    # prepare clustering mask locations in zarr (40x)
    fibrosis1_mask, fibrosis2_mask, inflammation_mask, structuralcollagen_mask = (
        prepare_clustering_masks(zarr_path, s0_array)
    )
    print("Preparing Masks for Segmentation")
    # prepare histological segmentation mask locations in zarr (10x)
    tubule_mask_10x, vessel_mask_10x, cap_mask_10x = prepare_seg_masks(zarr_path, s2_array, "10x")
    # prepare histological segmentation mask locations in zarr (40x)
    tubule_mask, vessel_mask, cap_mask = prepare_seg_masks(zarr_path, s0_array, "40x")

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
        f.close()

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
    print_gpu_usage(device)
    model = torch.load("model_unet_dataset5_01Jan2026_glom0_LSDs_final.pt", weights_only=False)
    binary_head = torch.load("binaryhead_unet_dataset5_01Jan2026_glom0_LSDs_final.pt", weights_only=False)
    print_gpu_usage(device)

    # predict
    model_prediction_lsds(cap_mask_10x, s2_array, patch_size_final, model, binary_head, device, "Cap ID")

    # remove small objects from cap mask
    cap_mask_10x._source_data[:] = remove_small_objects(
        cap_mask_10x._source_data[:].astype(bool), min_size=2000
    )

    cap_mask_10x._source_data[:] = remove_small_holes(
        cap_mask_10x._source_data[:].astype(bool), area_threshold=5000
    )

    print("Predicting Tubules with U-Net")
    #### Use U-net to predict tubules ####
    patch_size_final = patch_shape_final * s2_array.voxel_size  # size in nm
    # remove previous model from gpu
    del model 
    torch.cuda.empty_cache()
    # load mode
    print_gpu_usage(device)
    model = torch.load("model_unet_dataset3_tubule0_LSDs_400.pt", weights_only=False)
    binary_head = torch.load("binaryhead_unet_dataset3_tubule0_LSDs_400.pt", weights_only=False)
    # predict
    print_gpu_usage(device)
    model_prediction_lsds(tubule_mask_10x, s2_array, patch_size_final, model, binary_head, device, "Tubule ID")

    print("Predicting Vessels with U-Net")
    #### Use U-net to predict vessel ####
    # remove previous model from gpu
    del model
    torch.cuda.empty_cache()
    # load model
    print_gpu_usage(device)
    model = torch.load("model_unet_dataset12_vessel0_LSDs_final.pt", weights_only=False)
    binary_head = torch.load("binaryhead_unet_dataset12_vessel0_LSDs_final.pt", weights_only=False)
    print_gpu_usage(device)

    # predict
    model_prediction_lsds(vessel_mask_10x, s2_array, patch_size_final, model, binary_head, device, "Vessel ID")
    # remove small objects from vessel mask
    vessel_mask_10x._source_data[:] = remove_small_objects(
        vessel_mask_10x._source_data[:].astype(bool), min_size=2000
    )
    # remove previous model from gpu
    del model 
    torch.cuda.empty_cache()

    # blockwise upsample from 10x to 40x
    upsampling_factor = vessel_mask_10x.voxel_size / fincap_mask.voxel_size
    upsample(cap_mask_10x, fincap_mask, upsampling_factor, s2_array, s0_array)
    upsample(tubule_mask_10x, tubule_mask, upsampling_factor, s2_array, s0_array)
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

    # Remove overlaps in masks 
    clean_visualization(vessel_mask, fincap_mask, tubule_mask, s0_array)

    print("Identifying TBM")
    # need to erode and dilate to generate a TBM class
    id_tbm(tubule_mask, structuralcollagen_mask, fibrosis1_mask, fibrosis2_mask, tbm_mask, s0_array, fg_eroded_s0)

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
    vessel40xdilated_multiplication = (
        vessel40xdilated.data
        * (1 - inflammation_mask.data)
        * (1 - tubule_mask.data)
        * (1 - fincap_mask.data)
        * (1 - tbm_mask.data)
        * (1 - bc_mask.data)
        * (fibrosis1_mask.data + fibrosis2_mask.data + structuralcollagen_mask.data)
        * (eroded_cortex_mask_40x.data)
    )
    vessel_store = dask.array.store(vessel40xdilated_multiplication, vessel40xdilated._source_data, execute=False)
    dask.compute(vessel_store)
    print("Completed Saving Upsampling")

    ################################################
    print("Calculating Fibrosis Mask")
    # create final fibrosis overlay: fibrosis1 + fibrosis2 from clustering - tuft, capsule, vessel, tubules
    finfib_multiplication = (
        (fibrosis1_mask.data + fibrosis2_mask.data + structuralcollagen_mask.data)
        * (1 - vessel_mask.data)
        * (1 - vessel40xdilated.data)
        * (1 - tubule_mask.data)
        * (1 - fincap_mask.data)
        * (1 - tbm_mask.data)
        * (1 - bc_mask.data)
        * eroded_cortex_mask_40x.data
    )
    print("Saving Fibrosis Mask")
    # execute multiplication
    finfib_store = dask.array.store(finfib_multiplication, finfib_mask._source_data, compute=False)
    # save blockwise
    dask.compute(finfib_store)

    print("Removing Small Fibrosis Objects")
    smallfib_mask = prepare_mask(zarr_path, s0_array, "smallfib")
    remove_small_fib(finfib_mask, s0_array, smallfib_mask)

    #confused_roi = Roi((0, 10000000), (1000000, 1000000))
    #count = finfib_mask[confused_roi].sum()
    #assert count == 63238, count

    print("Completed Saving Fibrosis Mask")

    print("Generatinging Abnormal Tissue Mask")  # ABNORMAL VS NORMAL TISSUE WILL BE CALCULATED IN THE 5X
    # downsample fibrosis mask for abnormal tissue visualization
    downsample_fibrosis(finfib_mask, abnormaltissue_mask)
    # remove small objects from downsampled fibrosis mask
    labeled = label(abnormaltissue_mask.data.astype(bool), connectivity=2)
    filtered_fib = remove_small_objects(labeled, min_size=2000)
    # apply Gaussian blur to visualize abnormal tissue mask
    abnormaltissue = gaussian_filter(filtered_fib, sigma = 41.0, truncate = 4.0) > 0.01
    abnormaltissue = abnormaltissue.astype(bool)
    dask_abnormaltissue = dask.array.from_array(abnormaltissue)
    # apply 5x foreground mask to abnormal tissue mask
    dask_abnormaltissue = dask_abnormaltissue * eroded_cortex_mask_5x.data
    dask.array.store(dask_abnormaltissue, abnormaltissue_mask._source_data, compute=True)
   
    print("Calculating Structural Collagen Overlay")
    # create final collagen overlay
    fincollagen_mask_multiplication = (
        (structuralcollagen_mask.data + vessel40xdilated.data + tbm_mask.data + bc_mask.data)
        * (1 - tubule_mask.data)
        * (1 - fincap_mask.data)
        * (1 - vessel_mask.data)
        * eroded_cortex_mask_40x.data
    )
    fincollagen_mask_multiplication = fincollagen_mask_multiplication >> 0
    print("Saving Structural Collagen Overlay")
    # execute multiplication
    fincollagen_store = dask.array.store(fincollagen_mask_multiplication, fincollagen_mask._source_data, compute=False)
    # store blockwise
    dask.compute(fincollagen_store)

    print("Compeleted Saving Structural Collagen Overlay")
    print("Calculating Collagen Mask with Exclusion")
    # create final collagen exclusion overlay
    fincollagen_exclusion_mask_multiplication = (
        (vessel40xdilated.data + tbm_mask.data + bc_mask.data)
        * (1 - tubule_mask.data)
        * (1 - fincap_mask.data)
        * (1 - vessel_mask.data)
        * eroded_cortex_mask_40x.data
    )
    
    print("Saving Collagen Mask with Exclusion")
    # execulte multiplication
    fincollagen_exclusion_store = dask.array.store(fincollagen_exclusion_mask_multiplication, fincollagen_exclusion_mask._source_data, compute=False)
    # store blockwise
    dask.compute(fincollagen_exclusion_store)
    print("Completed Saving Collagen Mask with Exclusion")

    print("Calculating Inflammation Mask")
    # create final inflammation overlay
    fininflamm_mask_multiplication = (
        inflammation_mask.data
        * (1 - vessel_mask.data)
        * (1 - tubule_mask.data)
        * (1 - fincap_mask.data)
        * (1 - vessel_mask.data)
        * eroded_cortex_mask_40x.data
    )
    print("Saving Inflammation Mask")
    # execute multiplication
    inflamm_store = dask.array.store(fininflamm_mask_multiplication, fininflamm_mask._source_data, compute=False)
    # store blockwise
    dask.compute(inflamm_store)
    print("Completed Saving Inflammation Mask")

    # apply edge tissue & cortex mask to remaining masks
    print("Applying Foreground Cortex Mask to Masks")
    fincap_mask_multiplication = fincap_mask.data * eroded_cortex_mask_40x.data
    fincap_store = dask.array.store(fincap_mask_multiplication, fincap_mask._source_data, execute=False)
    dask.compute(fincap_store)
    tubule_mask_multiplication = tubule_mask.data * eroded_cortex_mask_40x.data
    tubule_store = dask.array.store(tubule_mask_multiplication, tubule_mask._source_data, execute=False)
    dask.compute(tubule_store)
    vessel_mask_multiplication = vessel_mask.data * eroded_cortex_mask_40x.data
    fin_vessel_store = dask.array.store(vessel_mask_multiplication, vessel_mask._source_data, execute=False)
    dask.compute(fin_vessel_store)

    # calculate ROI fibrosis score & save to txt file
    with open(Path(zarr_path.parent / f"{input_filename}_fibpx.txt"), "w") as f:
        f.writelines("# Fibrosis Pixels per Block \n")
        f.close()
    with open(Path(zarr_path.parent / f"{input_filename}_tissuepx.txt"), "w") as f:
        f.writelines("# Tissue Pixels per Block \n")
    with open(Path(zarr_path.parent / f"{input_filename}_tissuenoglomvesselpx.txt"), "w") as f:
        f.writelines("# Tissue without Capsules or Vessels Pixels per Block \n")
    with open(Path(zarr_path.parent / f"{input_filename}_inflammpx.txt"), "w") as f:
        f.writelines("# Inflammation Pixels per Block \n")
        f.close()
    with open(Path(zarr_path.parent / f"{input_filename}_interstitiumpx.txt"), "w") as f:
        f.writelines("# Interstitium Pixels per Block \n")
        f.close()

    # Calculate fibrosis score at 40x
    print("Calculating Fibrosis Score")
    calculate_fibscore(finfib_mask, eroded_cortex_mask_40x, vessel_mask, fincap_mask, zarr_path, input_filename, s0_array)
    fibpx = np.loadtxt(Path(zarr_path.parent / f"{input_filename}_fibpx.txt"), comments="#", dtype=int)
    tissuenoglomvesselpx = np.loadtxt(Path(zarr_path.parent / f"{input_filename}_tissuenoglomvesselpx.txt", comments="#", dtype=int))
    tissuepx = np.loadtxt(Path(zarr_path.parent / f"{input_filename}_tissuepx.txt", comments="#", dtype=int))

    print("Calculating Inflammation Score")
    calculate_inflammscore(fininflamm_mask, fincollagen_exclusion_mask, finfib_mask, zarr_path, input_filename, s0_array)
    inflammpx = np.loadtxt(Path(zarr_path.parent / f"{input_filename}_inflammpx.txt"), comments="#", dtype=int)
    interstitiumpx = np.loadtxt(Path(zarr_path.parent / f"{input_filename}_interstitiumpx.txt", comments="#", dtype=int))
    
    print("Finalizing Fibrosis Scores")
    total_fibpx = np.sum(fibpx)
    total_inflammpx = np.sum(inflammpx)
    total_tissuepx = np.sum(tissuepx)
    total_tissuenoglomvesselpx = np.sum(tissuenoglomvesselpx)
    total_interstitiumpx = np.sum(interstitiumpx)
    total_drsetty_fibscore = (total_fibpx / total_tissuenoglomvesselpx) * 100
    total_drsetty_fibscore_winflamm = ((total_fibpx + total_inflammpx) / total_tissuenoglomvesselpx) * 100
    total_alpers1_fibscore = (total_fibpx / total_tissuepx) * 100
    total_alpers1_fibscore_winflamm = (total_fibpx / total_tissuepx) * 100
    total_inflammscore = (total_inflammpx / total_interstitiumpx) * 100 

    # calculate fibrosis score with abnormal tissue in 5x
    abnormaltissuepx = dask.array.count_nonzero(abnormaltissue_mask.data)
    tissuepx_5x = dask.array.count_nonzero(eroded_cortex_mask_5x.data)
    total_alpers2_fibscore = abnormaltissuepx.compute() / tissuepx_5x.compute() * 100


    print("Fibrosis Score Calculated")
    print("Inflammation Score Calculated")

    with open(txt_path, "a") as f:
        f.write(f"Final Dr. Setty Fibscore: {total_drsetty_fibscore}\n")
        f.write(f"Final Dr. Setty Fibscore with Inflammation Included: {total_drsetty_fibscore}\n")
        f.write(f"Final Alpers 1 Score: {total_alpers1_fibscore}\n")
        f.write(f"Final Alpers 1 Score with Inflammation Included: {total_alpers1_fibscore_winflamm}\n")
        f.write(f"Final Alpers 2 Score: {total_alpers2_fibscore.compute()}\n")
        f.write(f"Final Inflammscore: {total_inflammscore}\n")
        f.close()
    return
