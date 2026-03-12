# Repo Tools
from fibrosis_score.zarr_utils import ndpi_to_zarr, svs_to_zarr
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
        threshold: int,
        gray_cluster_centers: Path,
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

    # convert file to zarr if not a zarr
    if input_file_ext == ".ndpi":
        # name zarr/home/amninder/Desktop/project/Folder_2/subfolder
        zarr_path = input_path.with_suffix(".zarr")
        # convert ndpi to zarr array levels 40x, 20x, 10x, 5x, and rechunk
        ndpi_to_zarr(input_path, zarr_path, offset, axis_names)

    elif input_file_ext == ".svs": 
        # name zarr/home/amninder/Desktop/project/Folder_2/subfolder
        zarr_path = input_path.with_suffix(".zarr")
        # convert ndpi to zarr array levels 40x, 20x, 10x, 5x, and rechunk
        svs_to_zarr(input_path, zarr_path, offset, axis_names)

    elif input_file_ext == ".zarr":
        zarr_path = input_path

    else:
        print(f"File must be of format .ndpi, .svs, or .zarr.  File extension is {input_file_ext}. Skipping {input_filename}")
        return

    # grab 5x, 10x levels & calculate foreground masks
    s3_array = open_ds(zarr_path / "raw" / "s3") # 5x
    s2_array = open_ds(zarr_path / "raw" / "s2") # 10x
    s1_array  = open_ds(zarr_path / "raw" / "s1") # 20x

    print("Generating Cortex Mask")
    abnormaltissue_mask = prepare_mask(zarr_path, s3_array, "abnormaltissue")
    cortex_mask = prepare_mask(zarr_path, s3_array, "desired_cortex") 
    cortex_mask_20x = prepare_mask(zarr_path, s1_array, "desired_cortex_20x")

    # Create background mask for clustering
    patch_shape_final = Coordinate(1056, 1056)
    patch_size_final = patch_shape_final * s1_array.voxel_size  # size in nm
    bg_mask = prepare_mask(zarr_path, s1_array, "background")
    # cluster out background
    clustering, n_clusters = prepare_clustering(gray_cluster_centers)
    background_cluster(bg_mask, clustering, n_clusters, patch_size_final, s1_array, "Identifying Background")

    print("Preparing Masks for Segmentation")
    # prepare histological segmentation mask locations in zarr (10x)
    tubule_mask_10x, vessel_mask_10x, cap_mask_10x = prepare_seg_masks(zarr_path, s2_array, "10x")
    # prepare histological segmentation mask locations in zarr (20x)
    tubule_mask, vessel_mask, nuclei_mask = prepare_seg_masks(zarr_path, s1_array, "20x")

    # prepare postprocessing masks in zarr (20x)
    tbm_mask, bc_mask, fincap_mask, finfib_mask, fincollagen_mask, fincollagen_exclusion_mask, fininflamm_mask = (
        prepare_postprocessing_masks(zarr_path, s1_array)
    )

    # create text file for fibrosis score data
    txt_path = input_path.with_suffix(".txt")
    with open(txt_path, "w") as f:
        f.writelines("ROI fibrosis scores \n")
        f.close()

    # begin u-net preds: define preprocessing & patch size
    inp_transforms = T.Compose(
        [
            T.Grayscale(),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
        ]
    )

    # size to feed to 5x u-net in pixels
    patch_shape_final = Coordinate(1056, 1056)
    # size to feed to 5x u-net in nm
    patch_size_final = patch_shape_final * s3_array.voxel_size  # size in nm
    # calculate size affected by padding of the 1056 x 1056 block
    padding_affected_shape = Coordinate(208,208)
    padding_affected_size = padding_affected_shape * s3_array.voxel_size  # size in nm

    print("Predicting Cortex")
    # load model
    print_gpu_usage(device)
    model = torch.load("model_unet_cortexdataset1_5x_LSDs_final.pt", weights_only=False)
    binary_head = torch.load("binaryhead_unet_cortexdataset1_5x_LSDs_final.pt", weights_only=False)
    print_gpu_usage(device)
    model_prediction_lsds(cortex_mask, s3_array, patch_size_final, padding_affected_size, model, binary_head, device, "ID Cortex")


    # size to feed to 10x u-nets in pixels
    patch_shape_final = Coordinate(1056, 1056)
    # size to feed to 10x u-nets in nm
    patch_size_final = patch_shape_final * s1_array.voxel_size  # size in nm
    # calculate size affected by padding of the 1056 x 1056 block
    padding_affected_shape = Coordinate(208,208)
    padding_affected_size = padding_affected_shape * s1_array.voxel_size  # size in nm

    print("Predicting Gloms with U-Net")
    #### Use U-net to predict gloms ####
    # remove previous model from gpu
    del model
    torch.cuda.empty_cache()

    # load glom model
    print_gpu_usage(device)
    model = torch.load("model_unet_dataset5_01Jan2026_glom0_LSDs_final.pt", weights_only=False)
    binary_head = torch.load("binaryhead_unet_dataset5_01Jan2026_glom0_LSDs_final.pt", weights_only=False)
    print_gpu_usage(device)

    # predict
    model_prediction_lsds(cap_mask_10x, s2_array, patch_size_final, padding_affected_size, model, binary_head, device, "Cap ID")

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
    model_prediction_lsds(tubule_mask_10x, s2_array, patch_size_final, padding_affected_size, model, binary_head, device, "Tubule ID")

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
    model_prediction_lsds(vessel_mask_10x, s2_array, patch_size_final, padding_affected_size, model, binary_head, device, "Vessel ID")
    # remove small objects from vessel mask
    vessel_mask_10x._source_data[:] = remove_small_objects(
        vessel_mask_10x._source_data[:].astype(bool), min_size=2000
    )
    # remove previous model from gpu
    del model 
    torch.cuda.empty_cache()

    # Prepare to predict at 20x #
    # size to feed to u-nets in pixels
    patch_shape_final = Coordinate(1056, 1056)
    # size to feed to u-nets in nm
    patch_size_final = patch_shape_final * s1_array.voxel_size  # size in nm
    # calculate size affected by padding of the 1056 x 1056 block
    padding_affected_shape = Coordinate(208,208)
    padding_affected_size = padding_affected_shape * s1_array.voxel_size  # size in nm

    print("Predicting Nuclei with U-Net")
    #### Use U-net to predict nuclei at 20x ####
    # load model
    model = torch.load("model_unet_inflammdataset1_inflamm0_20x_LSDs_final.pt", weights_only=False)
    binary_head = torch.load("binaryhead_unet_inflammdataset1_inflamm0_20x_LSDs_final.pt", weights_only=False)
    print_gpu_usage(device)

    # predict
    model_prediction_lsds(nuclei_mask, s1_array, patch_size_final, padding_affected_size, model, binary_head, device, "Nuclei ID")

    # remove previous model from gpu
    del model 
    torch.cuda.empty_cache()
    # blockwise upsample from 5x to 20x
    upsampling_factor = s3_array.voxel_size / s1_array.voxel_size
    upsample(cortex_mask, cortex_mask_20x, upsampling_factor, s3_array, s1_array)

    # blockwise upsample from 10x to 20x
    upsampling_factor = s2_array.voxel_size / s1_array.voxel_size
    upsample(cap_mask_10x, fincap_mask, upsampling_factor, s2_array, s1_array)
    upsample(tubule_mask_10x, tubule_mask, upsampling_factor, s2_array, s1_array)
    upsample(vessel_mask_10x, vessel_mask, upsampling_factor, s2_array, s1_array)

    # apply edge tissue & cortex mask to remaining masks
    print("Applying Foreground Cortex Mask to Masks")
    fincap_mask_multiplication = fincap_mask.data * cortex_mask_20x.data * (1 - bg_mask.data)
    fincap_store = dask.array.store(fincap_mask_multiplication, fincap_mask._source_data, execute=False)
    dask.compute(fincap_store)
    tubule_mask_multiplication = tubule_mask.data * cortex_mask_20x.data * (1 - bg_mask.data)
    tubule_store = dask.array.store(tubule_mask_multiplication, tubule_mask._source_data, execute=False)
    dask.compute(tubule_store)
    vessel_mask_multiplication = vessel_mask.data * cortex_mask_20x.data * (1 - bg_mask.data)
    fin_vessel_store = dask.array.store(vessel_mask_multiplication, vessel_mask._source_data, execute=False)
    dask.compute(fin_vessel_store)

    # scale eroded foreground mask to 20x from 5x for final multiplications
    fg_eroded = open_ds(zarr_path / "mask" / "foreground_eroded")  # in s3
    upsampling_factor = fg_eroded.voxel_size / s1_array.voxel_size
    fg_eroded_s1 = prepare_ds(
        zarr_path / "mask" / "foreground_eroded_s1",
        shape=Coordinate(fg_eroded.shape) * upsampling_factor,
        offset=fg_eroded.offset,
        voxel_size=vessel_mask.voxel_size,
        axis_names=fg_eroded.axis_names,
        units=fg_eroded.units,
        mode="w",
        dtype=fg_eroded.dtype,
    )

    # upsample tissue mask from 5x to 40x
    tissuemask_upsample(fg_eroded, fg_eroded_s1, upsampling_factor)

    # Remove overlaps in masks 
    clean_visualization(vessel_mask, fincap_mask, tubule_mask, s1_array)

    print("Identifying TBM")
    # need to erode and dilate to generate a TBM class
    id_tbm(tubule_mask, nuclei_mask, tbm_mask, s1_array)

    print("Identifying Bowman's Capsules")
    # need to erode and dilate to generate Bowman's Capsule class
    id_bc(
        fincap_mask,
        nuclei_mask,
        bc_mask,
        s1_array,
    )

    print("Identifying Structural Collagen around Vessels")
    # create masks for vessel
    vessel10xdilated = prepare_mask(zarr_path, s2_array, "vessel10xdilated")
    vessel20xdilated = prepare_mask(zarr_path, s1_array, "vessel20xdilated")

    # size of dilation of smallest object
    print("Dilating Vessels")
    varied_vessel_dilation(3, vessel_mask_10x, vessel10xdilated)
    print("Saving Dilation")
    upsampling_factor = vessel10xdilated.voxel_size / vessel20xdilated.voxel_size
    print("Upsampling Dilation")
    upsample(vessel10xdilated, vessel20xdilated, upsampling_factor, s2_array, s1_array)
    #vessel_40x_dilated = open_ds(zarr_path / "mask" / "vessel20xdilated", mode="a")
    print("Saving Upsample")
    vessel20xdilated_multiplication = (
        vessel20xdilated.data
        * (1 - bg_mask.data)
        * (1 - nuclei_mask.data)
        * (1 - tubule_mask.data)
        * (1 - fincap_mask.data)
        * (1 - tbm_mask.data)
        * (1 - bc_mask.data)
        * (cortex_mask_20x.data)
    )
    vessel_store = dask.array.store(vessel20xdilated_multiplication, vessel20xdilated._source_data, execute=False)
    dask.compute(vessel_store)
    print("Completed Saving Upsampling")

    ################################################
    print("Calculating Fibrosis Mask")
    # create final fibrosis overlay: fibrosis1 + fibrosis2 from clustering - tuft, capsule, vessel, tubules
    finfib_multiplication = (
        fg_eroded_s1.data
        * (1 - bg_mask.data)
        * (1 - nuclei_mask.data)
        * (1 - vessel_mask.data)
        * (1 - vessel20xdilated.data)
        * (1 - tubule_mask.data)
        * (1 - fincap_mask.data)
        * (1 - tbm_mask.data)
        * (1 - bc_mask.data)
        * cortex_mask_20x.data
    )
    print("Saving Fibrosis Mask")
    # execute multiplication
    finfib_store = dask.array.store(finfib_multiplication, finfib_mask._source_data, compute=False)
    # save blockwise
    dask.compute(finfib_store)

    print("Removing Small Fibrosis Objects")
    smallfib_mask = prepare_mask(zarr_path, s1_array, "smallfib")
    remove_small_fib(finfib_mask, s1_array, smallfib_mask)

    print("Completed Saving Fibrosis Mask")

    print("Generatinging Abnormal Tissue Mask")  # ABNORMAL VS NORMAL TISSUE WILL BE CALCULATED IN THE 5X
    # downsample fibrosis mask for abnormal tissue visualization
    downsample_fibrosis_20x(finfib_mask, abnormaltissue_mask)
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
   
    print("Calculating Structural Collagen Overlay with Exclusion")
    # create final collagen exclusion overlay
    fincollagen_exclusion_mask_multiplication = (
        (vessel20xdilated.data + tbm_mask.data + bc_mask.data)
        * (1 - bg_mask.data)
        * (1 - nuclei_mask.data)
        * (1 - tubule_mask.data)
        * (1 - fincap_mask.data)
        * (1 - vessel_mask.data)
        * cortex_mask_20x.data
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
        nuclei_mask.data
        * (1 - bg_mask.data)
        * (1 - vessel_mask.data)
        * (1 - tubule_mask.data)
        * (1 - fincap_mask.data)
        * (1 - vessel_mask.data)
        * cortex_mask_20x.data
    )
    print("Saving Inflammation Mask")
    # execute multiplication
    inflamm_store = dask.array.store(fininflamm_mask_multiplication, fininflamm_mask._source_data, compute=False)
    # store blockwise
    dask.compute(inflamm_store)
    print("Completed Saving Inflammation Mask")

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
    calculate_fibscore(finfib_mask, cortex_mask_20x, vessel_mask, fincap_mask, zarr_path, input_filename, s1_array)
    fibpx = np.loadtxt(Path(zarr_path.parent / f"{input_filename}_fibpx.txt"), comments="#", dtype=int)
    tissuenoglomvesselpx = np.loadtxt(Path(zarr_path.parent / f"{input_filename}_tissuenoglomvesselpx.txt", comments="#", dtype=int))
    tissuepx = np.loadtxt(Path(zarr_path.parent / f"{input_filename}_tissuepx.txt", comments="#", dtype=int))

    print("Calculating Inflammation Score")
    calculate_inflammscore(fininflamm_mask, fincollagen_exclusion_mask, finfib_mask, zarr_path, input_filename, s1_array)
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
    total_alpers1_fibscore_winflamm = ((total_fibpx + total_inflammpx) / total_tissuepx) * 100
    total_inflammscore = (total_inflammpx / total_interstitiumpx) * 100 

    # calculate fibrosis score with abnormal tissue in 5x
    abnormaltissuepx = dask.array.count_nonzero(abnormaltissue_mask.data)
    tissuepx_5x = dask.array.count_nonzero(eroded_cortex_mask_5x.data)
    total_alpers2_fibscore = abnormaltissuepx.compute() / tissuepx_5x.compute() * 100

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Convert to h:m:s
    hours   = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60


    print("Fibrosis Score Calculated")
    print("Inflammation Score Calculated")

    with open(txt_path, "a") as f:
        f.write(f"Final Dr. Setty Fibscore: {total_drsetty_fibscore}\n")
        f.write(f"Final Dr. Setty Fibscore with Inflammation Included: {total_drsetty_fibscore_winflamm}\n")
        f.write(f"Final Alpers 1 Score: {total_alpers1_fibscore}\n")
        f.write(f"Final Alpers 1 Score with Inflammation Included: {total_alpers1_fibscore_winflamm}\n")
        f.write(f"Final Alpers 2 Score: {total_alpers2_fibscore}\n")
        f.write(f"Final Inflammscore: {total_inflammscore}\n")
        f.write(f"Pipeline time: {hours}h {minutes}m {seconds:.2f}s")
        f.close()
    return
