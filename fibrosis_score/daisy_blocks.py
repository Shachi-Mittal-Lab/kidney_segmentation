# Repo Tools
from pred_utils import inp_transforms
from model import UNet, ConvBlock, Downsample, CropAndConcat, OutputConv


# Funke Lab Tools
import daisy
from funlib.persistence import Array
from funlib.geometry import Coordinate, Roi

# Tools
from PIL import Image
import torch
import numpy as np
from skimage.morphology import (
    binary_erosion,
    binary_dilation,
    disk,
)
from skimage.filters import gaussian
from pathlib import Path

def cap_prediction(
    cap_mask10x: Array,
    s2_array: Array,
    patch_size_final: Array,
    model: torch.nn.Module,
    device: torch.device,
):

    def process_block(block: daisy.Block):
        # in data slice
        inslices = s2_array._Array__slices(block.read_roi)
        # it was [:, 0:512, 0:512]
        # we want [0:512, 0:512, :]
        inslices = (inslices[1], inslices[2], inslices[0])
        img = Image.fromarray(s2_array[inslices])
        input = inp_transforms(img).unsqueeze(1).to(device)
        with torch.no_grad():
            preds = model(input)
            preds = preds.squeeze().squeeze().cpu().detach().numpy()
            preds = preds > 0.5
        cap_mask10x[block.write_roi] = preds

    cap_id_task = daisy.Task(
        "Cap ID",
        total_roi=s2_array.roi,
        read_roi=Roi((0, 0), patch_size_final),  # (offset, shape)
        write_roi=Roi((0, 0), patch_size_final),
        read_write_conflict=False,
        num_workers=2,
        process_function=process_block,
    )
    daisy.run_blockwise(tasks=[cap_id_task], multiprocessing=False)
    return


def cap_upsample(
    cap_mask10x: Array,
    fincap_mask: Array,
    upsampling_factor: Coordinate,
    s2_array: Array,
    s0_array: Array,
):

    def cap_upsample_block(block: daisy.Block):
        s2_data = cap_mask10x[block.read_roi]
        s0_data = s2_data
        for axis, reps in enumerate(upsampling_factor):
            s0_data = np.repeat(s0_data, reps, axis=axis)
        fincap_mask[block.write_roi] = s0_data

    block_roi = Roi((0, 0), (1000, 1000)) * s0_array.voxel_size

    cap_upsample_task = daisy.Task(
        "blockwise_upsample",
        total_roi=s2_array.roi,
        read_roi=block_roi,
        write_roi=block_roi,
        read_write_conflict=False,
        num_workers=2,
        process_function=cap_upsample_block,
    )
    daisy.run_blockwise(tasks=[cap_upsample_task], multiprocessing=False)
    return


def tissuemask_upsample(
    fg_eroded: Array,
    fg_eroded_s0: Array,
    s0_array: Array,
    upsampling_factor: Coordinate,
):
    # upsample 5x eroded tissue mask to use to filter predictions
    def upsample_block(block: daisy.Block):
        s3_data = fg_eroded[block.read_roi]
        s0_data = s3_data
        for axis, reps in enumerate(upsampling_factor):
            s0_data = np.repeat(s0_data, reps, axis=axis)
        fg_eroded_s0[block.write_roi] = s0_data

    block_roi = Roi((0, 0), (1000, 1000)) * fg_eroded_s0.voxel_size
    upsample_task = daisy.Task(
        "blockwise_upsample",
        total_roi=s0_array.roi,
        read_roi=block_roi,
        write_roi=block_roi,
        read_write_conflict=False,
        num_workers=2,
        process_function=upsample_block,
    )
    daisy.run_blockwise(tasks=[upsample_task], multiprocessing=False)
    return


def id_tbm(
    dt_mask: Array,
    pt_mask: Array,
    structuralcollagen_mask: Array,
    tbm_mask: Array,
    s0_array: Array,
    fg_eroded_s0: Array,
):
    # need to erode and dilate to generate a TBM class
    erode_kernel = disk(2, decomposition="sequence")
    dilate_kernel = disk(
        15, decomposition="sequence"
    )  # sequence for computational efficiency

    def tbm_block(block: daisy.Block):
        pt_data = dt_mask[block.read_roi]
        dt_data = pt_mask[block.read_roi]
        cluster = structuralcollagen_mask[block.read_roi]
        tubules = pt_data + dt_data
        eroded_tubules = binary_erosion(tubules, erode_kernel)
        tbm_tubule = binary_dilation(eroded_tubules, dilate_kernel)
        tbm = (tbm_tubule * ~eroded_tubules.astype(bool)) * cluster
        tbm = gaussian(tbm.astype(float), sigma=2.0) > 0.5
        tbm_mask[block.write_roi] = tbm

    block_roi = Roi((0, 0), (1000, 1000)) * fg_eroded_s0.voxel_size
    tbm_task = daisy.Task(
        "ID TBM",
        total_roi=s0_array.roi,
        read_roi=block_roi,
        write_roi=block_roi,
        read_write_conflict=False,
        num_workers=2,
        process_function=tbm_block,
    )
    daisy.run_blockwise(tasks=[tbm_task], multiprocessing=False)


def id_bc(
    fincap_mask: Array,
    structuralcollagen_mask: Array,
    fibrosis1_mask: Array,
    fibrosis2_mask: Array,
    bc_mask: Array,
    fg_eroded_s0: Array,
    s0_array: Array,
):
    erode_kernel = disk(14, decomposition="sequence")
    dilate_kernel = disk(
        9, decomposition="sequence"
    )  # sequence for computational efficiency

    def bc_block(block: daisy.Block):
        cap = fincap_mask[block.read_roi]
        cluster0 = structuralcollagen_mask[block.read_roi]
        cluster1 = fibrosis1_mask[block.read_roi]
        cluster2 = fibrosis2_mask[block.read_roi]
        cluster = cluster0 + cluster1 + cluster2
        eroded_cap = binary_erosion(cap, erode_kernel)
        dilated_cap = binary_dilation(cap, dilate_kernel)
        bc = (dilated_cap * (1 - eroded_cap)) * cluster
        bc = gaussian(bc.astype(float), sigma=2.0) > 0.5
        bc_mask[block.write_roi] = bc

    block_roi = Roi((0, 0), (1000, 1000)) * fg_eroded_s0.voxel_size
    bc_task = daisy.Task(
        "ID Bowman's Capsule",
        total_roi=s0_array.roi,
        read_roi=block_roi,
        write_roi=block_roi,
        read_write_conflict=False,
        num_workers=2,
        process_function=bc_block,
    )
    daisy.run_blockwise(tasks=[bc_task], multiprocessing=False)

def downsample_vessel(
        vessel_mask: Array,
        vessel10x: Array,
):
    downsample_factor = 4

    def downsample_block(block: daisy.Block):
        s0_data = vessel_mask[block.read_roi]
        new_shape = (
            s0_data.shape[0] // downsample_factor,
            downsample_factor,
            s0_data.shape[1] // downsample_factor,
            downsample_factor,
        )
        downsampled = s0_data.reshape(new_shape).mean(
            axis=(1, 3)
        )  # Average over grouped blocks
        vessel10x[block.write_roi] = downsampled

    block_roi = Roi((0, 0), (1000, 1000)) * vessel_mask.voxel_size

    downsample_task = daisy.Task(
        "blockwise_downsample",
        total_roi=vessel_mask.roi,
        read_roi=block_roi,
        write_roi=block_roi,
        read_write_conflict=False,
        num_workers=2,
        process_function=downsample_block,
    )
    daisy.run_blockwise(tasks=[downsample_task], multiprocessing=False)

def calculate_fibscore(
        finfib_mask: Array,
        fg_eroded_s0: Array,
        vessel_mask: Array,
        fincap_mask: Array,
        zarr_path: Path,
        s0_array: Array,
):
# blockwise mask multiplications
    def fibscore_block(block: daisy.Block):
        # in data
        fib = finfib_mask[block.read_roi]
        fg_er = fg_eroded_s0[block.read_roi]
        vessel = vessel_mask[block.read_roi]
        cap = fincap_mask[block.read_roi]
        # calc positive pixels in mask
        fibpx = np.count_nonzero(fib)
        fgpx = fg_er * (1 - vessel) * (1 - cap)
        fgpx = np.count_nonzero(fgpx)

        # write to text file
        # create text file
        with open(Path(zarr_path.parent / "fibpx.txt"), "a") as f:
            f.writelines(f"{fibpx} \n")
        with open(Path(zarr_path.parent / "tissuepx.txt"), "a") as f:
            f.writelines(f"{fgpx} \n")


    fibscore_task = daisy.Task(
        "fibscore calc",
        total_roi=s0_array.roi,
        read_roi=Roi((0, 0), (1000000, 1000000)),
        write_roi=Roi((0, 0), (1000000, 1000000)),
        read_write_conflict=False,
        num_workers=2,
        process_function=fibscore_block,
    )
    daisy.run_blockwise(tasks=[fibscore_task], multiprocessing=False)
