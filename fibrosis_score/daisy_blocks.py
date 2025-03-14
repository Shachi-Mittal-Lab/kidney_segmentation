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

# option 2
from functools import partial


def cap_prediction(
    cap_mask10x: Array,
    s2_array: Array,
    patch_size_final: Array,
    model: torch.nn.Module,
    device: torch.device,):

    def process_block(block:daisy.Block):
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
        s0_array: Array,):
    
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

