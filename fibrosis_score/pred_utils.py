# Funke Lab Tools
from funlib.geometry import Roi

# Tools
import torch
import torchvision.transforms as T
from tqdm import tqdm
import sys

from directory_paths import (
    cortmed_weights_path,
    cortmed_starting_model_path,
)

# cortext stuff Vgg16
def pred_cortex(inarray, offsets, patch_size, patch_spacing, pred_mask, device):
    
    with torch.no_grad():
        print("Identifying Cortex")
        # load vgg16 weights
        checkpoint = torch.load(
            cortmed_weights_path, map_location=torch.device("cpu"), weights_only=False
        )
        # load vgg16 architecture
        vgg16: torch.nn.Module = torch.load(
            cortmed_starting_model_path,
            map_location=device,
            weights_only=False,
        )
        # load weights into architecture
        vgg16.load_state_dict(checkpoint["model_state_dict"])
        vgg16.to(device)

        # for each patch, send to vgg16 and write label to mask
        for offset in tqdm(offsets):
            # world units roi selection
            roi = Roi(offset, patch_size)
            voxel_roi = (roi - inarray.offset) / inarray.voxel_size
            patch_raw = inarray[
                voxel_roi.begin[0] : voxel_roi.end[0], voxel_roi.begin[1] : voxel_roi.end[1]
            ]
            # format roi for vgg16
            in_data = T.Resize((224, 224))(
                torch.from_numpy(patch_raw).float().permute(2, 0, 1).unsqueeze(0) / 255
            ).to(device)
            # predict
            pred = vgg16(in_data).cpu().squeeze(0).numpy().argmax()
            #  write to mask, accounting for roi overlap
            context = (patch_size - patch_spacing * inarray.voxel_size) // 2
            slices = pred_mask._Array__slices(roi.grow(-context, -context))
            pred_mask._source_data[slices] = pred
            pred_mask._source_data[:] = pred_mask._source_data[:] == 1
    # remove vgg16 from gpu memmory
    del vgg16
    return

# Unet stuff
inp_transforms = T.Compose(
    [
        T.Grayscale(),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
    ]
)
