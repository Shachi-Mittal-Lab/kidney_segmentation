# Repo Tools
from fibrosis_score.pred_utils import inp_transforms, inp_transforms_rgb
from model import UNet, ConvBlock, Downsample, CropAndConcat, OutputConv


# Tools
from pathlib import Path
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np
import tifffile

# check there is a GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using {device}")

model = torch.load("model_unet_dataset1_tubule0_LSDs_400.pt", weights_only=False)
binary_head = torch.load("binaryhead_unet_dataset1_tubule0_LSDs_400.pt", weights_only=False)

tiff_dir = Path("/mnt/49c617b2-221d-486c-9542-4ff0dc297048/kidney/kidney_segmentation/new_tubule_rois")

for tiff_file in tiff_dir.rglob("*"):
    if tiff_file.suffix.lower() in (".tif", ".tiff") and "mask" not in tiff_file.stem.lower():
        img = Image.open(tiff_file)
        img = img.resize((750,750))
        border_size = 153
        fill_color = (255, 255, 255) 
        img = ImageOps.expand(img, border=border_size, fill=fill_color)
        input = inp_transforms(img).unsqueeze(1).to(device)
        model.eval()
        binary_head.eval()
        with torch.no_grad():
            embedding = model(input)
            preds = binary_head(embedding)
            preds = preds[:,0,:,:]
            preds = preds.squeeze().cpu().detach().numpy()
            preds = preds > 0.5
        # crop preds
        upsampling_factor = (904/226, 904/226)
        s2_data = preds[152:902, 152:902]
        s0_data = s2_data
        for axis, reps in enumerate(upsampling_factor):
             s0_data = np.repeat(s0_data, reps, axis=axis)
        mask_file = tiff_file.with_name(tiff_file.stem + "_mask" + tiff_file.suffix)
        tifffile.imwrite(mask_file, s0_data)        