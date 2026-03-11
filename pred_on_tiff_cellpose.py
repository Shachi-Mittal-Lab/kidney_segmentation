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
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from skimage.morphology import remove_small_objects
from skimage import exposure

# Cellpose Tools
from cellpose import models, io
from cellpose.io import imread

# check there is a GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using {device}")

# Cellpose Model
cellpose_model = Path("CP_20260223_084158")
model = models.CellposeModel(gpu=True, pretrained_model=str(cellpose_model))

tiff_dir = Path("/media/mrl/Data/pipeline_connection/ndpis/inflammation_annotations_firstround")

for tiff_file in tiff_dir.rglob("*"):
    if tiff_file.suffix.lower() in (".tif", ".tiff") and "mask" not in tiff_file.stem.lower():
        img = imread(tiff_file)
        img = Image.fromarray(img)
        img = ImageOps.invert(img.convert("L"))
        arr = np.asarray(img) # X,Y,C
        masks, flows, styles = model.eval(
            arr, 
            channels=[0, 0],        # single-channel grayscale
            channel_axis=-1,        # last axis holds channels
            z_axis=None,            # 2D image
            normalize=True,
        )
        binary_mask = (masks > 0).astype(np.uint8)
        mask_file = tiff_file.with_name(tiff_file.stem + "_mask" + tiff_file.suffix)
        tifffile.imwrite(mask_file, binary_mask)        