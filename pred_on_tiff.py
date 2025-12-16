# Repo Tools
from fibrosis_score.pred_utils import inp_transforms, inp_transforms_rgb

# Tools
from pathlib import Path
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np

# check there is a GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using {device}")

model = torch.load("model_unet_dataset1_tubule0_LSDs_400.pt", weights_only=False)
binary_head = torch.load("binaryhead_unet_dataset1_tubule0_LSDs_400.pt", weights_only=False)

tiff_path = Path("/home/riware/Desktop/mittal_lab/uic_regions/21-2002 A-1-9 Trich - 2021-03-22 12.36.47_2_0_region.tiff")
img = Image.open(tiff_path)
img = img.resize((750,750))
border_size = 306
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
s2_data = preds[306:750, 306:750]
s0_data = s2_data
for axis, reps in enumerate(upsampling_factor):
     s0_data = np.repeat(s0_data, reps, axis=axis)

# save 3000x3000 mask in napari readable format