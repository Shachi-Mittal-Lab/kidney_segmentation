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

# check there is a GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using {device}")

# Tubule Model
tubule_model = "model_unet_dataset3_tubule0_LSDs_400.pt"
tubule_binary_head = "binaryhead_unet_dataset3_tubule0_LSDs_400.pt"

# Glom Model
glom_model = "model_unet_dataset5_01Jan2026_glom0_LSDs_final.pt"
glom_binary_head = "binaryhead_unet_dataset5_01Jan2026_glom0_LSDs_final.pt"

# Vessel Model
vessel_model = "model_unet_dataset12_vessel0_LSDs_final.pt"
vessel_binary_head = "binaryhead_unet_dataset12_vessel0_LSDs_final.pt"

gray_cluster_centers = Path(
    "/media/mrl/Data/pipeline_connection/kidney_segmentation/fibrosis_score/average_centers_5_skimage.txt" #Eric edit
)

tiff_dir = Path("/media/mrl/Elements/unet_annotations/inflammation_annotations")

def prepare_clustering(gray_cluster_centers):
    # get cluster centers
    centers = np.loadtxt(gray_cluster_centers)
    centers = np.expand_dims(centers, axis=1)
    n_clusters = len(centers)
    # Assign cluster centers
    clustering = KMeans(n_clusters=n_clusters)
    clustering.fit(centers)
    return clustering, n_clusters

def pred(model:Path, binary_head: Path, input_img): 
    model = torch.load(str(model), weights_only=False)
    binary_head = torch.load(str(binary_head), weights_only=False)
    model.eval()
    binary_head.eval()
    img = Image.open(input_img)
    img = img.resize((750,750))
    border_size = 153
    fill_color = (255, 255, 255) 
    img = ImageOps.expand(img, border=border_size, fill=fill_color)
    input = inp_transforms(img).unsqueeze(1).to(device)
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
    del model
    del binary_head
    return s0_data

def grayscale_cluster(
    img,
    clustering,
    n_clusters,
):
    img = np.array(Image.open(img))
    roi_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    roi_gray_normalized = exposure.adjust_sigmoid(
        roi_gray,
        cutoff=0.7,  # midpoint (normalized 0–1)
        gain=2,  # steepness (higher = steeper)
        inv=False,
    )
    roi_gray_flat = roi_gray_normalized.flatten()
    # k means predict
    roi_gray_flat = np.expand_dims(roi_gray_flat, axis=1)
    prediction = clustering.predict(roi_gray_flat)
    # order centers from clustering for ease of color reassignment
    index = np.arange(0, n_clusters, 1).reshape(-1, 1)
    centers = np.hstack((index, clustering.cluster_centers_))
    ascending_centers = np.array(sorted(centers, key=lambda x: x[1]))
    ordered_centers = ascending_centers[::-1]
    new_order = ordered_centers[:, 0].astype(int).reshape(-1, 1)
    mapping = np.array(
        [x[1] for x in sorted(zip(new_order, index))], dtype=int
    ).reshape(-1, 1)
    # updated labels
    class_labels = mapping[prediction].reshape(roi_gray.shape)
    # save centers
    final_centers = np.uint8(ordered_centers[:, 1:])
    # save inflammation 4
    inflammation = class_labels == 4
    return inflammation

# get cluster centers
clustering, n_clusters = prepare_clustering(gray_cluster_centers)

for tiff_file in tiff_dir.rglob("*"):
    if tiff_file.suffix.lower() in (".tif", ".tiff") and "mask" not in tiff_file.stem.lower():
        tubule_mask = pred(tubule_model, tubule_binary_head, tiff_file)
        glom_mask = pred(glom_model, glom_binary_head, tiff_file)
        vessel_mask = pred(vessel_model, vessel_binary_head, tiff_file)
        remove_structures_mask = tubule_mask + glom_mask + vessel_mask
        inflamm_cluster = grayscale_cluster(tiff_file, clustering, n_clusters)
        fin_inflamm = (1 - remove_structures_mask) * inflamm_cluster
        binary_img = (fin_inflamm != 0).astype(np.uint8)
        mask_file = tiff_file.with_name(tiff_file.stem + "_inflammmask" + tiff_file.suffix)
        tifffile.imwrite(mask_file, fin_inflamm)        