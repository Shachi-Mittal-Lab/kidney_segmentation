import os
import tifffile
import cv2
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

PREDS_DIR = "/home/pattonw/Rachel/Mittallab/fibrosis_score_pipeline_testing/preds_w_contrast_adj"

# reading the image
IMG_DIR = "/home/pattonw/Rachel/Mittallab/fibrosis_score_pipeline_testing/4-class_full_preds"
RESULTS_DIR = "/home/pattonw/Desktop/postprocessingexploration"
pred_patch_width, pred_patch_height = 48, 48 

# create folder for results 
if os.path.exists(RESULTS_DIR):
    pass
else: 
    os.mkdir(RESULTS_DIR)

# Create image list
images = [
    os.path.join(PREDS_DIR, img)
    for img in os.listdir(PREDS_DIR)
]


for img in images:
    roi = os.path.basename(img)
    adjusted_img = os.path.join(img, "4_clusters", "adjusted_img.tiff")

    if os.path.isfile(adjusted_img):
        stitched_img = adjusted_img

    else:
        stitched_img = os.path.join(img, "4_clusters", "stitched_4classmodel_originalimage.tiff")

    labels_img = os.path.join(img, "4_clusters", "mask_finished_napari_4classmodel_originalimage.tiff")

    original_tiff =  tifffile.imread(stitched_img)
    full_pred = tifffile.imread(labels_img)
    full_unpred = full_pred == 0
    tiff = tifffile.imread(labels_img)[::48, ::48]

    """
    # Pull tubules
    tubules = np.float32(tiff == 4)
    tubules_full = np.float32(full_pred == 4)
    tubules_full[tubules_full == 0] = np.nan

    # Denoise tubules

    denoised_tubule = cv2.bilateralFilter(tubules, 3, 2, 2)
    denoised_tubule = denoised_tubule.repeat(48, axis = 0).repeat(48, axis = 1)
    denoised_tubule[denoised_tubule == 0] = np.nan

    denoised_tubule1 = cv2.GaussianBlur(tubules, (3,3), 0)
    denoised_tubule1 = denoised_tubule1.repeat(48, axis = 0).repeat(48, axis = 1)
    denoised_tubule1[denoised_tubule1 == 0] = np.nan

    denoised_tubule2 = cv2.medianBlur(tubules, 3)
    denoised_tubule2 = denoised_tubule2.repeat(48, axis = 0).repeat(48, axis = 1)
    denoised_tubule2[denoised_tubule2 == 0] = np.nan

    # colormap
    cmap = mpl.colormaps["prism"].copy()
    cmap.set_under(color='black')

    # padding
    width_diff = original_tiff.shape[0] - full_unpred.shape[0]
    height_diff = original_tiff.shape[1] - full_unpred.shape[1]
    padded_unpred = np.pad(~full_unpred, ((0, width_diff), (0, height_diff)), 'constant', constant_values=(0,0))
    padded_unpred = np.expand_dims(padded_unpred, axis=2)

    # visualizing the results of denoising tubules
    plt.rcParams.update({'font.size': 5})
    plt.figure(figsize=(5,2), layout = "tight")

    plt.subplot(121, title="original"), plt.imshow(original_tiff * padded_unpred)
    plt.subplot(121, title="original"), plt.imshow(full_pred, alpha = 0.5, cmap=cmap, vmin=0.0000001)

    plt.subplot(122, title="tubules"), plt.imshow(original_tiff * padded_unpred)
    plt.subplot(122, title="tubules"), plt.imshow(tubules_full, alpha = 0.5, cmap=cmap, vmin=0.0000001)
    plt.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False)
    plt.savefig(os.path.join(RESULTS_DIR, f"{roi}_noprocessing_tubules.png"), dpi = 1500)

    plt.figure(figsize=(5,2), layout = "tight")
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)

    plt.subplot(131, title="bilateral"), plt.imshow(original_tiff * padded_unpred)
    plt.subplot(131, title="bilateral"), plt.imshow(denoised_tubule, alpha = 0.5, cmap="autumn")

    plt.subplot(132, title="gaussian"), plt.imshow(original_tiff * padded_unpred)
    plt.subplot(132, title="gaussian"), plt.imshow(denoised_tubule1, alpha = 0.5, cmap = "autumn")

    plt.subplot(133, title="median"), plt.imshow(original_tiff * padded_unpred)
    plt.subplot(133, title="median"), plt.imshow(denoised_tubule2, alpha = 0.5, cmap=cmap, vmin=0.0000001)
    plt.savefig(os.path.join(RESULTS_DIR, f"{roi}_processing_tubules.png"), dpi = 1500)
    """

    # Pull Gloms
    gloms = np.float32(tiff == 2)
    gloms_full = np.float32(full_pred == 2)
    gloms_full[gloms_full == 0] = np.nan

    # Denoise gloms

    denoised_gloms = cv2.bilateralFilter(gloms, 3, 9, 9)
    denoised_gloms = denoised_gloms.repeat(48, axis = 0).repeat(48, axis = 1)
    denoised_gloms[denoised_gloms == 0] = np.nan

    denoised_gloms1 = cv2.GaussianBlur(gloms, (19,19), 0)
    denoised_gloms1 = denoised_gloms1.repeat(48, axis = 0).repeat(48, axis = 1)
    denoised_gloms1[denoised_gloms1 == 0] = np.nan

    denoised_gloms2 = cv2.medianBlur(gloms, 3)
    denoised_gloms2 = denoised_gloms2.repeat(48, axis = 0).repeat(48, axis = 1)
    denoised_gloms2[denoised_gloms2 == 0] = np.nan

    # colormap
    cmap = mpl.colormaps["prism"].copy()
    cmap.set_under(color='black')

    # padding
    width_diff = original_tiff.shape[0] - full_unpred.shape[0]
    height_diff = original_tiff.shape[1] - full_unpred.shape[1]
    padded_unpred = np.pad(~full_unpred, ((0, width_diff), (0, height_diff)), 'constant', constant_values=(0,0))
    padded_unpred = np.expand_dims(padded_unpred, axis=2)

    # visualizing the results of denoising tubules
    plt.rcParams.update({'font.size': 5})
    plt.figure(figsize=(5,2), layout = "tight")

    plt.subplot(121, title="original"), plt.imshow(original_tiff * padded_unpred)
    plt.subplot(121, title="original"), plt.imshow(full_pred, alpha = 0.5, cmap=cmap, vmin=0.0000001)

    plt.subplot(122, title="gloms"), plt.imshow(original_tiff * padded_unpred)
    plt.subplot(122, title="gloms"), plt.imshow(gloms_full, alpha = 0.5, cmap=cmap, vmin=0.0000001)
    plt.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False)
    plt.savefig(os.path.join(RESULTS_DIR, f"{roi}_noprocessing_gloms.png"), dpi = 1500)

    plt.figure(figsize=(5,2), layout = "tight")
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)

    plt.subplot(131, title="bilateral"), plt.imshow(original_tiff * padded_unpred)
    plt.subplot(131, title="bilateral"), plt.imshow(denoised_gloms, alpha = 0.5, cmap="autumn")

    plt.subplot(132, title="gaussian"), plt.imshow(original_tiff * padded_unpred)
    plt.subplot(132, title="gaussian"), plt.imshow(denoised_gloms1, alpha = 0.5, cmap = "autumn")

    plt.subplot(133, title="median"), plt.imshow(original_tiff * padded_unpred)
    plt.subplot(133, title="median"), plt.imshow(denoised_gloms2, alpha = 0.5, cmap=cmap, vmin=0.0000001)
    plt.savefig(os.path.join(RESULTS_DIR, f"{roi}_processing_gloms.png"), dpi = 1500)
