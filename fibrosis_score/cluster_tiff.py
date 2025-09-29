import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import io, color, filters, feature
from pathlib import Path
from sklearn.cluster import KMeans
import cv2

# colors for overlay
red = [(0, 0, 0, 0), (1, 0, 0, 1)]
yellow = [(0, 0, 0, 0), "yellow"]
purple = [(0, 0, 0, 0), "purple"]
green = [(0, 0, 0, 0), "green"]
orange = [(0, 0, 0, 0), "orange"]
pink = [(0, 0, 0, 0), "pink"]
white = [(0, 0, 0, 0), "white"]
cyan = [(0, 0, 0, 0), "cyan"]
fuchsia = [(0, 0, 0, 0), "fuchsia"]
red = [(0, 0, 0, 0), "red"]

# sigmoid norm 
def sigmoid_norm(gray_array):
    xmax = np.amax(gray_array)
    xmin = np.amin(gray_array)
    newmax = 255
    newmin = 0
    beta = xmax - xmin
    alpha = 255 / 2
    eterm = np.exp(-(gray_array - beta) / alpha)
    gray_array = (newmax - newmin) * (1 / (1 + eterm)) + newmin
    return gray_array
def overlay_roi(roi_array, mask_array, colors, fig_path):
    fig, ax = plt.subplots()
    cmap = ListedColormap(colors)
    plt.axis("off")
    ax.imshow(roi_array)
    ax.imshow(mask_array, cmap=cmap, alpha=0.7)
    plt.savefig(fig_path, dpi=2000, bbox_inches="tight")

def overlay_rois(overlay1, overlay2, overlay3, overlay4, fig_path):
    fig, ax = plt.subplots()
    plt.axis("off")
    ax.imshow(overlay1, cmap=ListedColormap(green), alpha=1)
    ax.imshow(overlay2, cmap=ListedColormap(orange), alpha=1)
    ax.imshow(overlay3, cmap=ListedColormap(yellow), alpha=1)
    ax.imshow(overlay4, cmap=ListedColormap(red), alpha=1)
    plt.savefig(fig_path, dpi=2000, bbox_inches="tight")

def save_img(roi_array, fig_path):
    fig, ax = plt.subplots()
    plt.axis("off")
    ax.imshow(roi_array)
    plt.imsave(fig_path, roi_array)

# Load cluster centers 
# kmeans clustering centers
gray_cluster_centers = Path(
    "/home/riware/Documents/MittalLab/kidney_project/kidney_segmentation/fibrosis_score/average_centers_5.txt" #Eric edit
)

# Load image and convert to grayscale
identifier = "testing"
image_path = Path('/home/riware/Desktop/loose_files/clustering_investigation_18Aug2025/BR21-2023-B-A-1-9-TRICHROME - 2022-11-09 14.48.18_1_region_.tiff')
image_rgba = io.imread(image_path)
image_rgb = image_rgba[..., :3] if image_rgba.shape[-1] == 4 else image_rgba
image_rgb = image_rgb[:,:]

image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
image_gray = sigmoid_norm(image_gray)
image_gray_flat = image_gray.flatten()
image_gray = color.rgb2gray(image_rgb[:,:])

# Cluster image
centers = np.loadtxt(gray_cluster_centers)
centers = np.expand_dims(centers, axis=1)
n_clusters = len(centers)

# Assign cluster centers
clustering = KMeans(n_clusters=n_clusters)
clustering.fit(centers)

# k means predict
roi_gray_flat = np.expand_dims(image_gray_flat, axis=1)
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
class_labels = mapping[prediction].reshape(image_gray.shape)

# save centers
final_centers = np.uint8(ordered_centers[:, 1:])
fibrosis1 = class_labels == 1
fibrosis2 = class_labels == 2
structural_collagen = class_labels == 3
inflammation = class_labels == 4

# Generate overlays
# overlay_roi(image_rgb, fibrosis1, green, image_path.parent / f"{image_path.stem}{identifier}_fibrosis1_green.png")
# overlay_roi(image_rgb, fibrosis2, orange, image_path.parent / f"{image_path.stem}{identifier}_fibrosis2_orange.png")
# overlay_roi(image_rgb, structural_collagen, yellow, image_path.parent / f"{image_path.stem}{identifier}_structuralcollagen_yellow.png")
# overlay_roi(image_rgb, inflammation, red, image_path.parent / f"{image_path.stem}{identifier}_inflamm_red.png")
overlay_rois(fibrosis1, fibrosis2, structural_collagen, inflammation, image_path.parent / f"{image_path.stem}{identifier}_all_colors.png")