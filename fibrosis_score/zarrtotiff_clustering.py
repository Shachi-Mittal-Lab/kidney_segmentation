from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


# Funke Lab Tools
import daisy
from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi

# image to visualize
zarr_path = Path("/home/riware/Desktop/loose_files/clustering_investigation_04Aug2025/E00499-A1-9-TRI - 2024-01-12 18.49.55_1.zarr")
zarr_folder = zarr_path.parent
zarr_name = zarr_path.stem
raw_path = zarr_path / "raw" / "s0"
identifier = "zoom"

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
    #plt.savefig(fig_path, dpi=2000, bbox_inches="tight", pad_inches=0)


# open raw and define roi location and size
raw = open_ds(raw_path)
roi = Roi((10000,10000), (3000,3000)) * raw.voxel_size  # E001

raw_roi = (roi - raw.offset) / raw.voxel_size
roi_data = raw._source_data[
    raw_roi.begin[0] : raw_roi.end[0],
    raw_roi.begin[1] : raw_roi.end[1],
]

# visualize roi
plt.imshow(roi_data)
plt.show()
# quit()
save_img(roi_data, zarr_folder / f"{zarr_name}_0_region.tiff")
# quit()
# clustering mask paths
fibrosis1 = zarr_path / "mask" / "fibrosis1"
fibrosis2 = zarr_path / "mask" / "fibrosis1"
structural_cluster = zarr_path / "mask" / "structuralcollagen"
inflamm_cluster = zarr_path / "mask" / "inflammation"

# open masks
fib1 = open_ds(fibrosis1)
fib2 = open_ds(fibrosis2)
struc_collagen = open_ds(structural_cluster)
inflamm = open_ds(inflamm_cluster)

# grab roi in mask
fib1_data = fib1._source_data[
    raw_roi.begin[0] : raw_roi.end[0],
    raw_roi.begin[1] : raw_roi.end[1],
]

fib2_data = fib2._source_data[
    raw_roi.begin[0] : raw_roi.end[0],
    raw_roi.begin[1] : raw_roi.end[1],
]

struc_collagen_data = struc_collagen._source_data[
    raw_roi.begin[0] : raw_roi.end[0],
    raw_roi.begin[1] : raw_roi.end[1],
]

inflamm_data = inflamm._source_data[
    raw_roi.begin[0] : raw_roi.end[0],
    raw_roi.begin[1] : raw_roi.end[1],
]

# Generate overlays
# overlay_roi(roi_data, fib1_data, green, zarr_folder / f"{zarr_name}_{identifier}_fibrosis1_green.png")
# overlay_roi(roi_data, fib2_data, orange, zarr_folder / f"{zarr_name}_{identifier}_fibrosis2_orange.png")
# overlay_roi(roi_data, struc_collagen_data, yellow, zarr_folder / f"{zarr_name}_{identifier}_structuralcollagen_yellow.png")
# overlay_roi(roi_data, inflamm_data, red, zarr_folder / f"{zarr_name}_{identifier}_inflamm_red.png")
overlay_rois(fib1_data, fib2_data, struc_collagen_data, inflamm_data, zarr_folder / f"{zarr_name}_{identifier}_all_colors.png")
