from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


# Funke Lab Tools
import daisy
from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi

# image to visualize
zarr_path = Path("/home/riware/Desktop/loose_files/glom_annotations/zarrs/BR22-2083-A-1-9-TRICHROME - 2022-11-09 16.01.24.zarr")
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


def overlay_roi(roi_array, mask_array, colors, fig_path):
    fig, ax = plt.subplots()
    cmap = ListedColormap(colors)
    plt.axis("off")
    ax.imshow(roi_array)
    ax.imshow(mask_array, cmap=cmap, alpha=0.7)
    plt.savefig(fig_path, dpi=2000, bbox_inches="tight")


def save_img(roi_array, fig_path):
    fig, ax = plt.subplots()
    plt.axis("off")
    ax.imshow(roi_array)
    plt.imsave(fig_path, roi_array)
    #plt.savefig(fig_path, dpi=2000, bbox_inches="tight", pad_inches=0)


# open raw and define roi location and size
raw = open_ds(raw_path)
# roi = Roi((0, 0), (10000, 30000)) * raw.voxel_size  # 22-2062A
# roi = Roi((2000, 1000), (3000, 3000)) * raw.voxel_size # 22-2062A zoom in
# roi = Roi((6000, 20000), (6000, 40000)) * raw.voxel_size # 21-2062B
# roi = Roi((8000, 48000), (3000, 4000)) * raw.voxel_size # 21-2062B zoom-in
# roi = Roi((8000, 8000), (18000, 18000)) * raw.voxel_size # 21-2049B
# roi = Roi((17000, 8000), (3000, 3000)) * raw.voxel_size # 21-2049B zoom-in
# roi = Roi((0, 0), (11000, 30000)) * raw.voxel_size # 21-2023B
# glom annotation regions
# roi = Roi((1000,6000), (3000, 3000)) * raw.voxel_size  # 21-2000 PAS region 0
# roi = Roi((31000,18500), (3000, 3000)) * raw.voxel_size  # 21-2000 PAS region 1
# roi = Roi((33000,26000), (3000, 3000)) * raw.voxel_size  # 21-2000 PAS region 2
# roi = Roi((2000,5500), (3000, 3000)) * raw.voxel_size  # 21-2000 TRI region 0
# roi = Roi((32500,17000), (3000, 3000)) * raw.voxel_size  # 21-2000 TRI region 1
# roi = Roi((36000,25000), (3000, 3000)) * raw.voxel_size  # 21-2000 TRI region 2
# roi = Roi((6000,58000), (3000, 3000)) * raw.voxel_size  # 21-2015 TRI region 0
# roi = Roi((6700,61000), (3000, 3000)) * raw.voxel_size  # 21-2015 TRI region 1
# roi = Roi((26500,5500), (3000, 3000)) * raw.voxel_size  # 21-2015 TRI region 2
# roi = Roi((22500,10000), (3000, 3000)) * raw.voxel_size  # 21-2015 TRI region 3
# roi = Roi((16000,13000), (3000,3000)) * raw.voxel_size  # 21-2008 TRI region 0
# roi = Roi((28000,21000), (3000,3000)) * raw.voxel_size  # 21-2008 TRI region 1
# roi = Roi((50000,34000), (3000,3000)) * raw.voxel_size  # 21-2008 TRI region 2
# roi = Roi((1750,9000), (3000,3000)) * raw.voxel_size  # 21-2056 TRI region 0
# roi = Roi((48000,4000), (3000,3000)) * raw.voxel_size  # 16A TRI region 0
# roi = Roi((48000,12250), (3000,3000)) * raw.voxel_size  # 16A TRI region 1
# roi = Roi((35600,25000), (3000,3000)) * raw.voxel_size  # 16A TRI region 2
# roi = Roi((39000,36250), (3000,3000)) * raw.voxel_size  # 16A TRI region 3
# roi = Roi((32500,17000), (3000,3000)) * raw.voxel_size  # 22-2099 A TRI region 0
roi = Roi((56500,47000), (3000,3000)) * raw.voxel_size  # 22-2083 A TRI region 0


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
quit()
# mask paths
fib_path = zarr_path / "mask" / "finfib"
structural_collagen_path = zarr_path / "mask" / "fincollagen"
inflamm = zarr_path / "mask" / "fininflamm"
dt_path = zarr_path / "mask" / "dt"
pt_path = zarr_path / "mask" / "pt"
vessel_path = zarr_path / "mask" / "vessel"
cap_path = zarr_path / "mask" / "fincap"
# clustering mask paths
fibrosis1 = zarr_path / "mask" / "fibrosis1"
fibrosis2 = zarr_path / "mask" / "fibrosis1"
structural_cluster = zarr_path / "mask" / "structuralcollagen"
inflamm_cluster = zarr_path / "mask" / "inflammation"

# open masks
fib = open_ds(fib_path)
fincollagen = open_ds(structural_collagen_path)
fininflamm = open_ds(inflamm)
dt = open_ds(dt_path)
pt = open_ds(pt_path)
vessel = open_ds(vessel_path)
cap = open_ds(cap_path)


# grab roi in mask
fib_data = fib._source_data[
    raw_roi.begin[0] : raw_roi.end[0],
    raw_roi.begin[1] : raw_roi.end[1],
]

collagen_data = fincollagen._source_data[
    raw_roi.begin[0] : raw_roi.end[0],
    raw_roi.begin[1] : raw_roi.end[1],
]

inflamm_data = fininflamm._source_data[
    raw_roi.begin[0] : raw_roi.end[0],
    raw_roi.begin[1] : raw_roi.end[1],
]

dt_data = dt._source_data[
    raw_roi.begin[0] : raw_roi.end[0],
    raw_roi.begin[1] : raw_roi.end[1],
]

pt_data = pt._source_data[
    raw_roi.begin[0] : raw_roi.end[0],
    raw_roi.begin[1] : raw_roi.end[1],
]

vessel_data = vessel._source_data[
    raw_roi.begin[0] : raw_roi.end[0],
    raw_roi.begin[1] : raw_roi.end[1],
]

cap_data = cap._source_data[
    raw_roi.begin[0] : raw_roi.end[0],
    raw_roi.begin[1] : raw_roi.end[1],
]

# Generate overlays
# overlay_roi(roi_data, fib_data, red, zarr_folder / f"{zarr_name}_{identifier}_fibrosis_red.png")
# overlay_roi(roi_data, collagen_data, yellow, zarr_folder / f"{zarr_name}_{identifier}_structural_collagen_yellow.png")
# overlay_roi(roi_data, inflamm_data, fuchsia, zarr_folder / f"{zarr_name}_{identifier}_inflammation_fuchsia.png")
# overlay_roi(roi_data, inflamm_data, fuchsia, zarr_folder / f"{zarr_name}_{identifier}_inflammation_fuchsia.png")
# overlay_roi(roi_data, dt_data, green, zarr_folder / f"{zarr_name}_{identifier}_dt_green.png")
# overlay_roi(roi_data, pt_data, orange, zarr_folder / f"{zarr_name}_{identifier}_pt_orange.png")
# overlay_roi(roi_data, vessel_data, pink, zarr_folder / f"{zarr_name}_{identifier}_vessel_pink.png")
overlay_roi(roi_data, cap_data, white, zarr_folder / f"{zarr_name}_{identifier}_glom_white.png")
