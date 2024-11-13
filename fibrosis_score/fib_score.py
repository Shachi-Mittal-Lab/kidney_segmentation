import zarr.convenience
from patch_utils_dask import openndpi, foreground_mask
from cluster_utils import sigmoid_norm
import dask
import dask.array
import zarr
import numpy as np
import cv2
from tqdm import tqdm
from skimage.transform import integral_image
from skimage.morphology import binary_erosion, binary_dilation, disk
from scipy.ndimage import label

from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from sklearn.cluster import KMeans

from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi
import daisy
from dask.diagnostics import ProgressBar


# inputs #

# file inputs
ndpi_path = Path(
    "/home/riware/Desktop/loose_files/pathmltest/BR21-2041-B-A-1-9-TRICHROME - 2022-11-09 16.31.43.ndpi"
)
zarr_path = Path("/home/riware/Desktop/loose_files/pathmltest/21-2041.zarr")

png_path = Path("/home/riware/Desktop/loose_files/pathmltest/pngs")
# background threshold
threshold = 50
# slide offset (zero)
offset = (0, 0)
# axis names
axis_names = [
    "x",
    "y",
    "c^",
]
# cortex vs medulla model info
cortmed_weights_path = "/home/riware/Documents/MittalLab/kidney_project/fibrosis_score/omniseg/cortexvsmedullavsartifact/model5_epoch8.pt"
cortmed_starting_model_path = "/home/riware/Documents/MittalLab/kidney_project/fibrosis_score/omniseg/cortexvsmedullavsartifact/starting_model.pt"

# tissue vs background model info
fgbg_starting_model_path = "/home/riware/Documents/MittalLab/kidney_project/fibrosis_score/omniseg/foregroundvsbackground/models/starting_model.pt"
fgbg_weights_path = "/home/riware/Documents/MittalLab/kidney_project/fibrosis_score/omniseg/foregroundvsbackground/models/modelF8/train/modelF8_epoch3.pt"

# kmeans clustering centers
gray_cluster_centers = "/home/riware/Documents/MittalLab/kidney_project/kidney_segmentation/fibrosis_score/average_centers_5.txt"

###########

"""# convert ndpi to zarr array levels 40x, 20x, 10x, 5x
for i in range(0, 4):
    # open the ndpi with openslide for info and tifffile as zarr
    dask_array, x_res, y_res, units = openndpi(ndpi_path, i)
    # define units in nm since native is cm and then you get a (0,0) res
    units = ("nm", "nm")
    # grab resolution in cm, convert to nm, calculate for each pyramid level
    voxel_size = Coordinate(int(1 / x_res * 1e7) * 2**i, int(1 / y_res * 1e7) * 2**i)
    # format data as funlib dataset
    raw = prepare_ds(
        zarr_path / "raw" / f"s{i}",
        dask_array.shape,
        offset,
        voxel_size,
        axis_names,
        units,
        dtype=np.uint8,
    )
    # storage info
    store_rgb = zarr.open(zarr_path / "raw" / f"s{i}")
    # store info with storage info

    # if image can be loaded in memory do this:
    # store_rgb[:] = dask_array[:]
    dask_array = dask_array.rechunk(raw.data.chunksize)
    with ProgressBar():
        dask_array.compute()
    with ProgressBar():
        dask.array.store(dask_array, store_rgb)

    # open pyramid level and assign appropriate voxel size and units to the dataset
    ds = zarr.open(zarr_path / "raw" / f"s{i}")
    ds.attrs["voxel_size"] = voxel_size
    ds.attrs["units"] = units"""

# grab new zarr with reasonable chunks as dask array
s3_array = open_ds(zarr_path / "raw" / "s3")

# create and save foreground mask
mask = foreground_mask(s3_array.data, threshold)
fmask = prepare_ds(
    zarr_path / "mask" / "foreground",
    s3_array.shape[0:2],
    s3_array.offset,
    s3_array.voxel_size,
    s3_array.axis_names[0:2],
    s3_array.units,
    mode="w",
    dtype=np.uint8,
)
store_fgbg = zarr.open(zarr_path / "mask" / "foreground")
dask.array.store(mask, store_fgbg)

# fill holes in foreground mask and save as filled mask
fill_hole_kernel = disk(
    15, decomposition="sequence"
)  # sequence for computational efficiency

# fill holes
shrink_kernel = disk(10, decomposition="sequence")
mask_filled = binary_erosion(binary_dilation(mask, fill_hole_kernel), shrink_kernel)
mask_filled_array = prepare_ds(
    zarr_path / "mask" / "foreground_filled",
    s3_array.shape[0:2],
    s3_array.offset,
    s3_array.voxel_size,
    s3_array.axis_names[0:2],
    s3_array.units,
    mode="w",
    dtype=np.uint8,
)
mask_filled_array._source_data[:] = mask_filled

# erosion to shave away edge tissue
shrink_kernel = disk(
    40, decomposition="sequence"
)  # sequence for computational efficiency
mask_eroded = binary_erosion(binary_dilation(mask, fill_hole_kernel), shrink_kernel)
mask_eroded_array = prepare_ds(
    zarr_path / "mask" / "foreground_eroded",
    s3_array.shape[0:2],
    s3_array.offset,
    s3_array.voxel_size,
    s3_array.axis_names[0:2],
    s3_array.units,
    mode="w",
    dtype=np.uint8,
)
mask_eroded_array._source_data[:] = mask_eroded

# create integral mask of filled foreground mask and save
integral_mask = integral_image(mask_filled)
imask = prepare_ds(
    zarr_path / "mask" / "integral_foreground",
    s3_array.shape[0:2],
    s3_array.offset,
    s3_array.voxel_size,
    s3_array.axis_names[0:2],
    s3_array.units,
    mode="w",
    dtype=np.uint64,
)
imask._source_data[:] = integral_mask

# create patch mask: this marks every pixel that can be the starting pixel
# of a patch of the defined patch size that will encompass mostly tissue
patch_shape = Coordinate(224, 224)  # Will convention: shape in voxels started 150 x 150
patch_size = patch_shape * imask.voxel_size  # size in nm
patch_roi = Roi(imask.roi.offset, imask.roi.shape - patch_size)
patch_counts = (
    imask[patch_roi]  # indexing with a roi gives you a plain numpy array
    + imask[patch_roi + patch_size]
    - imask[patch_roi + patch_size * Coordinate(0, 1)]
    - imask[patch_roi + patch_size * Coordinate(1, 0)]
)
patch_count_mask = (
    patch_counts >= patch_shape[0] * patch_shape[1] * 0.50
)  # 50% of voxels are tissue

# make grid on s3
patch_spacing = Coordinate(112, 112)  # was 100x100

# very strange... why we give y shape then x shape... but it works
grid_x, grid_y = np.meshgrid(
    range(patch_count_mask.shape[1]), range(patch_count_mask.shape[0])
)
gridmask = grid_x % patch_spacing[0] + grid_y % patch_spacing[1]
gridmask = gridmask == 0

# create mask of pixels for possible starting pixels gien the ROI size and threshold
patch_mask = prepare_ds(
    zarr_path / "mask" / "patch_mask",
    patch_count_mask.shape,
    s3_array.offset,
    s3_array.voxel_size,
    s3_array.axis_names[0:2],
    s3_array.units,
    mode="w",
    dtype=np.uint8,
)
patch_mask._source_data[:] = patch_count_mask * gridmask
# dask.array.store(patch_count_mask, store_patch_mask)

coords = np.nonzero(patch_mask._source_data[:])
offsets = [
    patch_mask.roi.offset + Coordinate(a, b) * patch_mask.voxel_size
    for a, b in zip(*coords)
]

# apply foreground vs background model at 5x (s3)
s3_array = open_ds(zarr_path / "raw" / "s3")

with torch.no_grad():
    # load vgg16 weights
    checkpoint = torch.load(
        cortmed_weights_path, map_location=torch.device("cpu"), weights_only=False
    )
    # load vgg16 architecture
    vgg16: torch.nn.Module = torch.load(
        cortmed_starting_model_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    # load weights into architecture
    vgg16.load_state_dict(checkpoint["model_state_dict"])

    pred_mask = open_ds(zarr_path / "mask" / "remove_artifacts")
    ## make dataset for mask
    #pred_mask = prepare_ds(
    #    zarr_path / "mask" / "remove_artifacts",
    #    s3_array.shape[0:2],
    #    s3_array.offset,
    #    s3_array.voxel_size,
    #    s3_array.axis_names[0:2],
    #    s3_array.units,
    #    mode="w",
    #    dtype=np.uint8,
    #)
    ## default pixel value to 2
    #pred_mask[:] = 3    # for each patch, send to vgg16 and write label to mask
    #for offset in tqdm(offsets):
    #    # world units roi selection
    #    roi = Roi(offset, patch_size)
    #    voxel_roi = (roi - s3_array.offset) / s3_array.voxel_size
    #    patch_raw = s3_array[
    #        voxel_roi.begin[0] : voxel_roi.end[0], voxel_roi.begin[1] : voxel_roi.end[1]
    #    ]
    #    # format roi for vgg16
    #    in_data = T.Resize((224, 224))(
    #        torch.from_numpy(patch_raw).float().permute(2, 0, 1).unsqueeze(0) / 255
    #    )
    #    # predict
    #    pred = vgg16(in_data).squeeze(0).numpy().argmax()        
    #    #  write to mask, accounting for roi overlap
    #    context = (patch_size - patch_spacing * s3_array.voxel_size) // 2
    #    slices = pred_mask._Array__slices(roi.grow(-context, -context))
    #    pred_mask._source_data[slices] = pred
    #    pred_mask._source_data[:] = pred_mask._source_data[:] == 1

# create patch mask: this marks every pixel that can be the starting pixel
# of a patch of the defined patch size that will encompass mostly tissue
patch_shape_final = Coordinate(
    150, 150
)  # Will convention: shape in voxels started 150 x 150
patch_size_final = patch_shape_final * imask.voxel_size  # size in nm
patch_roi_final = Roi(imask.roi.offset, imask.roi.shape - patch_size_final)
patch_counts_final = (
    imask[patch_roi_final]  # indexing with a roi gives you a plain numpy array
    + imask[patch_roi_final + patch_size_final]
    - imask[patch_roi_final + patch_size_final * Coordinate(0, 1)]
    - imask[patch_roi_final + patch_size_final * Coordinate(1, 0)]
)
patch_count_mask_final = (
    patch_counts_final >= patch_shape_final[0] * patch_shape_final[1] * 0.50
)  # 50% of voxels are tissue

# make grid on s3
patch_spacing_final = Coordinate(100, 100)  # was 100x100

# very strange... why we give y shape then x shape... but it works
grid_x_final, grid_y_final = np.meshgrid(
    range(patch_count_mask_final.shape[1]), range(patch_count_mask_final.shape[0])
)
gridmaskfinal = (
    grid_x_final % patch_spacing_final[0] + grid_y_final % patch_spacing_final[1]
)
gridmaskfinal = gridmaskfinal == 0

# create mask of pixels for possible starting pixels gien the ROI size and threshold
patch_mask_final = prepare_ds(
    zarr_path / "mask" / "patch_mask_final",
    patch_count_mask_final.shape,
    s3_array.offset,
    s3_array.voxel_size,
    s3_array.axis_names[0:2],
    s3_array.units,
    mode="w",
    dtype=np.uint8,
)
patch_mask_final._source_data[:] = (
    patch_count_mask_final[:]
    * gridmaskfinal
    * pred_mask._source_data[: -patch_shape_final[0], : -patch_shape_final[1]]
)

# dask.array.store(patch_count_mask, store_patch_mask)
coords = np.nonzero(patch_mask_final._source_data[:])
offsets_final = [
    patch_mask_final.roi.offset + Coordinate(a, b) * patch_mask_final.voxel_size
    for a, b in zip(*coords)
]
pyramid = [0, 1, 2, 3]

# create visualization of regions pulled
roi_mask = prepare_ds(
    zarr_path / "mask" / "rois",
    patch_count_mask.shape,
    s3_array.offset,
    s3_array.voxel_size,
    s3_array.axis_names[0:2],
    s3_array.units,
    mode="w",
    dtype=np.uint8,
)
roi_mask._source_data[:] = 0

# get cluster centers
centers = np.loadtxt(gray_cluster_centers)
centers = np.expand_dims(centers, axis=1)
n_clusters = len(centers)
# Assign cluster centers
clustering = KMeans(n_clusters=n_clusters)
clustering.fit(centers)

# create visualization of fibrosis clusters
s0_array = open_ds(zarr_path / "raw" / "s0")

fibrosis1_mask = prepare_ds(
    zarr_path / "mask" / "fibrosis1",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
fibrosis1_mask._source_data[:] = 0

fibrosis2_mask = prepare_ds(
    zarr_path / "mask" / "fibrosis2",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
fibrosis2_mask._source_data[:] = 0

# create visualization of inflammation cluster
inflammation_mask = prepare_ds(
    zarr_path / "mask" / "inflammation",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
inflammation_mask._source_data[:] = 0

# create visualization of structural collagen cluster
structuralcollagen_mask = prepare_ds(
    zarr_path / "mask" / "structuralcollagen",
    s0_array.shape[0:2],
    s0_array.offset,
    s0_array.voxel_size,
    s0_array.axis_names[0:2],
    s0_array.units,
    mode="w",
    dtype=np.uint8,
)
structuralcollagen_mask._source_data[:] = 0

for offset in offsets_final:
    print(offset)
    # world units roi selection
    roi = Roi(offset, patch_size_final)  # in nm
    # visualize roi in mask
    voxel_mask_roi = (roi - roi_mask.offset) / roi_mask.voxel_size
    roi_mask._source_data[
        voxel_mask_roi.begin[0] : voxel_mask_roi.end[0],
        voxel_mask_roi.begin[1] : voxel_mask_roi.end[1],
    ] = 1

    # save image at diff pyramid levels for Omni-Seg input
    for level in pyramid:
        # name roi
        patchname = f"{zarr_path.stem}_{offset}_pyr{level}.png"
        patchpath = png_path / patchname

        # select roi
        array = open_ds(zarr_path / "raw" / f"s{level}")
        voxel_roi = (roi - array.offset) / array.voxel_size
        patch_raw = array[
            voxel_roi.begin[0] : voxel_roi.end[0], voxel_roi.begin[1] : voxel_roi.end[1]
        ]
        patch_raw = np.transpose(patch_raw, axes=(1, 0, 2))
        patch_raw = patch_raw[:, :, ::-1]  # flip to rgb
        cv2.imwrite(patchpath, patch_raw)

        # cluster 40x pyramid level
        if level == 0:
            # convert to grayscale & format for k means
            roi_gbr = cv2.imread(patchpath)
            roi_rgb = cv2.cvtColor(roi_gbr, cv2.COLOR_BGR2RGB)
            roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2GRAY)
            roi_gray = sigmoid_norm(roi_gray)
            roi_gray_flat = roi_gray.flatten()
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
            labels = mapping[prediction].reshape(roi_gray.shape)
            print(np.unique(labels))
            # save centers
            final_centers = np.uint8(ordered_centers[:, 1:])

            # save inflammation 4
            inflammation = labels == 4
            # visualize roi in mask, transpose to align
            inflammation_mask._source_data[
                voxel_roi.begin[0] : voxel_roi.end[0],
                voxel_roi.begin[1] : voxel_roi.end[1],
            ] = inflammation.T

            # save fibrosis 1
            fibrosis1 = labels == 1
            # connected components fib1
            structure = np.ones((3, 3), dtype=np.int8)  # allows any connection
            #labels, ncomponents = label(fibrosis1, structure)
            #unique, counts = np.unique(labels, return_counts=True)
            #small_obj = counts <= 200
            #to_remove = unique[small_obj]
            #small_obj_mask = np.isin(labels, to_remove)
            #fibrosis1[small_obj_mask] = 0
#
            # visualize roi in mask, transpose to align
            fibrosis1_mask._source_data[
                voxel_roi.begin[0] : voxel_roi.end[0],
                voxel_roi.begin[1] : voxel_roi.end[1],
            ] = fibrosis1.T

            # save fibrosis 2
            fibrosis2 = labels == 2
            # connected components fib2
            #structure = np.ones((3, 3), dtype=np.int8)  # allows any connection
            #labels, ncomponents = label(fibrosis2, structure)
            #unique, counts = np.unique(labels, return_counts=True)
            #small_obj = counts <= 100
            #to_remove = unique[small_obj]
            #small_obj_mask = np.isin(labels, to_remove)
            #fibrosis2[small_obj_mask] = 0
            fibrosis2_mask._source_data[
                voxel_roi.begin[0] : voxel_roi.end[0],
                voxel_roi.begin[1] : voxel_roi.end[1],
            ] = fibrosis2.T
            # save structural collagen 3
            structural_collagen = labels == 3
            structuralcollagen_mask._source_data[
                voxel_roi.begin[0] : voxel_roi.end[0],
                voxel_roi.begin[1] : voxel_roi.end[1],
            ] = structural_collagen.T
    # grab omniseg output
    # apply clustering
    # save outputs to zarr

