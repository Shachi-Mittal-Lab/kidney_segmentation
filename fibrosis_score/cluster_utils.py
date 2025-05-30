# Tools
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from skimage.morphology import remove_small_objects

# sigmoid normalization
# https://en.wikipedia.org/wiki/Normalization_(image_processing)
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


def prepare_clustering(gray_cluster_centers):
    # get cluster centers
    centers = np.loadtxt(gray_cluster_centers)
    centers = np.expand_dims(centers, axis=1)
    n_clusters = len(centers)
    # Assign cluster centers
    clustering = KMeans(n_clusters=n_clusters)
    clustering.fit(centers)
    return clustering, n_clusters


def grayscale_cluster(
    patchpath,
    patch_raw,
    voxel_roi,
    clustering,
    n_clusters,
    fibrosis1_mask,
    fibrosis2_mask,
    inflammation_mask,
    structuralcollagen_mask,
):
    
    patch_raw = np.transpose(patch_raw, axes=(1, 0, 2))
    patch_raw = patch_raw[:, :, ::-1]  # flip to rgb
    #roi_rgb = Image.fromarray(patch_raw, 'RGB')
    #cv2.imwrite(str(patchpath), patch_raw)
    # convert to grayscale & format for k means
    # roi_gbr = cv2.imread(str(patchpath))
    #roi_rgb = cv2.cvtColor(roi_gbr, cv2.COLOR_BGR2RGB)
    roi_gray = cv2.cvtColor(patch_raw, cv2.COLOR_RGB2GRAY)
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
    class_labels = mapping[prediction].reshape(roi_gray.shape)
    # save centers
    final_centers = np.uint8(ordered_centers[:, 1:])
    # save inflammation 4
    inflammation = class_labels == 4
    # visualize roi in mask, transpose to align
    inflammation_mask._source_data[
        voxel_roi.begin[0] : voxel_roi.end[0],
        voxel_roi.begin[1] : voxel_roi.end[1],
    ] = inflammation.T
    # save fibrosis 1
    fibrosis1 = class_labels == 1
    # connected components fib1
    fibrosis1 = remove_small_objects(fibrosis1.astype(bool), min_size=50)
    # visualize roi in mask, transpose to align
    fibrosis1_mask._source_data[
        voxel_roi.begin[0] : voxel_roi.end[0],
        voxel_roi.begin[1] : voxel_roi.end[1],
    ] = fibrosis1.T
    # save fibrosis 2
    fibrosis2 = class_labels == 2
    # connected components fib2
    fibrosis2 = remove_small_objects(fibrosis2.astype(bool), min_size=50)
    fibrosis2_mask._source_data[
        voxel_roi.begin[0] : voxel_roi.end[0],
        voxel_roi.begin[1] : voxel_roi.end[1],
    ] = fibrosis2.T
    # save structural collagen 3
    structural_collagen = class_labels == 3
    structuralcollagen_mask._source_data[
        voxel_roi.begin[0] : voxel_roi.end[0],
        voxel_roi.begin[1] : voxel_roi.end[1],
    ] = structural_collagen.T
    return
