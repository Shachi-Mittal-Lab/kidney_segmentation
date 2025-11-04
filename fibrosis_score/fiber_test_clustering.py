import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, feature
from pathlib import Path
from sklearn.cluster import KMeans
import cv2

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

# Load cluster centers 
# kmeans clustering centers
gray_cluster_centers = Path(
    "/home/riware/Documents/MittalLab/kidney_project/kidney_segmentation/fibrosis_score/average_centers_5.txt" #Eric edit
)

# Load image and convert to grayscale
image_rgba = io.imread('/home/riware/Desktop/loose_files/kpmp_regions/5a91767b-cff9-423f-afb3-0f8cf256d20c_S-2402-002819_TRI_1of2_0_region_.tiff')
image_rgb = image_rgba[..., :3] if image_rgba.shape[-1] == 4 else image_rgba
image_rgb = image_rgb[:500,:500]

image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
image_gray = sigmoid_norm(image_gray)
image_gray_flat = image_gray.flatten()
image_gray = color.rgb2gray(image_rgb[:500,:500])

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
inflammation = class_labels == 4
everythingbutinflamm = (image_gray * ~inflammation)
plt.imshow(everythingbutinflamm, cmap='gray')
plt.show()

# Apply Gaussian smoothing to reduce noise
smoothed = filters.gaussian(image_gray, sigma=1)

# Compute structure tensor components
Axx, Axy, Ayy = feature.structure_tensor(image_gray, sigma=1)

# Compute orientation angles
orientation = 0.5 * np.arctan2(2 * Axy, Axx - Ayy)

# Sample grid for vector field visualization
step = 20
Y, X = np.mgrid[0:image_gray.shape[0]:step, 0:image_gray.shape[1]:step]
U = np.cos(orientation[::step, ::step])
V = np.sin(orientation[::step, ::step])

# Plot grayscale image with orientation vectors
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image_rgb, cmap='gray')
ax.quiver(X, Y, U, V, color='red', pivot='middle', scale=20)
ax.set_title('Local Fiber Orientation via Structure Tensor')
ax.axis('off')
plt.tight_layout()
plt.show()
