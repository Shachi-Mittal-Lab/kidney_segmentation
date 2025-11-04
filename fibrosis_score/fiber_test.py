import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, feature, morphology, measure
from pathlib import Path
from sklearn.cluster import KMeans
import cv2


# Load image and convert to grayscale
# image_rgba = io.imread('/home/riware/Desktop/loose_files/kpmp_regions/03f9da88-1dbd-44f3-8a2a-dbefe629898b_S-2308-002019_TRI_1of2_0_region.tiff')
image_rgba = io.imread('/home/riware/Desktop/loose_files/tubule_annotations/dt_pt/tri/tri_im_2.png')
image_rgb = image_rgba[..., :3] if image_rgba.shape[-1] == 4 else image_rgba
image_rgb = image_rgb[:,:]
image_gray = color.rgb2gray(image_rgb[:,:])

plt.imshow(image_gray, cmap="gray")
plt.show()

# sharpen
sharpened = filters.unsharp_mask(image_gray, radius=2, amount=5)
plt.imshow(sharpened, cmap="gray")
plt.show()

# Compute structure tensor components
Axx, Axy, Ayy = feature.structure_tensor(sharpened, sigma=3)

# Compute orientation angles
orientation = 0.5 * np.arctan2(2 * Axy, Axx - Ayy)

# Sample grid for vector field visualization
step = 20
Y, X = np.mgrid[0:image_rgb.shape[0]:step, 0:image_rgb.shape[1]:step]
U = np.cos(orientation[::step, ::step])
V = np.sin(orientation[::step, ::step])

# Plot grayscale image with orientation vectors
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image_rgb, cmap='gray')
ax.quiver(X, Y, U, V, color='black', pivot='middle', scale=50)
ax.set_title('Local Fiber Orientation via Structure Tensor')
ax.axis('off')
plt.tight_layout()
plt.show()

# Compute gradient of orientation
orientation_dx = filters.sobel_h(orientation)
orientation_dy = filters.sobel_v(orientation)
orientation_gradient_magnitude = np.hypot(orientation_dx, orientation_dy)

# Visualize orientation gradient magnitude as a heatmap
fig, ax = plt.subplots(figsize=(8, 8))
heatmap = ax.imshow(orientation_gradient_magnitude, cmap='inferno')
ax.set_title('Orientation Gradient Magnitude Heatmap')
ax.axis('off')
plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

"""
# Identify regions with minimal change (low gradient)
#low_threshold = np.percentile(orientation_gradient_magnitude, 20)
low_threshold = 1.5
low_change_regions = orientation_gradient_magnitude < low_threshold

# Overlay these regions on the grayscale image
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image_rgb, cmap='gray')
ax.imshow(low_change_regions, cmap='gray', alpha=0.5)
ax.set_title('Regions of Minimal Directional Change in Fiber Orientation')
ax.axis('off')
plt.tight_layout()
plt.show()
"""
