# Tools
from pathlib import Path
from PIL import Image, ImageOps

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have PyQt5 installed separately
import matplotlib.pyplot as plt

import numpy as np
import cv2

tiff_dir = Path("/home/riware/Desktop/mittal_lab/bg_removal_test_regions/original_imgs")

def remove_trichrome_background(image_path, sat_thresh=30, val_thresh=210):
    """
    Remove background from Masson's Trichrome stained tissue image.
    Returns the image with transparent background for visualization.
    
    Parameters:
    - sat_thresh: Saturation threshold (higher = more aggressive, default 30)
    - val_thresh: Value/brightness threshold (lower = more aggressive, default 210)
    """
    print(f"Processing: {image_path}")
    
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    
    # Downsample for faster processing if image is large
    original_shape = img.shape[:2]
    max_dim = 2048
    
    if max(original_shape) > max_dim:
        scale = max_dim / max(original_shape)
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img_small = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        print(f"Downsampled to: {img_small.shape[1]}x{img_small.shape[0]} for processing")
        process_small = True
    else:
        img_small = img
        process_small = False
    
    # Convert to HSV
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]  # Saturation channel
    v = hsv[:, :, 2]  # Value channel
    
    # Background: low saturation AND high brightness
    # More aggressive thresholds
    tissue_mask = ~((s < sat_thresh) & (v > val_thresh))
    tissue_mask = tissue_mask.astype(np.uint8) * 255
    
    print("Applying morphological cleanup...")
    # More aggressive morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # was 7,7
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Upscale mask if we downsampled
    if process_small:
        print("Upscaling mask to original resolution...")
        tissue_mask = cv2.resize(tissue_mask, (original_shape[1], original_shape[0]), 
                                 interpolation=cv2.INTER_LINEAR)
    
    # Apply mask to original full-resolution image
    print("Applying mask to image...")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    img_rgba = np.dstack([img_rgb, tissue_mask])  # Add alpha channel
    
    print("✓ Done!")
    return img_rgba, tissue_mask

for tiff_file in tiff_dir.rglob("*"):
    if tiff_file.suffix.lower() in (".tif", ".tiff") and "mask" not in tiff_file.stem.lower():
        new_img, mask = remove_trichrome_background(tiff_file, sat_thresh=30, val_thresh=210)
        
        if new_img is not None:
            # Show original, mask, and result side by side
            original = cv2.cvtColor(cv2.imread(str(tiff_file)), cv2.COLOR_BGR2RGB)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            axes[0].imshow(original)
            axes[0].set_title(f'Original\n{tiff_file.name}')
            axes[0].axis('off')
            
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Mask\n(white=tissue, black=background)')
            axes[1].axis('off')
            
            axes[2].imshow(new_img)
            axes[2].set_title('Background Removed')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()