import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def visualize_image_mask_pairs(root_folder):
    """
    Traverse a folder containing numbered subfolders with image.tif and mask.tif pairs.
    Visualize each pair.
    
    Args:
        root_folder: Path to the root folder containing numbered subfolders
    """
    root_path = Path(root_folder)
    
    # Get all subdirectories and sort them numerically
    subfolders = sorted([f for f in root_path.iterdir() if f.is_dir()], 
                       key=lambda x: int(x.name) if x.name.isdigit() else float('inf'))
    
    if not subfolders:
        print(f"No subfolders found in {root_folder}")
        return
    
    print(f"Found {len(subfolders)} folders to process\n")
    
    for folder in subfolders:
        image_path = folder / "image.tif"
        mask_path = folder / "mask.tif"
        
        # Check if both files exist
        if not image_path.exists() or not mask_path.exists():
            print(f"Skipping {folder.name}: Missing image.tif or mask.tif")
            continue
        
        try:
            # Load image and mask
            img = np.array(Image.open(image_path))
            mask = np.array(Image.open(mask_path))
            
            # Create visualization
            f, axarr = plt.subplots(1, 2, figsize=(12, 6))
            
            # Handle different image formats (grayscale vs RGB)
            if len(img.shape) == 2:  # Grayscale
                axarr[0].imshow(img, cmap='gray')
            else:  # RGB or multi-channel
                axarr[0].imshow(img)
            
            axarr[0].set_title(f"Image - Folder {folder.name}")
            
            # Display mask
            axarr[1].imshow(mask, interpolation=None)
            axarr[1].set_title(f"Mask - Folder {folder.name}")
            
            # Remove axes
            _ = [ax.axis("off") for ax in axarr]
            
            # Print info
            print(f"Folder {folder.name}:")
            print(f"  Image size: {img.shape}")
            print(f"  Mask size: {mask.shape}")
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error processing folder {folder.name}: {str(e)}")
            continue

def visualize_random_pair(root_folder):
    """
    Visualize a random image-mask pair from the dataset.
    
    Args:
        root_folder: Path to the root folder containing numbered subfolders
    """
    root_path = Path(root_folder)
    subfolders = [f for f in root_path.iterdir() if f.is_dir() 
                  and (f / "image.tif").exists() and (f / "mask.tif").exists()]
    
    if not subfolders:
        print("No valid image-mask pairs found")
        return
    
    # Select random folder
    random_folder = np.random.choice(subfolders)
    
    # Load and visualize
    img = np.array(Image.open(random_folder / "image.tif"))
    mask = np.array(Image.open(random_folder / "mask.tif"))
    
    f, axarr = plt.subplots(1, 2, figsize=(12, 6))
    
    if len(img.shape) == 2:
        axarr[0].imshow(img, cmap='gray')
    else:
        axarr[0].imshow(img)
    
    axarr[0].set_title(f"Image - Folder {random_folder.name}")
    axarr[1].imshow(mask, interpolation=None)
    axarr[1].set_title(f"Mask - Folder {random_folder.name}")
    
    _ = [ax.axis("off") for ax in axarr]
    
    print(f"Random folder: {random_folder.name}")
    print(f"Image size: {img.shape}")
    print(f"Mask size: {mask.shape}")
    
    plt.tight_layout()
    plt.show()

# Usage examples:
if __name__ == "__main__":
    # Specify your root folder path
    root_folder = "/home/riware/Desktop/mittal_lab/cortex_annotations_formatted_split/cortex_dataset1/training"
    
    # Visualize all pairs (will show them one by one)
    visualize_image_mask_pairs(root_folder)
    
    # Or visualize just one random pair
    # visualize_random_pair(root_folder)
