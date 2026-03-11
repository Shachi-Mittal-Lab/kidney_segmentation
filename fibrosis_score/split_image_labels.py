import os
from pathlib import Path
import numpy as np
import PIL
from PIL import Image
from typing import Tuple, List

PIL.Image.MAX_IMAGE_PIXELS = 1000000000

def split_image_into_patches_with_overlap(image: np.ndarray, patch_size: int = 3000, overlap: float = 0.5) -> List[tuple]:
    """
    Split an image into patches with overlap. Discards edge patches that don't meet the size requirement.
    
    Args:
        image: numpy array of the image
        patch_size: size of square patches (default 3000)
        overlap: overlap ratio (default 0.5 for 50% overlap)
        
    Returns:
        List of tuples (patch, row_idx, col_idx, patch_idx)
    """
    height, width = image.shape[:2]
    
    # Calculate stride (step size between patches)
    stride = int(patch_size * (1 - overlap))
    
    patches = []
    patch_idx = 0
    
    row = 0
    row_idx = 0
    while row + patch_size <= height:
        col = 0
        col_idx = 0
        while col + patch_size <= width:
            patch = image[row:row + patch_size, col:col + patch_size]
            patches.append((patch, row_idx, col_idx, patch_idx))
            patch_idx += 1
            col += stride
            col_idx += 1
        row += stride
        row_idx += 1
    
    return patches

def has_sufficient_positive_pixels(mask: np.ndarray, threshold: float = 0.10) -> bool:
    """
    Check if mask has at least threshold percentage of positive (non-zero) pixels.
    
    Args:
        mask: numpy array of the mask
        threshold: minimum percentage of positive pixels required (default 0.10 for 10%)
        
    Returns:
        True if mask contains at least threshold percentage of non-zero pixels, False otherwise
    """
    total_pixels = mask.size
    positive_pixels = np.count_nonzero(mask)
    positive_ratio = positive_pixels / total_pixels
    return positive_ratio >= threshold

def process_image_pair(image_path: Path, mask_path: Path, output_folder: Path, 
                       folder_name: str, patch_size: int = 3000, overlap: float = 0.5,
                       positive_threshold: float = 0.10) -> None:
    """
    Process a single image-mask pair by splitting into overlapping patches.
    Only saves patches where at least positive_threshold of mask pixels are positive.
    
    Args:
        image_path: Path to the original image
        mask_path: Path to the corresponding mask
        output_folder: Root output folder
        folder_name: Name of the subfolder (to maintain structure)
        patch_size: size of square patches (default 3000)
        overlap: overlap ratio (default 0.5 for 50%)
        positive_threshold: minimum percentage of positive pixels (default 0.10 for 10%)
    """
    # Read images
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path))
    
    # Verify dimensions match
    if image.shape[:2] != mask.shape[:2]:
        print(f"Warning: Dimension mismatch for {image_path.name}")
        return
    
    # Split into patches
    image_patches = split_image_into_patches_with_overlap(image, patch_size, overlap)
    mask_patches = split_image_into_patches_with_overlap(mask, patch_size, overlap)
    
    if len(image_patches) == 0:
        print(f"Warning: {image_path.name} is smaller than {patch_size}x{patch_size}, no patches created")
        return
    
    # Extract identification string from filename
    # Format: im_{identification string}_5x.tif or.tiff
    filename = image_path.stem  # Remove extension
    id_string = filename.replace('im_', '').replace('_5x', '')
    
    # Create output subfolder
    output_subfolder = output_folder / folder_name
    output_subfolder.mkdir(parents=True, exist_ok=True)
    
    # Save each patch pair (only if mask has sufficient positive pixels)
    saved_count = 0
    skipped_count = 0
    
    for (img_patch, row, col, idx), (mask_patch, _, _, _) in zip(image_patches, mask_patches):
        # Check if mask has sufficient positive pixels
        if not has_sufficient_positive_pixels(mask_patch, positive_threshold):
            skipped_count += 1
            continue
        
        # Create new filenames with patch index (maintain original extensions)
        new_image_name = f"im_{id_string}_5x_patch{idx}.tif"
        new_mask_name = f"im_{id_string}_5x_mask_patch{idx}.tiff"
        
        # Save patches
        Image.fromarray(img_patch).save(output_subfolder / new_image_name)
        Image.fromarray(mask_patch).save(output_subfolder / new_mask_name)
        saved_count += 1
    
    total_patches = len(image_patches)
    print(f"Processed: {image_path.name} -> {saved_count}/{total_patches} patches saved, {skipped_count} patches skipped (<{positive_threshold*100}% positive pixels)")

def process_folders(input_root: str, output_root: str, folder_names: List[str], 
                    patch_size: int = 3000, overlap: float = 0.5, 
                    positive_threshold: float = 0.10) -> None:
    """
    Process all image-mask pairs in multiple folders.
    
    Args:
        input_root: Root directory containing the input folders
        output_root: Root directory for output folders
        folder_names: List of folder names to process
        patch_size: size of square patches (default 3000)
        overlap: overlap ratio (default 0.5 for 50%)
        positive_threshold: minimum percentage of positive pixels (default 0.10 for 10%)
    """
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    # Create output root if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_patches_saved = 0
    total_patches_skipped = 0
    
    for folder_name in folder_names:
        folder_path = input_path / folder_name
        
        if not folder_path.exists():
            print(f"Warning: Folder {folder_path} does not exist, skipping...")
            continue
        
        print(f"\nProcessing folder: {folder_name}")
        print(f"Settings: patch_size={patch_size}px, overlap={overlap*100}%, positive_threshold={positive_threshold*100}%")
        
        # Find all image files (not masks) - images end with _5x.tif or _5x.tiff
        image_files = sorted([f for f in folder_path.glob("im_*_5x.tif") 
                             if not f.stem.endswith("_mask")])
        
        for image_path in image_files:
            # Construct corresponding mask path
            # Mask has format: im_{id}_5x_mask.tiff (note:.tiff not.tif)
            mask_name = image_path.stem + "_mask.tiff"
            mask_path = folder_path / mask_name
            
            if not mask_path.exists():
                print(f"Warning: Mask not found for {image_path.name}, skipping...")
                continue
            
            # Process the pair
            process_image_pair(image_path, mask_path, output_path, folder_name, 
                             patch_size, overlap, positive_threshold)
            total_processed += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total image pairs processed: {total_processed}")
    print(f"Settings used:")
    print(f"  - Patch size: {patch_size}x{patch_size} pixels")
    print(f"  - Overlap: {overlap*100}%")
    print(f"  - Positive pixel threshold: {positive_threshold*100}%")
    print(f"Output location: {output_path.absolute()}")

# Example usage
if __name__ == "__main__":
    # Configure these paths
    INPUT_ROOT = "/home/riware/Desktop/mittal_lab/cortex_annotations_formatted"  # Parent folder containing the 4 folders
    OUTPUT_ROOT = "/home/riware/Desktop/mittal_lab/cortex_annotations_formatted_split_3"  # Where split images will be saved
    FOLDER_NAMES = ["he", "pas", "sil", "tri"]  # Your 4 folder names
    PATCH_SIZE = 3000  # Size of square patches
    OVERLAP = 0.5  # 50% overlap
    POSITIVE_THRESHOLD = 0.15  # 10% of pixels must be positive
    
    # Run the processing
    process_folders(INPUT_ROOT, OUTPUT_ROOT, FOLDER_NAMES, PATCH_SIZE, OVERLAP, POSITIVE_THRESHOLD)
