import numpy as np
import os
import tifffile as tiff
from tqdm import tqdm

INPUT_DIR = "/home/riware/Desktop/mittal_lab/all_cortex_annotations_30Mar2026/to_update/fix_annotations"

# Find all base images by their naming convention
images = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".tif", ".tiff"))
    and not f.endswith("_mask.tiff")
    and not f.endswith("_tissuemask.tif")
    and not f.endswith("_corticaltissuemask.tiff")
]

for img_file in tqdm(images):
    base = os.path.splitext(img_file)[0]  # e.g. im_{ID}_5x

    cortex_mask_path = os.path.join(INPUT_DIR, f"{base}_mask.tiff")
    tissue_mask_path = os.path.join(INPUT_DIR, f"{base}_tissuemask.tif")
    output_path      = os.path.join(INPUT_DIR, f"{base}_corticaltissuemask.tiff")

    # Check both masks exist
    if not os.path.exists(cortex_mask_path):
        print(f"Missing cortex mask for {img_file}, skipping.")
        continue
    if not os.path.exists(tissue_mask_path):
        print(f"Missing tissue mask for {img_file}, skipping.")
        continue

    cortex_mask = tiff.imread(cortex_mask_path)
    tissue_mask = tiff.imread(tissue_mask_path)

    if cortex_mask.shape != tissue_mask.shape:
        print(f"Shape mismatch for {img_file}: {cortex_mask.shape} vs {tissue_mask.shape}, skipping.")
        continue

    # Binarize both masks defensively (in case values aren't strictly 0/1)
    cortex_binary = (cortex_mask > 0).astype(np.uint8)
    tissue_binary = (tissue_mask > 0).astype(np.uint8)

    # Multiply to keep only pixels positive in both masks
    combined = cortex_binary * tissue_binary

    tiff.imwrite(output_path, combined, photometric="minisblack")