from pathlib import Path
import shutil
import tifffile
import numpy as np
from PIL import Image
import cv2

# Settings
input_dir = Path("/home/riware/Desktop/loose_files/tubule_annotations/dt_pt_formatted/tri")  # <-- Replace with your folder path
output_dir = input_dir / "reformatted"
output_dir.mkdir(exist_ok=True)

# Resize target
target_size = (750, 750)

# Loop through all image files
for img_path in input_dir.glob("*"):
    if not img_path.is_file():
        continue

    # Output file name with .tif extension
    output_path = output_dir / (img_path.stem + ".tif")

    # Check if this is a mask
    is_mask = "mask" in img_path.stem.lower()

    # If already a tif, just read and resize; otherwise convert first
    if img_path.suffix.lower() in [".tif", ".tiff"]:
        image = tifffile.imread(img_path)
    else:
        image = np.array(Image.open(img_path))

    # Resize
    if is_mask:
        resized = cv2.resize(image.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
        resized = resized.astype(np.uint8)  # Ensure binary mask stays uint8
    else:
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Save as tif
    tifffile.imwrite(output_path, resized)

print("✅ Conversion and resizing complete!")
