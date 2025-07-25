import os
import numpy as np
import tifffile as tiff

def binarize_tif_images(input_folder, output_folder=None):
    if output_folder is None:
        output_folder = input_folder

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load image
            img = tiff.imread(input_path)

            # Binarize image
            binary_img = (img != 0).astype(np.uint8)

            # Save image
            tiff.imwrite(output_path, binary_img)

            print(f"Binarized and saved: {output_path}")

# Example usage
input_folder = '/mnt/Data/kidney/rachel_training_data_10x/glom/all_glom_annotations_resized/tri/masks'  # Replace with your actual path
output_folder = '/mnt/Data/kidney/rachel_training_data_10x/glom/all_glom_annotations_resized/tri'  # Optional; can be same as input

binarize_tif_images(input_folder, output_folder)