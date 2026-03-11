#!/usr/bin/env python3

from pathlib import Path
from PIL import Image, ImageOps

# ---------------------------------------------------------
# Set your directory path here:
# Example: FOLDER_PATH = Path(r"C:\Users\you\Desktop\my_tiffs")
# ---------------------------------------------------------
FOLDER_PATH = Path("/home/riware/Desktop/mittal_lab/regions_for_cellpose")

SUFFIX = "_invertedgrayscale"


def process_tiff(in_path: Path) -> Path:
    """Convert a TIFF to grayscale, invert it, and save with a suffix."""
    out_name = f"{in_path.stem}{SUFFIX}{in_path.suffix.lower()}"
    out_path = in_path.with_name(out_name)

    # Skip if already processed
    if in_path.stem.endswith(SUFFIX):
        print(f"Skipping (already processed): {in_path.name}")
        return out_path
    if out_path.exists():
        print(f"Skipping (output already exists): {out_path.name}")
        return out_path

    # Process the image
    with Image.open(in_path) as im:
        # Convert to grayscale (mode 'L')
        if im.mode != "L":
            im = im.convert("L")

        # Invert
        inverted = ImageOps.invert(im)

        # Save TIFF
        inverted.save(out_path, format="TIFF", compression="tiff_deflate")

    print(f"Saved: {out_path.name}")
    return out_path


def main():
    folder = FOLDER_PATH

    if not folder.exists() or not folder.is_dir():
        print(f"Error: {folder} is not a valid folder.")
        return

    tiff_files = [p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in (".tif", ".tiff")]

    if not tiff_files:
        print("No TIFF files found in the folder.")
        return

    for tif in sorted(tiff_files):
        try:
            process_tiff(tif)
        except Exception as e:
            print(f"Failed to process {tif.name}: {e}")


if __name__ == "__main__":
    main()