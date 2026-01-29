from fibrosis_score.zarr_utils import ndpi_to_zarr_padding
from pathlib import Path
from matplotlib import pyplot as plt
import openslide


## inputs ##
# images
ndpi_path = Path("/home/riware/Desktop/mittal_lab/padded_images/BR21-2023-B-A-1-9-TRICHROME - 2022-11-09 14.48.18.ndpi")
zarr_path = Path("/home/riware/Desktop/mittal_lab/padded_images/BR21-2023-B-A-1-9-TRICHROME - 2022-11-09 14.48.18_padded.zarr")
# slide offset (zero)
offset = (0, 0)
padding = (5000, 0, 0)
# axis names
axis_names = [
    "x",
    "y",
    "c^",
]
#############
ndpi_to_zarr_padding(ndpi_path, zarr_path, offset, axis_names, padding)
quit()


# grab resolution from image 
def openndpi(ndpi_path):
    slide = openslide.OpenSlide(str(ndpi_path)) #Eric edit
    x_res = float(slide.properties["tiff.XResolution"])
    y_res = float(slide.properties["tiff.YResolution"])
    units = slide.properties["tiff.ResolutionUnit"]
    return x_res, y_res, units

if __name__ == "__main__": 
    # open highest pyramid level
    x_res, y_res, units = openndpi(ndpi_path)

    # read region & convert to RGB from RGBA and np to dask
    slide = openslide.OpenSlide(str(ndpi_path))
    max_width, max_height = slide.dimensions
    print(f"width: {max_width}, height: {max_height}")

    # preview slide 
    preview_level = 4
    max_prev_width, max_prev_height = slide.level_dimensions[preview_level]
    preview_region = slide.read_region(location=(0,0), level=preview_level, size=(max_prev_width, max_prev_height))

    plt.imshow(preview_region)
    plt.show()
    
