# Repo Tools
from main_pipeline import run_full_pipeline

# Tools
from pathlib import Path
import sys

sys.path.append("/media/mrl/Data/pipeline_connection/kidney_segmentation")

# import inputs from inputs file
from directory_paths import (
    input_path,
    png_path,
    threshold,
    offset,
    axis_names,
    gray_cluster_centers,
    output_directory,
    omni_seg_path,
    subprocesswd,
)

input_folder = Path("/home/riware/Desktop/loose_files/ndpis")

input_files = [x for x in input_folder.glob("**/*") if x.is_file()]

for file in input_files:
    print(f"Running pipeline on {file.stem}")
    run_full_pipeline(
        input_path,
        png_path,
        threshold,
        offset,
        axis_names,
        gray_cluster_centers,
        output_directory,
        omni_seg_path,
        subprocesswd,
    )
