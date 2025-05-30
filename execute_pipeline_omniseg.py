# Repo Tools
from fibrosis_score.main_pipeline_omniseg import run_full_pipeline
from model import UNet, ConvBlock, Downsample, CropAndConcat, OutputConv

# Tools
from pathlib import Path

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

input_files = [x for x in input_path.glob("*")]
print(f"input files: {input_files}")

for file_path in input_files:
    print(f"Running pipeline on {file_path.stem}")
    run_full_pipeline(
        file_path,
        png_path,
        threshold,
        offset,
        axis_names,
        gray_cluster_centers,
        output_directory,
        omni_seg_path,
        subprocesswd,
    )
