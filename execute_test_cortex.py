# Repo Tools
from fibrosis_score.main_pipeline_test_cortex import run_full_pipeline
from model import UNet, ConvBlock, Downsample, CropAndConcat, OutputConv

# Tools
from pathlib import Path

# import inputs from inputs file
from directory_paths import (
    input_path,
    offset,
    axis_names,
)

input_files = [x for x in input_path.glob("*")]
print(f"input files: {input_files}")

for file_path in input_files:
    print(f"Running pipeline on {file_path.stem}")
    run_full_pipeline(
        file_path,
        offset,
        axis_names,
    )
