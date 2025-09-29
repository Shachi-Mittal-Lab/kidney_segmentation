# Repo Tools
from fibrosis_score.single_model_pred import unet_models_pred
from model import UNet, ConvBlock, Downsample, CropAndConcat, OutputConv

# Tools
from pathlib import Path

# import inputs from inputs file
from directory_paths import (
    input_path,
    threshold,
    offset,
    axis_names,
)

input_files = [x for x in input_path.glob("*")]
print(f"input files: {input_files}")

for file_path in input_files:
    print(f"Running only U-net preds on {file_path.stem}")
    unet_models_pred(
        file_path,
        threshold,
        offset,
        axis_names,
    )
