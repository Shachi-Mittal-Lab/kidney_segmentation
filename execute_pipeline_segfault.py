# Repo Tools
from fibrosis_score.main_pipeline import run_full_pipeline
from model import UNet, ConvBlock, Downsample, CropAndConcat, OutputConv

# Tools
from pathlib import Path
from dask.distributed import LocalCluster, Client
import multiprocessing

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

if __name__ == "__main__": 

    input_files = [x for x in input_path.glob("*")]
    print(f"input files: {input_files}")
    """
    # Generate Local Cluster and Client for dask memory management
    cluster = LocalCluster(processes=False)
    client = Client(cluster)
    print("Dashboard link:")
    print(f"{client.dashboard_link}") # This will print the URL
    """
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

    # Close the dask client and shutdown the cluster
    """
    client.close()
    cluster.close()
    """
