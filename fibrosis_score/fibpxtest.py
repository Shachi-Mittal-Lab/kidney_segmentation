from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi

from pathlib import Path
import numpy as np

# Funke Lab Tools
import daisy

s0_array = open_ds("/media/mrl/Data/pipeline_connection/ndpis/high_fib_regions_22Sep2025/threshold_3000/BR22-2096-A-1-9-TRICHROME - 2022-11-11 17.42.10_region.zarr/raw/s0")
inputfilename = Path("BR22-2096-A-1-9-TRICHROME - 2022-11-11 17.42.10_region")

fib_threshold = open_ds("/media/mrl/Data/pipeline_connection/ndpis/high_fib_regions_22Sep2025/threshold_3000/BR22-2096-A-1-9-TRICHROME - 2022-11-11 17.42.10_region.zarr/mask/finfib")
threshold_zarr_path = Path("/media/mrl/Data/pipeline_connection/ndpis/high_fib_regions_22Sep2025/threshold_3000/BR22-2096-A-1-9-TRICHROME - 2022-11-11 17.42.10_region.zarr")
fib_nothreshold= open_ds("/media/mrl/Data/pipeline_connection/ndpis/high_fib_regions_22Sep2025/threshold_none/BR22-2096-A-1-9-TRICHROME - 2022-11-11 17.42.10_region.zarr/mask/finfib")
nothreshold_zarr_path = Path("/media/mrl/Data/pipeline_connection/ndpis/high_fib_regions_22Sep2025/threshold_none/BR22-2096-A-1-9-TRICHROME - 2022-11-11 17.42.10_region.zarr")

def calculate_fibscore(
        finfib_mask: Array,
        zarr_path: Path,
        input_filename: Path,
        s0_array: Array,
):
# blockwise mask multiplications
    def fibscore_block(block: daisy.Block):
        # in data
        fib = finfib_mask[block.read_roi]

        # calc positive pixels in mask
        fibpx = np.count_nonzero(fib)

        # write to text file
        # create text file
        with open(Path(zarr_path.parent / f"{input_filename}_fibpx_test.txt"), "a") as f:
            f.writelines(f"{fibpx} \n")


    fibscore_task = daisy.Task(
        "fibscore calc",
        total_roi=s0_array.roi,
        read_roi=Roi((0, 0), (1000000, 1000000)),
        write_roi=Roi((0, 0), (1000000, 1000000)),
        read_write_conflict=False,
        num_workers=2,
        process_function=fibscore_block,
    )
    daisy.run_blockwise(tasks=[fibscore_task], multiprocessing=False)

calculate_fibscore(fib_threshold, threshold_zarr_path, inputfilename, s0_array)
calculate_fibscore(fib_nothreshold, nothreshold_zarr_path, inputfilename, s0_array)
print("done!")
