import daisy
import numpy as np
from funlib.persistence import Array,open_ds
from funlib.geometry import Coordinate, Roi

import zarr
in_zarr = zarr.open_array("in_zarr.zarr", shape=(100,100))
in_zarr[:] = np.random.random((100,100))
out_zarr = zarr.open_array("out_zarr.zarr", shape=(100,100))

in_array = open_ds("in_zarr.zarr")
out_array = open_ds("out_zarr.zarr", "r+")

def process_block(block: daisy.Block):
    in_data = in_array[block.read_roi]
    out_array[block.write_roi] = in_data.max()
    print(in_data.max())

task = daisy.Task(
    "blockwise_test",
    total_roi=in_array.roi,
    read_roi=Roi((0,0),(10,10)),
    write_roi=Roi((1,1),(8,8)),
    read_write_conflict=False,
    num_workers=2,
    process_function=process_block
)

daisy.run_blockwise(tasks=[task], multiprocessing=True)

print(out_array[:].max())