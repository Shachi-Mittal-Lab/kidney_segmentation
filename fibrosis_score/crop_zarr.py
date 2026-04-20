"""
zarr_crop_tool.py

Iterates through custom.zarr files in a folder, displays a GUI for
selecting a crop region on the PREVIEW_DS dataset, then saves the
cropped zarr with the same raw/s0 through raw/s4 pyramid structure
and funlib metadata as the original.

Configure the variables in the USER INPUTS section below, then run:
    python zarr_crop_tool.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import zarr
import daisy

from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds, prepare_ds

# ===========================================================================
# USER INPUTS — edit these before running
# ===========================================================================

INPUT_DIR   = Path("/media/riware/Articuno/HM_cases_to_crop_10Apr2026")
OUTPUT_DIR  = Path("/media/riware/Articuno/HM_cases_cropped_10Apr2026")
PREVIEW_DS  = "raw/s4"
DATASETS    = ["raw/s0", "raw/s1", "raw/s2", "raw/s3", "raw/s4"]
NUM_WORKERS = 2
BLOCK_SHAPE = Coordinate(4096, 4096)        # block size in pixels
PADDING     = Coordinate(5000, 5000)           # padding in pixels at s0 resolution
                                            # added to bottom and right of crop,
                                            # matching svs_to_zarr_padding


def find_zarr_files(input_dir: Path) -> list[Path]:
    return sorted([x for x in input_dir.glob("*") if x.suffix == ".zarr"])


def get_display_image(array) -> np.ndarray:
    """Load the preview array directly — format is always (Y, X, 3) uint8."""
    return np.array(array.data)


def select_crop_region(zarr_path: Path, display_img: np.ndarray, array, s0_array) -> Roi | None:
    """
    Display the image in a matplotlib GUI and let the user click-drag to
    select a crop region. Pixel selection is scaled to s0 resolution to
    ensure consistent cropping across all pyramid levels.
    Returns a world-space Roi or None if skipped.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title(f"Select crop region — {zarr_path.name}")
    ax.set_title("Click and drag to select crop region.\nPress ENTER to confirm, ESC to skip.", fontsize=10)
    ax.imshow(display_img, origin="upper")

    state = {"x0": None, "y0": None, "x1": None, "y1": None,
             "pressing": False, "confirmed": False, "rect": None}

    def on_press(event):
        if event.inaxes != ax or event.button != 1:
            return
        if state["rect"] is not None:
            state["rect"].remove()
            state["rect"] = None
        state.update(x0=event.xdata, y0=event.ydata,
                     x1=event.xdata, y1=event.ydata, pressing=True)

    def on_motion(event):
        if not state["pressing"] or event.inaxes != ax:
            return
        state["x1"], state["y1"] = event.xdata, event.ydata
        if state["rect"] is not None:
            state["rect"].remove()
        x0, y0, x1, y1 = state["x0"], state["y0"], state["x1"], state["y1"]
        state["rect"] = ax.add_patch(patches.Rectangle(
            (min(x0, x1), min(y0, y1)), abs(x1 - x0), abs(y1 - y0),
            linewidth=2, edgecolor="red", facecolor="red", alpha=0.2
        ))
        ax.set_title(
            f"Selected: x=[{int(min(x0,x1))}, {int(max(x0,x1))}]  "
            f"y=[{int(min(y0,y1))}, {int(max(y0,y1))}]\n"
            "Press ENTER to confirm, ESC to skip.", fontsize=10,
        )
        fig.canvas.draw_idle()

    def on_release(event):
        if event.button != 1:
            return
        state["pressing"] = False
        if event.inaxes == ax:
            state["x1"], state["y1"] = event.xdata, event.ydata

    def on_key(event):
        if event.key == "enter":
            if (state["x0"] is not None
                    and abs(state["x1"] - state["x0"]) > 5
                    and abs(state["y1"] - state["y0"]) > 5):
                state["confirmed"] = True
                plt.close(fig)
            else:
                ax.set_title("No valid selection — click and drag first.\n"
                             "Press ENTER to confirm, ESC to skip.", fontsize=10)
                fig.canvas.draw_idle()
        elif event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event",   on_press)
    fig.canvas.mpl_connect("motion_notify_event",  on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event",      on_key)

    plt.tight_layout()
    plt.show()

    if not state["confirmed"]:
        return None

    x0_px = int(min(state["x0"], state["x1"]))
    y0_px = int(min(state["y0"], state["y1"]))
    x1_px = int(max(state["x0"], state["x1"]))
    y1_px = int(max(state["y0"], state["y1"]))

    print(f"  Pixel selection on preview: x=[{x0_px}, {x1_px}]  y=[{y0_px}, {y1_px}]")

    # Scale pixel selection from preview resolution to s0 resolution,
    # then build world-space Roi from s0 voxel size so the crop aligns
    # consistently on every pyramid level's voxel grid
    preview_to_s0 = array.voxel_size / s0_array.voxel_size
    x0_s0 = int(x0_px * preview_to_s0[1])
    y0_s0 = int(y0_px * preview_to_s0[0])
    x1_s0 = int(x1_px * preview_to_s0[1])
    y1_s0 = int(y1_px * preview_to_s0[0])

    print(f"  Pixel selection at s0     : x=[{x0_s0}, {x1_s0}]  y=[{y0_s0}, {y1_s0}]")

    ndim = len(s0_array.voxel_size)
    roi_offset = Coordinate([
        y0_s0 * s0_array.voxel_size[ndim - 2] if dim == ndim - 2 else
        x0_s0 * s0_array.voxel_size[ndim - 1] if dim == ndim - 1 else 0
        for dim in range(ndim)
    ])
    roi_shape = Coordinate([
        (y1_s0 - y0_s0) * s0_array.voxel_size[ndim - 2] if dim == ndim - 2 else
        (x1_s0 - x0_s0) * s0_array.voxel_size[ndim - 1] if dim == ndim - 1 else
        s0_array.shape[dim] * s0_array.voxel_size[dim]
        for dim in range(ndim)
    ])

    crop_roi = Roi(roi_offset, roi_shape).intersect(s0_array.roi)
    print(f"  Crop Roi (world space)    : {crop_roi}")
    return crop_roi

def crop_and_save_dataset(
    ds_path: str,
    zarr_path: Path,
    output_zarr_path: Path,
    crop_roi: Roi,
    s0_voxel_size: Coordinate,
) -> bool:
    """
    Crop one dataset blockwise using daisy and save it to the output zarr.
    Offset is set to zero. PADDING pixels (at s0 resolution) are added to
    the bottom and right of the crop as white (255) fill, matching
    svs_to_zarr_padding.
    """
    try:
        src = open_ds(zarr_path / ds_path)
    except Exception as e:
        print(f"  WARNING: could not open '{ds_path}': {e}")
        return False

    dataset_roi = crop_roi.intersect(src.roi)
    if dataset_roi.empty:
        print(f"  SKIP '{ds_path}': crop Roi does not intersect dataset Roi.")
        return False

    n_spatial        = len(dataset_roi.shape)
    crop_voxel_shape = dataset_roi.shape / src.voxel_size
    channel_shape    = src.data.shape[n_spatial:]

    # Scale s0 pixel padding to this pyramid level's pixel space
    # e.g. s0 padding of 5000px at 263nm → s4 padding of 5000/16 = 312px at 4208nm
    downsample_factor = src.voxel_size / s0_voxel_size
    padding_voxels    = Coordinate([
        int(PADDING[dim] / downsample_factor[dim]) for dim in range(n_spatial)
    ])

    # full container = crop + padding in voxels, plus channel dim
    full_shape = tuple(crop_voxel_shape + padding_voxels) + tuple(channel_shape)

    print(f"\n  Cropping '{ds_path}'  {src.data.shape} → {full_shape}")
    print(f"    roi            : {dataset_roi}")
    print(f"    voxel_size     : {src.voxel_size}")
    print(f"    downsample     : {downsample_factor}")
    print(f"    padding_voxels : {padding_voxels}")

    src_zarr   = zarr.open(str(zarr_path / ds_path), mode="r")
    out_chunks = tuple(min(src_zarr.chunks[i], full_shape[i]) for i in range(len(full_shape)))

    dst = prepare_ds(
        output_zarr_path / ds_path,
        shape=full_shape,
        offset=Coordinate([0] * n_spatial),
        voxel_size=src.voxel_size,
        axis_names=src.axis_names,
        units=src.units,
        mode="w",
        dtype=np.uint8,
        fill_value=255,
        chunk_shape=out_chunks,
    )

    # daisy writes only into the crop region — padding stays white from fill_value
    write_roi  = Roi(Coordinate([0] * n_spatial), dataset_roi.shape)
    block_size = Coordinate([
        min(b, w) for b, w in zip(BLOCK_SHAPE * src.voxel_size, dataset_roi.shape)
    ])
    block_roi  = Roi(Coordinate([0] * n_spatial), block_size)

    def copy_block(block: daisy.Block):
        src_block_roi = daisy.Roi(
            block.write_roi.offset + dataset_roi.offset,
            block.write_roi.shape,
        )
        dst[block.write_roi] = src[src_block_roi]

    daisy.run_blockwise(
        tasks=[daisy.Task(
            f"crop_{ds_path.replace('/', '_')}",
            total_roi=write_roi,
            read_roi=block_roi,
            write_roi=block_roi,
            read_write_conflict=False,
            num_workers=NUM_WORKERS,
            process_function=copy_block,
        )],
        multiprocessing=False,
    )

    return True

def main():
    zarr_files = find_zarr_files(INPUT_DIR)
    if not zarr_files:
        print(f"No.zarr files found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Found {len(zarr_files)} zarr file(s):")
    for z in zarr_files:
        print(f"  {z.name}")

    for zarr_path in zarr_files:
        print(f"\n{'='*60}\nProcessing: {zarr_path.name}")

        try:
            preview_array = open_ds(zarr_path / PREVIEW_DS)
            s0_array      = open_ds(zarr_path / "raw/s0")
        except Exception as e:
            print(f"  ERROR opening datasets: {e}  Skipping.\n")
            continue

        print(f"  Preview — Shape: {preview_array.shape}  Voxel size: {preview_array.voxel_size}")
        print(f"  s0      — Shape: {s0_array.shape}       Voxel size: {s0_array.voxel_size}")

        crop_roi = select_crop_region(
            zarr_path, get_display_image(preview_array), preview_array, s0_array
        )

        if crop_roi is None:
            print("  Skipped (no selection confirmed).\n")
            continue

        output_zarr_path = OUTPUT_DIR / (zarr_path.stem + "_cropped.zarr")
        print(f"  Writing to: {output_zarr_path}")

        results = [crop_and_save_dataset(ds, zarr_path, output_zarr_path, crop_roi, s0_array.voxel_size) for ds in DATASETS]

        print(f"\n  Done: {sum(results)}/{len(DATASETS)} datasets cropped → {output_zarr_path}\n")

    print("All files processed.")


if __name__ == "__main__":
    main()