import os
import numpy as np
import cv2
from PIL import Image
from skimage import io
import re


# Calculate percent background in a patch to define removal threshold
def percent_background(roi_array, color_delta):
    roi_width = roi_array.shape[0]
    roi_height = roi_array.shape[1]
    white = np.array([255, 255, 255])
    threshold = np.subtract(white, color_delta)
    sub_threshold = np.subtract(roi_array, threshold)
    sub_threshold = np.where(sub_threshold < 0, 0, sub_threshold)
    num_background_pix = np.count_nonzero(np.all(sub_threshold, axis=2))
    total_pixels = roi_width * roi_height
    percent_background = num_background_pix / total_pixels * 100
    return percent_background


def extract_tiles(
    case, image_array, IMG_DIRECTORY, x_tile_size, y_tile_size, threshold, color_delta
):
    # Calc number of tiles to create.  Cuts off last row/column of image if ROI not divisible by tile size
    roi_size_x = image_array.shape[1]
    roi_size_y = image_array.shape[0]
    x_tile_num = int(np.floor(roi_size_x / x_tile_size))
    y_tile_num = int(np.floor(roi_size_y / y_tile_size))
    # Left to right and top to bottom
    for iy in range(y_tile_num):
        for ix in range(x_tile_num):
            # Coordinates of the upper left corner of each image
            start_x = int(ix * x_tile_size)
            start_y = int(iy * y_tile_size)
            end_x = start_x + x_tile_size
            end_y = start_y + y_tile_size

            # Grab patch and only make image if the threshold for black pixels is satisfied
            cur_tile = image_array[
                start_y:end_y,
                start_x:end_x,
            ]

            per_background = percent_background(cur_tile, color_delta)
            # py.imshow(cur_tile)
            # py.show()
            if per_background < threshold:
                patch_savename = f"{case}_{iy}-{ix}.png"
                io.imsave(os.path.join(IMG_DIRECTORY, patch_savename), cur_tile)
            else:
                print(f"{case}_{iy}-{ix} has too much background: {per_background}")
    return x_tile_num, y_tile_num


def concat_vh(list_2d):
    # return final image
    return cv2.vconcat([cv2.hconcat(list_h) for list_h in list_2d])


def stitch_img(
    x_tile_num,
    y_tile_num,
    x_tile_size,
    y_tile_size,
    classes,
    img_files,
    IMG_DIRECTORY,
    ROI_PRED_DIRECTORY,
    predictions,
):
    locations = []
    image_locations = []

    for i in range(y_tile_num):
        placeholder = ["_" for j in range(x_tile_num)]
        locations.append(placeholder)

    for image in img_files:
        img_location = os.path.basename(image)
        img_location = re.search("tiff_(.*?).png", img_location)
        assert img_location is not None
        img_location = img_location.group(1)
        x_y = str.split(img_location, "-")
        img_location = [int(x_y[0]), int(x_y[1])]
        image_locations.append(img_location)

    img = Image.new("RGB", (x_tile_size, y_tile_size), (0, 0, 0))
    black_img_path = os.path.join(IMG_DIRECTORY, "black.png")
    img.save(black_img_path, "PNG")

    for row in range(len(locations)):
        for column in range(len(locations[row])):
            if [row, column] in image_locations:
                index = image_locations.index([row, column])
                locations[row][column] = img_files[index]
            else:
                locations[row][column] = black_img_path

    # Make stitched ROI tiff
    imgs = [[cv2.imread(j) for j in locations[i]] for i, path in enumerate(locations)]
    img_tile = concat_vh(imgs)
    img_name = os.path.join(ROI_PRED_DIRECTORY, "stitched.tiff")
    cv2.imwrite(img_name, img_tile)
    print(f"Saved stitched tiff as {img_name}")

    # Make mask files
    MASK_DIRECTORY = os.path.join(IMG_DIRECTORY, "mask")
    os.mkdir(MASK_DIRECTORY)

    mask = []
    for i in range(y_tile_num):
        placeholder = ["_" for j in range(x_tile_num)]
        mask.append(placeholder)

    # Create black image for tiles that were not predicted on
    not_pred = np.full((x_tile_size, y_tile_size, 3), [0, 0, 0])
    not_pred_path = os.path.join(MASK_DIRECTORY, "mask_unpred.png")
    cv2.imwrite(not_pred_path, not_pred)

    # Create napari labels file
    for i, img in enumerate(img_files):
        category = predictions[i] + 1
        location = re.search("tiff_(.*?).png", img)
        assert location is not None
        location = location.group(1)
        mask_img = np.full((x_tile_size, y_tile_size), category)
        mask_path = os.path.join(MASK_DIRECTORY, f"mask_{location}.png")
        cv2.imwrite(mask_path, mask_img)
        x_y = str.split(location, "-")
        mask[int(x_y[0])][int(x_y[1])] = mask_path
    for i, row in enumerate(mask):
        for j, label in enumerate(row):
            if label == "_":
                mask[i][j] = not_pred_path
            else:
                pass

    masks = [
        [cv2.imread(j, cv2.IMREAD_GRAYSCALE) for j in mask[i]]
        for i, path in enumerate(mask)
    ]
    mask_tile = concat_vh(masks)
    mask_name = os.path.join(ROI_PRED_DIRECTORY, "mask_finished_napari.tiff")
    cv2.imwrite(mask_name, mask_tile)
    print(f"Saved napari mask file as {mask_name}")

    # Create tiff labels file

    # color options
    colors = {
        "purple": [82, 22, 180],
        "magenta": [255, 0, 255],
        "yellow": [255, 211, 25],
        "cyan": [60, 214, 230],
        "blue": [21, 52, 80],
        "brown": [81, 62, 62],
        "red": [237, 0, 38],
        "maroon": [111, 0, 61],
        "orange": [255, 97, 25],
        "white": [255, 255, 255],
    }

    color_names = [color for color in colors.keys()]
    color_legend = [
        f"{category}, {color}" for (category, color) in zip(classes, colors)
    ]

    with open(os.path.join(ROI_PRED_DIRECTORY, "color_legend.txt"), "w") as f:
        f.writelines("\n".join(color_legend))

    # Create rgb labels file
    for i, img in enumerate(img_files):
        category = predictions[i]
        location = re.search("tiff_(.*?).png", img)
        assert location is not None
        location = location.group(1)
        mask_img = np.full(
            (x_tile_size, y_tile_size, 3), np.float32(colors[color_names[category]])  # type: ignore
        )
        mask_path = os.path.join(MASK_DIRECTORY, f"mask_{location}.png")
        cv2.imwrite(mask_path, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))
        x_y = str.split(location, "-")
        mask[int(x_y[0])][int(x_y[1])] = mask_path
    for i, row in enumerate(mask):
        for j, label in enumerate(row):
            if label == "_":
                mask[i][j] = not_pred_path
            else:
                pass

    masks = [[cv2.imread(j) for j in mask[i]] for i, path in enumerate(mask)]
    mask_tile = concat_vh(masks)
    mask_name = os.path.join(ROI_PRED_DIRECTORY, "mask_finished.tiff")
    cv2.imwrite(mask_name, mask_tile)
    print(f"Saved tiff mask as {mask_name}")
