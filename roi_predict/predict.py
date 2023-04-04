import os
from style_helpers import classify_style
from tqdm import tqdm

IMAGE_DIR = "/home/riware/Desktop/fibrosis_score/rgb_exploration/ROIs"

# Create image list
images = [
    os.path.join(IMAGE_DIR, image)
    for image in os.listdir(IMAGE_DIR)
    if image.lower().endswith((".png", ".tif", ".tiff"))
]

img_styles = list()

print("\n Classifying stains \n")

for image in tqdm(images):
    style = classify_style(image)
    img_styles.append(style)

for image in tqdm(img_styles):
    if image.stain_contrast == "low" and image.stain_darkness == "light":
        print("low light")
    elif image.stain_contrast == "low" and image.stain_darkness == "medium":
        print("low medium")
    elif image.stain_contrast == "low" and image.stain_darkness == "dark":
        print("low dark")
    elif image.stain_contrast == "medium" and image.stain_darkness == "light":
        print("medium light")
    elif image.stain_contrast == "medium" and image.stain_darkness == "medium":
        print("medium medium")
    elif image.stain_contrast == "medium" and image.stain_darkness == "dark":
        print("medium dark")
    elif image.stain_contrast == "high" and image.stain_darkness == "light":
        print("high light")
    elif image.stain_contrast == "high" and image.stain_darkness == "medium":
        print("high medium")
    elif image.stain_contrast == "high" and image.stain_darkness == "dark":
        print("high dark")
    else:
        print("Invalid style classification")



