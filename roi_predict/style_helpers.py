import numpy as np
import cv2
import os
from typing import NamedTuple

# Stop_pt is set to 200 as the histograms of the images have a huge spike
# above 200 for background pixels
stop_pt = 200

# define namedtuple for storage of style information 
Image = NamedTuple('image', [('path', str), ('stain_contrast', str), ('stain_darkness', str)])

# Create histogram and determine max value within the range of 0 to stop_pt
def histmax(hist,stop_pt):
    # Convert histogram to simple list
    hist = [val[0] for val in hist]
    # Find maximum 
    return np.argmax(hist[0:stop_pt])

# Calculate the overlap of the histogram of red and blue pixel values in the 
# histogram of the image
# This is generally a good indicator of stain contrast

def histoverlap(hist1, hist2):
    tot1 = sum(hist1[:stop_pt, 0])
    tot2 = sum(hist1[:stop_pt, 0])
    tot = tot1 + tot2
    mins = [0] * (stop_pt)
    for i in range(stop_pt):
        min = np.minimum(int(hist1[i,0]), int(hist2[i,0]))
        mins[i] = min
    return sum(mins) / tot

def classify_style(img_path:str):
    # Open image
    imageGBR = cv2.imread(img_path)
    imageRGB = cv2.cvtColor(imageGBR, cv2.COLOR_BGR2RGB)

    # Create RGB histograms
    blue_color = cv2.calcHist([imageRGB], [0], None, [256], [0, 256])
    red_color = cv2.calcHist([imageRGB], [1], None, [256], [0, 256])

    # Calc overlap of red and blue histograms
    rboverlap = histoverlap(red_color, blue_color)

    # Convert to grayscale
    gray = cv2.cvtColor(imageGBR, cv2.COLOR_BGR2GRAY)

    # Grayscale histogram
    grayhist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    graymax = histmax(grayhist, stop_pt)
   
    # Calc darkness category
    if graymax < 50:
        staindarkness = "dark"
    elif graymax > 70:
        staindarkness = "light"
    else: 
        staindarkness = "medium"

    # Calc stain darkness category
    if rboverlap < 0.25:
        staincontrast = "high"
    elif rboverlap > 0.32:
        staincontrast= "low"
    else: 
        staincontrast = "medium"

    img_summary = Image(img_path, staincontrast, staindarkness)

    return img_summary
