# Background Removal
This process utilizes both a simple threshold (distance from 255) and a VGG16 model trained on PAS and Masson's Trichrome stained kidney images to remove background tiles from a ROI in a tiff file format.

## dataloader.py
Defines dataloader which adds filepath as a saved value to the standard torchvision.datasets ImageFolder dataloader.

## patch_utils.py
Defines functions for simple threshold and patch management (stitching, mask creation, etc.).

## predict.py
Defines prediction function used for VGG16 to determine background

## remove_background.py
Executes both the simple and the VGG16 background thresholding on all ROIs in a tiff file format in the selected folder.

## requirements.txt
Use this file to install all necessary packages.  

```python
pip install -r requirements.txt
```