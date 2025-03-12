from skimage.morphology import (
    binary_erosion,
    binary_dilation,
    disk,
)

def fill_holes(inmask, filldisk, shrinkdisk):
    """
    Fill holes in mask by dilation followed by erosion.
    Uses "sequence" decomposition for computational efficiency.
    
    """
    fill_hole_kernel = disk(
        filldisk, decomposition="sequence"
    )
    shrink_kernel = disk(shrinkdisk, decomposition="sequence")
    mask_filled = binary_erosion(binary_dilation(inmask, fill_hole_kernel), shrink_kernel)
    return mask_filled

def erode(inmask, shrinkdisk): 
    """
    Erode mask.  Uses "sequence" decomposition for computational
    efficiency.
    """
    shrink_kernel = disk(shrinkdisk, decomposition="sequence")
    mask_eroded = binary_erosion(inmask, shrink_kernel)
    return mask_eroded