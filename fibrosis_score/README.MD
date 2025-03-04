# Zarr Format Information and Organization
Our data's native file format is .NDPI.  This is the Hamamatsu whole slide image format.  It is a modified pyramidal tiff that we cannot write to.  Information regarding this format can be found in the Openslide documentation here:

https://openslide.org/formats/hamamatsu/

Therefore, our pipeline converts this .NDPI file to a Zarr so that we can write our own masks to these files.  This allows for our masks to exist in the same file as the raw data, which is easiest for organization.

## Zarr File Organization
The Zarr file appears as a folder in your file viewer.  Inside the Zarr file will be two folders named "raw" and "mask".  

The "raw" folder contains the raw data - the whole slide image.  There will be four folders inside this one: s0, s1, s2, and s3.  These correspond to 40x, 20x, 10x, and 5x levels, respectively.

The "mask" folder contains the relevant masks we've generated throughout the pipeline so that we can visualize them as needed. 

Both the pyramid level folders and the mask folders will have multiple folders inside of them named by index number.  These are the chunks of these data.  These chunks are what the code uses to process any operation block-wise, so that we do not run out of memory operating on these large data files. To see the metadata around these chunks, you need to show hidden files in your file viewer.  Two files will appear: .zattrs (contains the voxel size, axis names, world units, and offset associated with that data) and .zarray (which contains chunk size, shape, data type, etc.).

# How to Visualize with Neuroglancer 
To visualize the zarr, we use the Funke Lab's tools with Neuroglancer:

https://github.com/funkelab/funlib.show.neuroglancer

This is browser-based and easy to navigate the raw data and masks using tabs. 

*insert picture here of WSI in neuroglancer* 

## Opening Raw Data and Masks

We will use the terminal to open Neuroglancer.  To visualize just the raw data, we will type:

``` 
neuroglancer -d /your/data/path.zarr/raw*
```

The raw* indicates all pyramid levels should be loaded.  Neuroglancer will lazy load these layers automatically.

To load masks in addition to the raw data, you will type: 

``` 
neuroglancer -d /your/data/path.zarr/raw* /your/data/path.zarr/mask/maskname1 
```

This will load mask with name maskname1.  You may load multiple masks as follows, ensuring your path is complete and there are spaces between each path:

``` 
neuroglancer -d /your/data/path.zarr/raw* /your/data/path.zarr/mask/maskname1 /tour/data/path.zarr/mask/maskname2
```

This will open Neuroglancer in your browser with multiple tabs visible: your raw, and any other masks you've included in your command above.  If there are tabs missing, there was a mistake in your path.  Return to your terminal, hit enter to exit, and try again.

## Visualizing Color

Your raw data will load grayscale by default.  To fix this, right click on your raw tab and an options window will appear on the right.  Navigate to the "render" tab, delete the contents of the shader window, and copy and paste this shader: 

```
void main() {
    emitRGB(
        vec3(
            toNormalized(getDataValue(0)),
            toNormalized(getDataValue(1)),
            toNormalized(getDataValue(2)))
        );
}
```

## Visualizing Masks

When the browser opens from your intitial command, all masks are visible by default, but they load as images, and therefore will not be visible.  To fix this, right click on the mask you would like to visualize and an options window will appear on the right.  Navigate to the img button on the top, click, and select "seg" in the dropdown.  This will visualize your mask in blue.

To change the color of the mask, navigate to the "render" tab, and click "fixed color". Select the color you wish to visualize that tab in. 

# Navigation
Click and drag the mouse to navigate the image.

Ctrl + scroll will zoom in and out.

Clicking "a" on your keyboard will remove the crosshairs from the visualization. 
