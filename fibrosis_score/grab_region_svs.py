from pathlib import Path
import openslide
from PIL import Image
from matplotlib import pyplot as plt

# image to visualize
svs_path = Path("/home/riware/Desktop/mittal_lab/kpmp_regions/svs/daea1032-d81b-45d3-8d26-831c141d2cab_S-2010-012855_PAS_1of2.svs")
svs_folder = svs_path.parent
svs_name = svs_path.stem
identifier = "1_region"

def readsvs(svs_path: Path, level: int, location: tuple, size: tuple):
    slide = openslide.OpenSlide(svs_path)
    img = slide.read_region(location=location, level=level, size=size)
    return img

# left/right, up/down 

# region selection for glom training 
img = readsvs(svs_path, 0, (18500,14000), (3000, 3000)) # 35e623b2-c9e8-4098-85fb-1489d04fc41d_S-2006-003982_TRI_2of2.svs region 0 


plt.imshow(img)
plt.show()

img.save(svs_folder / (svs_name + f"_{identifier}" + ".tiff"))