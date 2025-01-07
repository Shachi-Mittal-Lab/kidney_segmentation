import subprocess
from pathlib import Path

# send all pyramid images of ROI to omni-seg and predict - Eric
#auto predict script path
omni_seg_path = '/media/mrl/Data/pipeline_connection/Omni_seg_pipeline_gpu/Auto_Omni_predict.py'
#run the script
subprocess.run(["python3", omni_seg_path], cwd='/media/mrl/Data/pipeline_connection/Omni_seg_pipeline_gpu', check=True)
