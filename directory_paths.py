from pathlib import Path

# inputs #

# file inputs
ndpi_path = Path("/media/mrl/Data/pipeline_connection/ndpis/predict/BR21-2051-A-1-9-TRI  - 2022-08-10 16.30.12.ndpi") #Eric edit
zarr_path = Path("/media/mrl/Data/pipeline_connection/ndpis/predict/21-2002_region.zarr") #Eric edit

#Eric edit:no pngs before it was created 
png_path = Path("/media/mrl/Data/pipeline_connection/kidney_segmentation/omni-seg/input")
txt_path = Path("/media/mrl/Data/pipeline_connection/kidney_segmentation/fibscores.txt")

# background threshold
threshold = 50
# slide offset (zero)
offset = (0, 0)
# axis names
axis_names = [
    "x",
    "y",
    "c^",
]

# cortex vs medulla model info
cortmed_weights_path = Path(
    "/media/mrl/Data/pipeline_connection/cortexvsmedullavsothers/model5_epoch8.pt" #Eric edit
)
cortmed_starting_model_path = Path(
    "/media/mrl/Data/pipeline_connection/cortexvsmedullavsothers/starting_model.pt" #Eric edit
)

# kmeans clustering centers
gray_cluster_centers = Path(
    "/media/mrl/Data/pipeline_connection/kidney_segmentation/fibrosis_score/average_centers_5.txt" #Eric edit
)

# omni-seg output path
output_directory = f'/media/mrl/Data/pipeline_connection/kidney_segmentation/omni-seg/outputs'
#output_path = Path(f"{output_directory}/{os.listdir(output_directory)[0]}") #Eric edit

#Eric 12/27/2024: path for the patch to be saved to for omni-seg's input
#40x
inputdir = f'/media/mrl/Data/pipeline_connection/kidney_segmentation/omni-seg/input'
 # send all pyramid images of ROI to omni-seg and predict - Eric
#auto predict script path
omni_seg_path = '/media/mrl/Data/pipeline_connection/kidney_segmentation/omni-seg/Auto_Omni_predict.py'
#run the script
subprocesswd='/media/mrl/Data/pipeline_connection/kidney_segmentation/omni-seg'



##________________________________________________________________________________##
#for the auto_omni_seg process
subdirclean1 = '/media/mrl/Data/pipeline_connection/kidney_segmentation/omni-seg/outputs/segmentation_merge'
subdirclean2 = '/media/mrl/Data/pipeline_connection/kidney_segmentation/omni-seg/outputs/clinical_patches'
subdirclean3 = '/media/mrl/Data/pipeline_connection/kidney_segmentation/omni-seg/outputs/final_merge'

subprocess_step1 = '/media/mrl/Data/pipeline_connection/kidney_segmentation/omni-seg/1024_Step1_GridPatch_overlap_padding.py'
subprocess_step1_5 = '/media/mrl/Data/pipeline_connection/kidney_segmentation/omni-seg//1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py'
subprocess_step2 = '/media/mrl/Data/pipeline_connection/kidney_segmentation/omni-seg/Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py'
subprocess_step3 = '/media/mrl/Data/pipeline_connection/kidney_segmentation/omni-seg/step3.py'
subprocess_step4 = '/media/mrl/Data/pipeline_connection/kidney_segmentation/omni-seg/step4.py'

result_list = '/media/mrl/Data/pipeline_connection/kidney_segmentation/omni-seg/outputs/final_merge'
result_output_list = '/media/mrl/Data/pipeline_connection/kidney_segmentation/omni-seg/outputs'
result_input_list = '/media/mrl/Data/pipeline_connection/kidney_segmentation/omni-seg/input'

