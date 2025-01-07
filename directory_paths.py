from pathlib import Path

# inputs #

# file inputs
ndpi_path = Path("/media/mrl/Data/pipeline_connection/sample_data/BR21-2041-B-A-1-9-TRICHROME - 2022-11-09 16.31.43.ndpi") #Eric edit

zarr_path = Path("/media/mrl/Data/pipeline_connection/21-2041.zarr") #Eric edit

#Eric edit:no pngs before it was created 
pipeline_path = '/media/mrl/Data/pipeline_connection'

png_path = Path("/media/mrl/Data/pipeline_connection/pngs") #Eric edit

txt_path = Path("/media/mrl/Data/pipeline_connection/fibscores.txt") #Eric edit

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
output_directory = f'/media/mrl/Data/pipeline_connection/kidney_segmentation/Omni_seg_pipeline_gpu/output'
#output_path = Path(f"{output_directory}/{os.listdir(output_directory)[0]}") #Eric edit

#Eric 12/27/2024: path for the patch to be saved to for omni-seg's input
#40x
path40x = f'/media/mrl/Data/pipeline_connection/kidney_segmentation/Omni_seg_pipeline_gpu/input/40X'
path10x = f'/media/mrl/Data/pipeline_connection/kidney_segmentation/Omni_seg_pipeline_gpu/input/10X'
path5x = f'/media/mrl/Data/pipeline_connection/kidney_segmentation/Omni_seg_pipeline_gpu/input/5X'

 # send all pyramid images of ROI to omni-seg and predict - Eric
#auto predict script path
omni_seg_path = '/media/mrl/Data/pipeline_connection/kidney_segmentation/Omni_seg_pipeline_gpu/Auto_Omni_predict.py'
#run the script
subprocesswd='/media/mrl/Data/pipeline_connection/kidney_segmentation/Omni_seg_pipeline_gpu'



##________________________________________________________________________________##
#for the auto_omni_seg process
subdirclean1 = '/media/mrl/Data/pipeline_connection/kidney_segmentation/Omni_seg_pipeline_gpu/segmentation_merge'
subdirclean2 = '/media/mrl/Data/pipeline_connection/kidney_segmentation/Omni_seg_pipeline_gpu/clinical_patches'
subdirclean3 = '/media/mrl/Data/pipeline_connection/kidney_segmentation/Omni_seg_pipeline_gpu/final_merge'

subprocess_step1 = '/media/mrl/Data/pipeline_connection/kidney_segmentation/Omni_seg_pipeline_gpu/1024_Step1_GridPatch_overlap_padding.py'
subprocess_step1_5 = '/media/mrl/Data/pipeline_connection/kidney_segmentation/Omni_seg_pipeline_gpu/1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py'
subprocess_step2 = '/media/mrl/Data/pipeline_connection/kidney_segmentation/Omni_seg_pipeline_gpu/Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py'
subprocess_step3 = '/media/mrl/Data/pipeline_connection/kidney_segmentation/Omni_seg_pipeline_gpu/step3.py'
subprocess_step4 = '/media/mrl/Data/pipeline_connection/kidney_segmentation/Omni_seg_pipeline_gpu/step4.py'

result_list = '/media/mrl/Data/pipeline_connection//kidney_segmentation/Omni_seg_pipeline_gpu/final_merge'
result_output_list = '/media/mrl/Data/pipeline_connection/kidney_segmentation/Omni_seg_pipeline_gpu/output'
result_input_list = '/media/mrl/Data/pipeline_connection/kidney_segmentation/Omni_seg_pipeline_gpu/input'

