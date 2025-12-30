from pathlib import Path

# inputs #

# file inputs
input_path = Path("/mnt/49c617b2-221d-486c-9542-4ff0dc297048/kidney/full_pipeline_results_bowser_04Dec2025") #Eric edit

#Eric edit:no pngs before it was created 
png_path = Path("/home/riware@netid.washington.edu/Documents/kidney/kidney_segmentation/omni-seg/input")

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
    "/mnt/49c617b2-221d-486c-9542-4ff0dc297048/kidney/kidney_segmentation/model5_epoch8.pt" #Eric edit
)
cortmed_starting_model_path = Path(
    "/mnt/49c617b2-221d-486c-9542-4ff0dc297048/kidney/kidney_segmentation/starting_model.pt" #Eric edit
)

# kmeans clustering centers
gray_cluster_centers = Path(
    "/mnt/49c617b2-221d-486c-9542-4ff0dc297048/kidney/kidney_segmentation/fibrosis_score/average_centers_5.txt" #Eric edit
)

# omni-seg output path
output_directory = f'/home/riware@netid.washington.edu/Documents/kidney/kidney_segmentation/omni-seg/outputs'
#output_path = Path(f"{output_directory}/{os.listdir(output_directory)[0]}") #Eric edit

#Eric 12/27/2024: path for the patch to be saved to for omni-seg's input
#40x
inputdir = f'/home/riware@netid.washington.edu/Documents/kidney/kidney_segmentation/omni-seg/input'
 # send all pyramid images of ROI to omni-seg and predict - Eric
#auto predict script path
omni_seg_path = '/home/riware@netid.washington.edu/Documents/kidney/kidney_segmentation/omni-seg/Auto_Omni_predict.py'
#run the script
subprocesswd='/home/riware@netid.washington.edu/Documents/kidney/kidney_segmentation/omni-seg'

omniseg_weights_file=""


##________________________________________________________________________________##
#for the auto_omni_seg process
subprocess_step1 = '/home/riware@netid.washington.edu/Documents/kidney/kidney_segmentation/omni-seg/1024_Step1_GridPatch_overlap_padding.py'
subprocess_step1_5 = '/home/riware@netid.washington.edu/Documents/kidney/kidney_segmentation/omni-seg//1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py'
subprocess_step2 = '/home/riware@netid.washington.edu/Documents/kidney/kidney_segmentation/omni-seg/Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py'
subprocess_step3 = '/home/riware@netid.washington.edu/Documents/kidney/kidney_segmentation/omni-seg/step3.py'
subprocess_step4 = '/home/riware@netid.washington.edu/Documents/kidney/kidney_segmentation/omni-seg/step4.py'

result_list = '/home/riware@netid.washington.edu/Documents/kidney/kidney_segmentation/omni-seg/outputs/final_merge'
result_output_list = '/home/riware@netid.washington.edu/Documents/kidney/kidney_segmentation/omni-seg/outputs'
result_input_list = '/home/riware@netid.washington.edu/Documents/kidney/kidney_segmentation/omni-seg/input'

