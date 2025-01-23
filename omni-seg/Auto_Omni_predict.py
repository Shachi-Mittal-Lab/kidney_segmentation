#for file creation and mannipulation
import os
#for importing other directory's module
import sys
#for moving files between directories
import shutil
#to run the Step 1-4 python files
import subprocess
#to show error message
import traceback
#to prevent computer freeze
import psutil
import time
from threading import Thread

sys.path.append('/media/mrl/Data/pipeline_connection/kidney_segmentation')
from directory_paths import subprocess_step1, subprocess_step1_5, subprocess_step2,subprocess_step2, subprocess_step3, result_output_list, result_input_list, inputdir, output_directory

'''
run a for loop that loop through the output and all ROI packs directory and compare if they have the same names?
Pro: checks which image has been run
Con: not to the level of individual annoation
However, good first stage to work on
'''

def main():
    '''
    monitor_thread = Thread(target=cpu_limiter)
    #set as a daemon it coexist with the main thread at the same time
    monitor_thread.daemon = True
    monitor_thread.start()
    '''
    # remove all contents of folder
    # unless it is the clinical patches, final_merge, or segmentation merge folders
    # instead step inside of these folders and delete the contents
    def clean_dirs(directory):
        contents = os.listdir(directory)
        print(f"{directory} output contents: {contents}")
        for item in contents:
            if item in ["clinical_patches", "final_merge", "segmentation_merge"]:
                print(f"deleting folder {item}")
                shutil.rmtree(os.path.join(directory, item))
                print(f"remaking folder {item}")
                os.mkdir(os.path.join(directory, item))
            elif os.path.isdir(os.path.join(directory, item)):
                print(f"Deleting folder {item}")
                shutil.rmtree(os.path.join(directory, item))
            else:
                print(f"Deleting {item}")
                os.remove(os.path.join(directory, item))

    #clean output dirs if there was a run prior
    clean_dirs(output_directory)

    #run the Step 1 to 4 in sequence, using subporcess module
    #add try feature, so that we can know which step stopped
    try:
        subprocess.run(['python3', subprocess_step1])
    except:
        print('Step 1 is interrupted by error')

    try:
        subprocess.run(['python3', subprocess_step1_5])
    except:
        print('Step 1.5 is interrupted by error')

    try:
        print("Starting step 2 process")
        subprocess.run(['python3', subprocess_step2])

    except:
        print('Step 2 is interrupted by error')

    try:
        subprocess.run(['python3', subprocess_step3])
    except:
        print('Step 3 is interrupted by error')

    # after run and result collection, clean the input directory
    clean_dirs(inputdir)
     

#after the script, code in a all-running-involved-directory reset protocal for when there is an error and the process stops half way  
def if_error(error):
    '''
    build for scenairo if the process stoped due to an error:
    -return the error
    -return which image is currently being analysed
    -return which annoation is currently being analysed
    -return which image and annotation has been completed
    -return the error message itself(i think the traceback error message will show anyways)
    -clean the following directories so that the process can run again without issues
        -return the currently loaded image back into the ROI pack folder
        -the input folders(done with the above step)
        -the segmentation_merge, clinical_patches, and final_merge(will be cleaned before a run anyway)
        -the result directory in output folder of the current run, need to check if a result pack already exist or not
    '''
    print('Error has occured.')
    traceback.print_exc()

    #code for determining which image and annotation is being analysed
    current_file = os.listdir(f'{result_input_list}/5X')
    print(current_file)

    #code for which file and annotation has been completed: using the record_file variable added in the main script section
    
    #current_image = os.listdir('/media/mrl/Data/pipeline_connection/Omni_seg_pipeline_gpu/input')
    error_file_name = os.listdir(f'{result_input_list}/5X')

    #code for moving loaded image back to ROI pack
    for resolution in [5, 10, 40]:
        #where the ROI png file is from and where they are going
        #error_to_dir = f'./input/all input ROI/{current_image}/{resolution}X/{error_file_name}'
        error_from_dir = f'{result_input_list}/{resolution}X/{error_file_name}'
        #moving files
        try:
            #shutil.move(error_from_dir, error_to_dir)
            os.remove(error_from_dir)
            print(f'{error_file_name} unloaded for Omni-Seg')
        except FileNotFoundError:
            print(f'{error_file_name} is not in the {error_from_dir}')
    
    #checking if there is already a result folder folder
    
    for exisitng_result in os.listdir(result_output_list):
        if exisitng_result == error_file_name:
            print(f'{exisitng_result} is in the output folder, with an error present, the {exisitng_result} file will be deleted in preparation for rerun')

try:
    
    main()
except Exception as e:
    if_error(e)
        








