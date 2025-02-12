import os
import shutil

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