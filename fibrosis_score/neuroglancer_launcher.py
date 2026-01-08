import subprocess

# ---- CONFIGURATION ----
zarr_file = "/media/mrl/Data/pipeline_connection/ndpis/testing_new_full_pipeline_07Jan2025/BR21-2050-A-1-9-TRI - 2022-08-10 23.52.57.zarr"  # Change this to your Zarr file name

# List of datasets to visualize
datasets = [
    "raw*",
    "mask/fincap",
    "mask/vessel_40x",
    "mask/tubule_40x",
    "mask/fininflamm",
    "mask/fincollagen_exclusion",
    "mask/finfib",
    "mask/abnormaltissue",
    "mask/foreground_eroded"
]

# ---- BUILD COMMAND ----
command = ["neuroglancer", "-d"]
for dataset in datasets:
    command.append(f"{zarr_file}/{dataset}")

# ---- EXECUTE COMMAND ----
print("Executing command:")
print(" ".join(command))

subprocess.run(command)