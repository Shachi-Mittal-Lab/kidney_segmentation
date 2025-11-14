import subprocess

# ---- CONFIGURATION ----
zarr_file = "/media/mrl/Data/pipeline_connection/ndpis/full_pipeline_preds_14Nov2025/BR21-2024-A-1-9-TRI - 2022-08-10 23.46.06.zarr"  # Change this to your Zarr file name

# List of datasets to visualize
datasets = [
    "raw*",
    "mask/fincap",
    "mask/vessel_40x",
    "mask/tubule_40x",
    "mask/fininflamm",
    "mask/fincollagen_exclusion",
    "mask/finfib"
]

# ---- BUILD COMMAND ----
command = ["neuroglancer", "-d"]
for dataset in datasets:
    command.append(f"{zarr_file}/{dataset}")

# ---- EXECUTE COMMAND ----
print("Executing command:")
print(" ".join(command))

subprocess.run(command)