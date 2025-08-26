import os
import pydicom
import numpy as np

base_folder = "/Volumes/ExtremeSSD/AA_PhD_Projects/MRI_Koper/DICOM/25081920/BEAT_FQ-split/Subject1"

# Map series number to velocity component
series_map = {
    "SN56": "x",
    "SN58": "y",
    "SN60": "z"
}

all_volumes = []

for sn, vel_axis in series_map.items():
    series_folder = os.path.join(base_folder, f"BEAT_FQ_{sn}")
    print(f"Processing folder: {series_folder}")

    # Get all DICOM files (ignore hidden files)
    dcm_files = [os.path.join(series_folder, f)
                 for f in os.listdir(series_folder) if not f.startswith("._")]

    print(f"Found {len(dcm_files)} slices")

    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(f)
        arr = ds.pixel_array
        # Ensure first dim = phases
        if arr.shape[0] == 17:  # adjust if your number of phases differs
            slices.append(arr)
        else:
            slices.append(np.transpose(arr, (2, 0, 1)))

    # Stack along slice axis → shape: (phases, slices, height, width)
    volume = np.stack(slices, axis=1)
    print(f"Volume shape before reorder: {volume.shape}")

    # Reorder to (x, y, z, phases)
    volume_reordered = np.transpose(volume, (2, 3, 1, 0))
    print(f"Volume shape after reorder: {volume_reordered.shape}")
    print("Original dtype:", volume_reordered.dtype)

    # Convert to float32 before saving
    volume_reordered = volume_reordered.astype(np.float32)
    print("Converted dtype:", volume_reordered.dtype)

    all_volumes.append(volume_reordered)

    # Save with prefix data_sub01_ and velocity suffix
    folder_name = os.path.basename(series_folder)
    save_name = f"DATA_sub01_{folder_name}_velocity_{vel_axis}.npy"
    save_path = os.path.join(base_folder, save_name)
    np.save(save_path, volume_reordered)
    print(f"Saved: {save_path}\n")

# # ------------------------------
# # Create and save a common mask
# # ------------------------------
# # Stack all volumes along a new axis → shape: (3, x, y, z, phases)
# stacked_volumes = np.stack(all_volumes, axis=0)
#
# # Compute max intensity across all volumes and phases
# max_intensity = stacked_volumes.max()
# threshold = 0.1 * max_intensity  # 10% of max
#
# # Create boolean mask (True where intensity >= threshold)
# mask = stacked_volumes.max(axis=0) >= threshold
# print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, #True voxels: {mask.sum()}")
#
# # Save mask
# mask_save_path = os.path.join(base_folder, "DATA_sub01_mask_10percent.npy")
# np.save(mask_save_path, mask)
# print(f"Saved mask: {mask_save_path}")

# -----------------------------------
# Load SN54 and create mask
# -----------------------------------
mask_sn = "SN54"
mask_folder = os.path.join(base_folder, f"BEAT_FQ_{mask_sn}")
print(f"Processing mask folder: {mask_folder}")

dcm_files = [os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if not f.startswith("._")]
slices = []
for f in dcm_files:
    ds = pydicom.dcmread(f)
    arr = ds.pixel_array
    if arr.shape[0] == 17:  # adjust if necessary
        slices.append(arr)
    else:
        slices.append(np.transpose(arr, (2, 0, 1)))

volume_mask = np.stack(slices, axis=1)
volume_mask = np.transpose(volume_mask, (2, 3, 1, 0)).astype(np.float32)
print(f"Mask volume shape after reorder: {volume_mask.shape}")

# Create boolean mask (10% of max intensity)
threshold = 0.1 * volume_mask.max()
mask_bool = volume_mask >= threshold
print(f"Mask dtype: {mask_bool.dtype}, #True voxels: {mask_bool.sum()}")

# Save mask
mask_save_path = os.path.join(base_folder, "DATA_sub01_mask_SN54_10percent.npy")
np.save(mask_save_path, mask_bool)
print(f"Saved mask: {mask_save_path}")