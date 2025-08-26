import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os

# Load multiple DICOM slices into a 3D volume
path = "/Volumes/ExtremeSSD/AA_PhD_Projects/MRI_Koper/DICOM/25081920/42200000"  # folder containing all slices
files = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path)]

# Sort slices by ImagePositionPatient (or InstanceNumber if not available)
files.sort(key=lambda x: float(x.ImagePositionPatient[2]))

# Stack slices into a 3D numpy array
volume = np.stack([f.pixel_array for f in files])

print("Volume shape:", volume.shape)  # (num_slices, height, width)

# Display different views
slice_axial = volume[volume.shape[0] // 2, :, :]     # axial (top-down)
slice_coronal = volume[:, volume.shape[1] // 2, :]   # coronal (front view)
slice_sagittal = volume[:, :, volume.shape[2] // 2]  # sagittal (side view)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(slice_axial, cmap="gray")
axes[0].set_title("Axial")
axes[1].imshow(slice_coronal, cmap="gray")
axes[1].set_title("Coronal")
axes[2].imshow(slice_sagittal, cmap="gray")
axes[2].set_title("Sagittal")
plt.show()
