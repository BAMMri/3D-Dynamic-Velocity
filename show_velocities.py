import math
import numpy as np
import matplotlib.pyplot as plt

# load data
mri_data = np.load("/Volumes/ExtremeSSD/AA_PhD_Projects/MRI_Koper/DICOM/25081920/BEAT_FQ-split/Subject1/DATA_sub01_BEAT_FQ_SN60_velocity_z.npy")
print(mri_data.shape)
print(mri_data.dtype)
# pick indices
z_idx = 30        # slice index in z
venc_idx = 2      # venc component (not used in your abs version yet)

# take magnitude image (abs) for chosen slice across phases
#data = np.abs(mri_data[50, :, :, :])  # shape (100, 76, 33)
# take magnitude image (abs) for chosen slice across phases
data = mri_data[50, :, :, :]  # shape (100, 76, 33)

# keep every other phase
data = data[:, :, ::2]
n_phases = data.shape[-1]

# grid layout
cols = math.ceil(math.sqrt(n_phases))
rows = math.ceil(n_phases / cols)

fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

for i in range(n_phases):
    r, c = divmod(i, cols)
    ax = axes[r, c]
    ax.imshow(data[:, :, i].T, cmap='gray', origin='lower', vmin=-1, vmax=2598)
    ax.set_title(f"Phase {i*2}")  # multiply by 2 to show original phase number
    ax.axis('off')

# hide any empty subplots
for j in range(n_phases, rows*cols):
    r, c = divmod(j, cols)
    axes[r, c].axis('off')

plt.tight_layout()
plt.show()
