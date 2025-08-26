import numpy as np
import matplotlib.pyplot as plt


import sys
sys.path.append('/Users/sabine/bart/python')
from bart import cfl

# Path to your file (without extension)
file_path = "/Volumes/Extreme SSD/AA_PhD_Projects/NMESOptimization-Local/data/measurements/meas_MID00092_FID18752_pulseq_US9_venc20"  # e.g., "/Users/sabine/data/reco"

# Read .cfl data
data = cfl.readcfl(file_path)

# Take magnitude if complex
if np.iscomplexobj(data):
    data = np.abs(data)

data = np.squeeze(data)
print(data.shape)
print(np.max(data))
print(np.min(data))
# Take first slice (works for 3D or higher)
slice_2d = data[50, :, :, 0,8,0]
print(np.max(slice_2d))
# Plot
plt.imshow(slice_2d, cmap='gray', vmin=-0, vmax=0.000001)
plt.title("First Slice")
plt.axis('off')
plt.show()
