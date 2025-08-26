import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, EllipseSelector, Button
from matplotlib.patches import Ellipse
import os
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy import interpolate, signal

# === Create Side-by-Side Viewer for 3 Datasets ===

# Create figure with 3 subplots for the datasets
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(bottom=0.25)

# Slice and dim4 index
slice_idx = default_slice_idx
dim4_idx = default_dim4_idx

# Create processors for each dataset
roi_processor1 = ArrayROIProcessor(data1, selected_slice=slice_idx, selected_dim4=dim4_idx)
roi_processor2 = ArrayROIProcessor(data2, selected_slice=slice_idx, selected_dim4=dim4_idx)
roi_processor3 = ArrayROIProcessor(data3, selected_slice=slice_idx, selected_dim4=dim4_idx)

# Shared ROI processor instance
shared_roi = RotatableEllipseROI(ax1)  # Draw on ax1, sync to others
roi_processor1.roi_processor = shared_roi
roi_processor2.roi_processor = shared_roi
roi_processor3.roi_processor = shared_roi

# Set TR, VENC, names
for processor, name in zip([roi_processor1, roi_processor2, roi_processor3],
                           [dataset1_name, dataset2_name, dataset3_name]):
    processor.set_repetition_time(repetition_time)
    processor.set_venc(venc)
    processor.set_dataset_names(name)

# Initial plot for each dataset
img1 = plot_phase_data(data1, slice_idx, dim4_idx, 0, ax1)
img2 = plot_phase_data(data2, slice_idx, dim4_idx, 0, ax2)
img3 = plot_phase_data(data3, slice_idx, dim4_idx, 0, ax3)

# Colorbars
divider1 = make_axes_locatable(ax1)
cbar1 = fig.colorbar(img1, cax=divider1.append_axes("right", size="5%", pad=0.05))

divider2 = make_axes_locatable(ax2)
cbar2 = fig.colorbar(img2, cax=divider2.append_axes("right", size="5%", pad=0.05))

divider3 = make_axes_locatable(ax3)
cbar3 = fig.colorbar(img3, cax=divider3.append_axes("right", size="5%", pad=0.05))

# Ellipse selector (only drawn on ax1, shared)
toggle_selector = EllipseSelector(
    ax1, shared_roi.ellipse_select_callback,
    useblit=True, button=[1], minspanx=5, minspany=5,
    spancoords='pixels', interactive=True
)

# Handle key press (left/right) for ellipse rotation
fig.canvas.mpl_connect('key_press_event', shared_roi.on_key)


# === SLIDERS ===
# Phase slider
ax_phase = plt.axes([0.2, 0.1, 0.6, 0.03])
phase_slider = Slider(ax_phase, 'Phase', 0, data1.shape[4] - 1, valinit=0, valstep=1)

# Slice slider
ax_slice = plt.axes([0.2, 0.05, 0.4, 0.03])
slice_slider = Slider(ax_slice, 'Slice', 0, data1.shape[0] - 1, valinit=slice_idx, valstep=1)


def update_all_views(val):
    phase = int(phase_slider.val)
    sl = int(slice_slider.val)

    # Update internal slice
    for processor, data, ax in zip(
        [roi_processor1, roi_processor2, roi_processor3],
        [data1, data2, data3],
        [ax1, ax2, ax3]
    ):
        processor.selected_slice = sl
        processor.slice_data1 = data[sl, :, :, :, :]
        ax.clear()
        plot_phase_data(data, sl, dim4_idx, phase, ax)

        # Re-add ellipse if exists
        if shared_roi.ellipse is not None:
            cx = (shared_roi.x0 + shared_roi.x1) / 2
            cy = (shared_roi.y0 + shared_roi.y1) / 2
            width = abs(shared_roi.x1 - shared_roi.x0)
            height = abs(shared_roi.y1 - shared_roi.y0)
            ellipse = Ellipse((cx, cy), width, height,
                              angle=shared_roi.rotation, fill=False, edgecolor='red')
            shared_roi.ellipse = ellipse
            ax.add_patch(ellipse)

    fig.canvas.draw_idle()


phase_slider.on_changed(update_all_views)
slice_slider.on_changed(update_all_views)


# === PROCESS BUTTON ===
def process_all(event=None):
    roi_processor1.process_roi()
    roi_processor2.process_roi()
    roi_processor3.process_roi()


ax_btn = plt.axes([0.7, 0.05, 0.2, 0.075])
btn = Button(ax_btn, "Process ROI")
btn.on_clicked(process_all)

# === Instructions ===
plt.figtext(0.05, 0.005,
            f"Draw ROI on left image, rotate with arrow keys, shared across all datasets. "
            f"Click 'Process ROI' to extract & fit. TR={repetition_time}ms, VENC={venc}.",
            wrap=True, fontsize=9)

plt.show()
