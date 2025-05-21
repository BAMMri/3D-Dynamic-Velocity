import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, EllipseSelector, Button
from matplotlib.patches import Ellipse
import os
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy import interpolate, signal


class RotatableEllipseROI:
    def __init__(self, ax):
        self.ax = ax
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.rotation = 0
        self.ellipse = None
        self.rotation_active = False

    def ellipse_select_callback(self, eclick, erelease):
        """Handle initial ellipse selection"""
        self.x0, self.y0 = eclick.xdata, eclick.ydata
        self.x1, self.y1 = erelease.xdata, erelease.ydata

        # Calculate center and axes lengths
        center_x = (self.x0 + self.x1) / 2
        center_y = (self.y0 + self.y1) / 2
        width = abs(self.x1 - self.x0)
        height = abs(self.y1 - self.y0)

        # Remove previous ellipse if exists
        if self.ellipse:
            self.ellipse.remove()

        # Create new ellipse patch
        self.ellipse = Ellipse((center_x, center_y), width, height,
                               angle=self.rotation,
                               fill=False,
                               edgecolor='red')
        self.ax.add_patch(self.ellipse)
        self.ax.figure.canvas.draw_idle()

    def on_key(self, event):
        """Handle rotation with arrow keys"""
        if not self.ellipse:
            return

        if event.key == 'left':
            self.rotation -= 5
        elif event.key == 'right':
            self.rotation += 5

        # Update ellipse rotation
        self.ellipse.set_angle(self.rotation)
        self.ax.figure.canvas.draw_idle()

    def create_rotated_mask(self, image_shape):
        """Create rotated elliptical mask"""
        if self.x0 is None:
            print("No ROI selected!")
            return None

        # Calculate parameters
        center_x = (self.x0 + self.x1) / 2
        center_y = (self.y0 + self.y1) / 2
        width = abs(self.x1 - self.x0)
        height = abs(self.y1 - self.y0)

        # Create mask grid
        y, x = np.ogrid[:image_shape[0], :image_shape[1]]

        # Rotation in radians
        angle_rad = np.deg2rad(self.rotation)

        # Rotate coordinates
        dx = x - center_x
        dy = y - center_y

        # Rotation matrix
        x_rot = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
        y_rot = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)

        # Ellipse equation
        mask = ((x_rot / (width / 2)) ** 2 + (y_rot / (height / 2)) ** 2) <= 1

        return mask


class ArrayROIProcessor:
    def __init__(self, data1, data2=None, selected_slice=None, selected_dim4=1):
        """
        Initialize processor with array data

        Parameters:
        - data1: First dataset x,y,z,nPhases
        - data2: Optional second array with same shape as data1
        - selected_slice: Slice to process in the 1st dimension (default is middle slice)
        - selected_dim4: Index for the 4th dimension (default 0)
        """
        self.data1 = data1
        self.data2 = data2
        self.selected_slice = selected_slice if selected_slice is not None else data1.shape[0] // 2
        self.selected_dim4 = selected_dim4

        # Get data for selected slice and dimension
        self.slice_data1 = data1[self.selected_slice, :, :, :, :]
        self.slice_data2 = None if data2 is None else data2[self.selected_slice, :, :, :, :]

        self.roi_processor = None
        self.dataset_names = ['Dataset 1', 'Dataset 2']

        # Add default repetition time for time calculations
        self.repetition_time = 6.7  # ms, default value

        # Add default VENC value
        self.venc = 100.0  # Default value

    def set_repetition_time(self, tr_ms):
        """Set the repetition time in milliseconds"""
        self.repetition_time = tr_ms

    def set_venc(self, venc_value):
        """Set the VENC value"""
        self.venc = venc_value

    def set_dataset_names(self, name1, name2=None):
        """Set custom names for the datasets for better labeling"""
        self.dataset_names = [name1, name2 if name2 else 'Dataset 2']

    def sigma_func(self, x, a, b, x0, dx):
        """
        Sigmoid function for fitting strain curves

        Parameters:
        - x: time points
        - a: upper plateau
        - b: lower plateau
        - x0: midpoint
        - dx: slope parameter
        """
        try:
            x_clipped = np.clip(x, -500, 500)
            exp_term = np.clip((x_clipped - x0) / dx, -500, 500)
            y = b + (a - b) / (1 + np.exp(exp_term))
            return y
        except Exception as e:
            print(f"Error in sigma_func: {e}")
            return np.zeros_like(x)

    def fit_signal(self, signal_data):
        """
        Fit signal data using sigmoid function for buildup and release rates

        Parameters:
        - signal_data: Mean signal data from ROI

        Returns:
        - Dictionary with fitted parameters and rates
        """
        # Calculate timebase in milliseconds
        dt = self.repetition_time
        timebase = np.arange(len(signal_data)) * dt

        # Find peak for splitting into buildup and release phases
        mx = np.max(signal_data)
        ind = np.argmax(signal_data)

        # Build-up phase - from start to peak
        e_bUP_roi = np.concatenate([signal_data[0:ind], mx * np.ones((ind))], axis=0)

        # Release phase - from peak to end (reversed for fitting)
        e_Rel_roi = np.concatenate([mx * np.ones((ind)), signal_data[ind:]], axis=0)
        if len(signal_data[ind:]) > 0:  # Ensure there's data after the peak
            e_Rel_roi = e_Rel_roi[::-1]

        # Create time arrays for both phases
        xdata_bUP = dt * np.arange(len(e_bUP_roi))
        xdata_Rel = dt * np.arange(len(e_Rel_roi))

        # Weight the endpoints for better fitting
        sigma_bUP = np.ones(len(xdata_bUP))
        sigma_bUP[[0, -1]] = 0.01

        sigma_Rel = np.ones(len(xdata_Rel))
        sigma_Rel[[0, -1]] = 0.01

        # Fit the buildup phase
        try:
            params_bUP, _ = curve_fit(self.sigma_func, xdata_bUP, e_bUP_roi,
                                      p0=(np.max(e_bUP_roi), np.min(e_bUP_roi), len(e_bUP_roi) / 2 * dt, 10 * dt),
                                      bounds=([0, -1, -30., -30.], [10, 0.6, 600., 400.]),
                                      method='trf', sigma=sigma_bUP)

            # Calculate buildup rate
            buildUp_rate = (params_bUP[0] - params_bUP[1]) / params_bUP[2]

            # Generate fitted curve for plotting
            t_bUP = np.linspace(0, xdata_bUP[-1], 100)
            y_bUP_fit = self.sigma_func(t_bUP, *params_bUP)

        except RuntimeError:
            print("Buildup curve fitting failed. Using default parameters.")
            params_bUP = (np.max(e_bUP_roi), np.min(e_bUP_roi), len(e_bUP_roi) / 2 * dt, 10 * dt)
            buildUp_rate = 0
            t_bUP = np.linspace(0, xdata_bUP[-1], 100)
            y_bUP_fit = self.sigma_func(t_bUP, *params_bUP)

        # Fit the release phase if there's data after the peak
        if len(signal_data[ind:]) > 10000:
            try:
                params_Rel, _ = curve_fit(self.sigma_func, xdata_Rel, e_Rel_roi,
                                          p0=(np.max(e_Rel_roi), np.min(e_Rel_roi), len(e_Rel_roi) / 2 * dt, 10 * dt),
                                          bounds=([0, -1, -60., -60.], [10, 0.6, 800., 200.]),
                                          method='trf', sigma=sigma_Rel)

                # Calculate release rate
                release_rate = (params_Rel[0] - params_Rel[1]) / params_Rel[2]

                # Generate fitted curve for plotting
                t_Rel = np.linspace(0, xdata_Rel[-1], 100)
                y_Rel_fit = self.sigma_func(t_Rel, *params_Rel)

            except RuntimeError:
                print("Release curve fitting failed. Using default parameters.")
                params_Rel = (np.max(e_Rel_roi), np.min(e_Rel_roi), len(e_Rel_roi) / 2 * dt, 10 * dt)
                release_rate = 0
                t_Rel = np.linspace(0, xdata_Rel[-1], 100)
                y_Rel_fit = self.sigma_func(t_Rel, *params_Rel)
        else:
            # Handle case where peak is at the end (no release phase)
            params_Rel = (0, 0, 0, 0)
            release_rate = 0
            t_Rel = []
            y_Rel_fit = []

        # Return fit results
        return {
            'buildup_params': params_bUP,
            'release_params': params_Rel,
            'buildup_rate': buildUp_rate,
            'release_rate': release_rate,
            't_buildup': t_bUP,
            'y_buildup_fit': y_bUP_fit,
            't_release': t_Rel,
            'y_release_fit': y_Rel_fit,
            'peak_index': ind,
            'timebase': timebase
        }

    def process_roi(self, event=None):
        """Create ROI mask and calculate mean signals for datasets"""
        if not self.roi_processor or self.roi_processor.x0 is None:
            print("Please draw an ellipse first!")
            return

        # Create rotated ROI mask
        roi_mask = self.roi_processor.create_rotated_mask((self.data1.shape[1], self.data1.shape[2]))
        if roi_mask is None:
            return

        # Save ROI mask
        np.save(f'roi_mask_slice_{self.selected_slice}.npy', roi_mask)

        # Extract data for the selected dimension from first dataset
        data_dim4_1 = self.slice_data1[:, :, self.selected_dim4, :]

        # Apply mask to first dataset
        roi_data_1 = data_dim4_1[roi_mask]

        # Calculate mean per phase for first dataset
        mean_per_phase_1 = np.mean(roi_data_1, axis=0)  # Shape will be (27,)

        # Save first mean signal data
        np.save(f'mean_signal_slice_{self.selected_slice}_dim4_{self.selected_dim4}_data1.npy', mean_per_phase_1)

        # Fit signal for first dataset
        fit_results_1 = self.fit_signal(mean_per_phase_1)

        # Handle second dataset if it exists
        mean_per_phase_2 = None
        fit_results_2 = None
        if self.data2 is not None:
            # Extract data from second dataset
            data_dim4_2 = self.slice_data2[:, :, self.selected_dim4, :]

            # Apply mask to second dataset
            roi_data_2 = data_dim4_2[roi_mask]

            # Calculate mean per phase for second dataset
            mean_per_phase_2 = np.mean(roi_data_2, axis=0)

            # Fit signal for second dataset
            fit_results_2 = self.fit_signal(mean_per_phase_2)

            # Save second mean signal data
            np.save(f'mean_signal_slice_{self.selected_slice}_dim4_{self.selected_dim4}_data2.npy', mean_per_phase_2)

        # Create figure for signal plot and fitted curves
        plt.figure(figsize=(12, 10))

        # Create subplots - one for mean signals, one for buildup fit, one for release fit
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 2, 3)
        ax3 = plt.subplot(2, 2, 4)

        # Plot mean signals on top subplot
        timebase = fit_results_1['timebase']
        ax1.plot(timebase, mean_per_phase_1, 'ro-', label=f"{self.dataset_names[0]} Mean Signal")
        if mean_per_phase_2 is not None:
            ax1.plot(timebase, mean_per_phase_2, 'bo-', label=f"{self.dataset_names[1]} Mean Signal")

        ax1.set_title(f'Mean Signal in ROI (Slice {self.selected_slice}, Dim4 = {self.selected_dim4})')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Signal Intensity')
        ax1.legend()
        ax1.grid(True)

        # Plot buildup fit on bottom-left subplot
        ax2.plot(timebase[:fit_results_1['peak_index'] + 1],
                 mean_per_phase_1[:fit_results_1['peak_index'] + 1], 'ro', label=f"{self.dataset_names[0]} Buildup")
        ax2.plot(fit_results_1['t_buildup'], fit_results_1['y_buildup_fit'], 'r-', label="Buildup Fit")

        if mean_per_phase_2 is not None:
            ax2.plot(timebase[:fit_results_2['peak_index'] + 1],
                     mean_per_phase_2[:fit_results_2['peak_index'] + 1], 'bo', label=f"{self.dataset_names[1]} Buildup")
            ax2.plot(fit_results_2['t_buildup'], fit_results_2['y_buildup_fit'], 'b-', label="Buildup Fit")

        ax2.set_title(f'Buildup Phase Fit')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Signal Intensity')
        ax2.legend()
        ax2.grid(True)

        # Display buildup rate on the plot
        ax2.text(0.05, 0.05, f"Buildup Rate: {fit_results_1['buildup_rate']:.5f}",
                 transform=ax2.transAxes, fontsize=10, color='red',
                 bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))

        if mean_per_phase_2 is not None:
            ax2.text(0.05, 0.15, f"Buildup Rate (2): {fit_results_2['buildup_rate']:.5f}",
                     transform=ax2.transAxes, fontsize=10, color='blue',
                     bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7))

        # Plot release fit on bottom-right subplot if there's data after the peak
        if fit_results_1['peak_index'] < len(mean_per_phase_1) - 1:
            ax3.plot(timebase[fit_results_1['peak_index']:],
                     mean_per_phase_1[fit_results_1['peak_index']:], 'ro', label=f"{self.dataset_names[0]} Release")

            # Need to map the release fit back to the original timebase
            release_timebase = np.linspace(timebase[fit_results_1['peak_index']],
                                           timebase[-1],
                                           len(fit_results_1['y_release_fit']))

            ax3.plot(release_timebase, fit_results_1['y_release_fit'][::-1], 'r-', label="Release Fit")

            if mean_per_phase_2 is not None and fit_results_2['peak_index'] < len(mean_per_phase_2) - 1:
                ax3.plot(timebase[fit_results_2['peak_index']:],
                         mean_per_phase_2[fit_results_2['peak_index']:], 'bo', label=f"{self.dataset_names[1]} Release")

                release_timebase2 = np.linspace(timebase[fit_results_2['peak_index']],
                                                timebase[-1],
                                                len(fit_results_2['y_release_fit']))

                ax3.plot(release_timebase2, fit_results_2['y_release_fit'][::-1], 'b-', label="Release Fit")

            ax3.set_title(f'Release Phase Fit')
            ax3.set_xlabel('Time (ms)')
            ax3.set_ylabel('Signal Intensity')
            ax3.legend()
            ax3.grid(True)

            # Display release rate on the plot
            ax3.text(0.05, 0.05, f"Release Rate: {fit_results_1['release_rate']:.5f}",
                     transform=ax3.transAxes, fontsize=10, color='red',
                     bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))

            if mean_per_phase_2 is not None:
                ax3.text(0.05, 0.15, f"Release Rate (2): {fit_results_2['release_rate']:.5f}",
                         transform=ax3.transAxes, fontsize=10, color='blue',
                         bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7))

        plt.tight_layout()
        plt.show()

        # Print fit results summary
        print("\n--- Fit Results Summary ---")
        print(f"Dataset: {self.dataset_names[0]}")
        print(f"Buildup Rate: {fit_results_1['buildup_rate']:.5f}")
        print(f"Release Rate: {fit_results_1['release_rate']:.5f}")
        print(f"Buildup Parameters (a, b, x0, dx): {fit_results_1['buildup_params']}")
        print(f"Release Parameters (a, b, x0, dx): {fit_results_1['release_params']}")

        if mean_per_phase_2 is not None:
            print(f"\nDataset: {self.dataset_names[1]}")
            print(f"Buildup Rate: {fit_results_2['buildup_rate']:.5f}")
            print(f"Release Rate: {fit_results_2['release_rate']:.5f}")
            print(f"Buildup Parameters (a, b, x0, dx): {fit_results_2['buildup_params']}")
            print(f"Release Parameters (a, b, x0, dx): {fit_results_2['release_params']}")

        # Save fit results
        fit_results = {
            'dataset1': {
                'name': self.dataset_names[0],
                'buildup_rate': fit_results_1['buildup_rate'],
                'release_rate': fit_results_1['release_rate'],
                'buildup_params': fit_results_1['buildup_params'],
                'release_params': fit_results_1['release_params']
            }
        }

        if mean_per_phase_2 is not None:
            fit_results['dataset2'] = {
                'name': self.dataset_names[1],
                'buildup_rate': fit_results_2['buildup_rate'],
                'release_rate': fit_results_2['release_rate'],
                'buildup_params': fit_results_2['buildup_params'],
                'release_params': fit_results_2['release_params']
            }

        np.save(f'fit_results_slice_{self.selected_slice}_dim4_{self.selected_dim4}.npy', fit_results)

        return mean_per_phase_1, mean_per_phase_2


def plot_phase_data(data, slice_idx, dim4_idx, phase_idx, ax):
    """Plot data for selected phase"""
    # Get data for the specific slice, dimension, and phase
    img_data = data[slice_idx, :, :, dim4_idx, phase_idx]

    # Plot the data
    img_plot = ax.imshow(img_data, cmap='viridis')
    ax.set_title(f'Slice {slice_idx}, Dim4 = {dim4_idx}, Phase {phase_idx}')
    return img_plot


def load_config_json(config_file_path):
    """Load configuration from JSON file"""
    try:
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_file_path}")
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default values")
        return None


# Fix Wayland plugin issue (if needed)
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Force X11 backend

# Configuration variables
# --------------------
# Set config file path
config_file_path = 'specifications.json'

# Set file paths for the datasets here
file_path1 =   # First dataset path (required)
file_path2 = None  # Second dataset path (optional)

# Set default values (to be overridden by config if available)
default_slice_idx = 70
default_dim4_idx = 0
repetition_time = 6.7  # ms
venc = 100.0  # Default VENC value
# --------------------

# Load configuration from JSON file
config = load_config_json(config_file_path)

# Update parameters from config file if available
if config:
    # Get repetition time from config
    if 'RepetitionTime' in config:
        repetition_time = float(config['RepetitionTime'])
        print(f"Using TR from config: {repetition_time} ms")

    # Get default slice from config if available
    if 'DefaultSlice' in config:
        default_slice_idx = int(config['DefaultSlice'])
        print(f"Using default slice from config: {default_slice_idx}")

    # Get VENC from config if available
    if 'VENC' in config:
        venc = float(config['VENC'])
        print(f"Using VENC from config: {venc}")

    # Get file paths from config if available
    if 'DataPath1' in config:
        file_path1 = config['DataPath1']
        print(f"Using data path from config: {file_path1}")

    if 'DataPath2' in config:
        file_path2 = config['DataPath2']
        print(f"Using second data path from config: {file_path2}")

# Load datasets
try:
    data1 = np.load(file_path1)
    dataset1_name = os.path.basename(file_path1).replace('.npy', '')
    print(f"Loaded {dataset1_name} with shape {data1.shape}")
except Exception as e:
    print(f"Error loading first dataset: {e}")
    raise SystemExit("Cannot continue without first dataset.")

# Try to load the second dataset if specified
data2 = None
dataset2_name = None
if file_path2 is not None:
    try:
        data2 = np.load(file_path2)
        dataset2_name = os.path.basename(file_path2).replace('.npy', '')
        print(f"Loaded {dataset2_name} with shape {data2.shape}")

        # Check if shapes match
        if data1.shape != data2.shape:
            print(f"Warning: Dataset shapes don't match! {data1.shape} vs {data2.shape}")
    except Exception as e:
        print(f"Error loading second dataset: {e}")
        print("Continuing with only the first dataset.")
        data2 = None

# Set default values
slice_idx = default_slice_idx
dim4_idx = default_dim4_idx

# Create figure and initial plot
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25)

# Create processor for datasets
roi_processor = ArrayROIProcessor(data1, data2, selected_slice=slice_idx, selected_dim4=dim4_idx)

# Set repetition time and VENC from config
roi_processor.set_repetition_time(repetition_time)
roi_processor.set_venc(venc)

# Set dataset names
if data2 is not None:
    roi_processor.set_dataset_names(dataset1_name, dataset2_name)
else:
    roi_processor.set_dataset_names(dataset1_name)

# Initial plot
img_plot = plot_phase_data(data1, slice_idx, dim4_idx, 0, ax)

# Create a fixed axes for the colorbar
divider = make_axes_locatable(ax)
cbar_ax = divider.append_axes("right", size="5%", pad=0.05)

# Create the colorbar
cbar = fig.colorbar(img_plot, cax=cbar_ax)

# Create Rotatable ROI Processor
roi_processor.roi_processor = RotatableEllipseROI(ax)

# Create EllipseSelector
toggle_selector = EllipseSelector(
    ax, roi_processor.roi_processor.ellipse_select_callback,
    useblit=True, button=[1],
    minspanx=5, minspany=5,
    spancoords='pixels', interactive=True
)

# Connect key press events for ellipse rotation
fig.canvas.mpl_connect('key_press_event', roi_processor.roi_processor.on_key)

# Create slider for phases
ax_phase = plt.axes([0.2, 0.1, 0.6, 0.03])
phase_slider = Slider(ax_phase, 'Phase', 0, data1.shape[4] - 1, valinit=0, valstep=1)


# Function to update when phase changes
def update(val):
    phase_idx = int(phase_slider.val)
    ax.clear()

    # Create new image plot for data1
    img_plot = plot_phase_data(data1, slice_idx, dim4_idx, phase_idx, ax)

    # Update colorbar
    cbar.update_normal(img_plot)

    # Re-add ROI if it exists
    if roi_processor.roi_processor.ellipse is not None:
        center_x = (roi_processor.roi_processor.x0 + roi_processor.roi_processor.x1) / 2
        center_y = (roi_processor.roi_processor.y0 + roi_processor.roi_processor.y1) / 2
        width = abs(roi_processor.roi_processor.x1 - roi_processor.roi_processor.x0)
        height = abs(roi_processor.roi_processor.y1 - roi_processor.roi_processor.y0)

        roi_processor.roi_processor.ellipse = Ellipse(
            (center_x, center_y),
            width, height,
            angle=roi_processor.roi_processor.rotation,
            fill=False,
            edgecolor='red'
        )
        ax.add_patch(roi_processor.roi_processor.ellipse)

    fig.canvas.draw_idle()


phase_slider.on_changed(update)

# Add button to trigger ROI processing
ax_roi = plt.axes([0.7, 0.05, 0.2, 0.075])
btn_roi = Button(ax_roi, 'Process ROI')
btn_roi.on_clicked(roi_processor.process_roi)

# Add slider for slice selection (first dimension)
ax_slice = plt.axes([0.2, 0.05, 0.4, 0.03])
slice_slider = Slider(ax_slice, 'Slice', 0, data1.shape[0] - 1, valinit=slice_idx, valstep=1)


# Function to update when slice changes
def update_slice(val):
    global slice_idx
    slice_idx = int(slice_slider.val)
    # Update the processor's selected slice
    roi_processor.selected_slice = slice_idx
    roi_processor.slice_data1 = data1[slice_idx, :, :, :, :]
    if data2 is not None:
        roi_processor.slice_data2 = data2[slice_idx, :, :, :, :]
    # Update the display
    update(phase_slider.val)


slice_slider.on_changed(update_slice)

# Add instructions text
plt.figtext(0.1, 0.001,
            f"Instructions: Draw ellipse with mouse, rotate with arrow keys, click 'Process ROI' to fit signal curves. TR={repetition_time}ms, VENC={venc}",
            wrap=True, fontsize=10)

# Display status message about loaded datasets
if data2 is not None:
    print(f"Ready to process both datasets: {dataset1_name} and {dataset2_name}")
else:
    print(f"Ready to process single dataset: {dataset1_name}")

plt.show()
