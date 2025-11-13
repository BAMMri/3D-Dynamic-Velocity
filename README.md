# 3D-Dynamic-velocity

A collection of Python tools for processing 4D flow MRI data, including reconstruction from k-space, velocity calculation, and 3D strain analysis.

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib
- BART Toolbox (for reconstruction)

For BART toolbox installation, follow instructions at [BART website](https://mrirecon.github.io/bart/).

## Data Processing Pipeline

The toolkit consists of two main scripts:

1. `Reco_Bart_GPU.py` - Reconstructs images from k-space data and calculates velocity components
2. `calc_strain.py` - Calculates 3D strain from velocity data

### Step 1: Reconstruction and Velocity Calculation

```bash
python Reco_Bart_GPU.py /path/to/data_dir --venc 20
```

#### Arguments

- `data_dir`: Directory containing .cfl/.hdr k-space data files
- `--venc`: VENC value in cm/s (default: 20)

This script will:
1. Find all .cfl/.hdr pairs in the data directory
2. Reconstruct images using BART's parallel imaging tools
3. Calculate velocity components (x, y, z) from phase differences
4. Create a mask based on magnitude data
5. Save all results as .npy files in the same directory

### Step 2: Strain Calculation

```bash
python calc_strain.py --data-path /path/to/data --config config.json
```

#### Required Arguments

- `--data-path`: Directory containing velocity data files
- `--config`: Path to JSON configuration file with parameters

#### Optional Arguments

- `--prefix`: Prefix of velocity data files (default: "DATA")
- `--mask`: Path to mask file (.npy)
- `--roi-mask`: Path to region of interest mask file (.npy)
- `--output-eig`: Output eigenvalue file (.npy) (default: "Eig_v_output.npy")
- `--output-disp-x`: Output displacement x file (.npy)
- `--output-disp-y`: Output displacement y file (.npy)
- `--output-disp-z`: Output displacement z file (.npy)
- `--output-plot`: Output strain plot file (.png)
- `--slice-index`: Slice index for 2D analysis
- `--no-display`: Do not display plots

## Configuration File

Example `config.json`:

```json
{
  "CardiacNumberOfImages": 27,
  "InPlaneResolution": [1.5, 1.5],
  "SliceThickness": 1.5,
  "RepetitionTime": 6.7,
  "VelocityDirectionMatrix": [1, 1, 1]
  "AffineMatrix":np.eye(4)
}
```

The VelocityDirectionMatrix is a diagonal matrix represented as three values that controls the sign of each velocity component (x, y, z) obtained from the gradient [probing sequence](https://github.com/BAMMri/Pulseq-4DFlow/blob/main/gradient_probing.py). Use 1 for positive, -1 for negative direction.

The affine matrix is extracted from the raw data after conversion to ISMRMRD data with the get_matrix script, if none is provided a identy matrix is used as stand in.

## Example Workflow

1. **Data Reconstruction**:
   ```bash
   python Reco_Bart_GPU.py  /path/to/data --venc 25
   ```

2. **Strain Calculation**:
   ```bash
   python calc_strain.py --data-path  /path/to/data --config settings.json --output-plot strain_results.png
   ```

## Output

- Velocity components (.npy files)
- Displacement vectors (.npy files, optional)
- Strain eigenvalues (.npy file)
- Strain plot (.png file, optional)
