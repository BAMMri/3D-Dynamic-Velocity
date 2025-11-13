import numpy as np
import nibabel as nib
import argparse
import os
import glob
import json
import re


def find_unique_file(directory, pattern):
    """Searches for a single file matching a pattern in a directory."""
    full_pattern = os.path.join(directory, pattern)
    files = glob.glob(full_pattern)

    if len(files) == 0:
        return None, f"Error: No file found matching pattern '{pattern}' in directory '{directory}'."
    elif len(files) > 1:
        file_list_str = ", ".join([os.path.basename(f) for f in files])
        return None, f"Error: Found {len(files)} files: {file_list_str}. Please ensure only one file matches the pattern '{pattern}'."
    else:
        return files[0], None


def load_json_config(config):
    """Load JSON config and extract the affine matrix."""
    if not os.path.exists(config):
        raise FileNotFoundError(f"Configuration file not found at: {config}")

    with open(config, 'r') as f:
        config1 = json.load(f)

    affine_list = config1.get('affine')
    if affine_list is None:
        # Fallback to Identity Matrix if affine is missing
        print("Warning: 'affine' key not found in the configuration file. Defaulting to Identity Matrix (np.eye(4)).")
        return np.eye(4, dtype=np.float32)

    try:
        affine_matrix = np.array(affine_list, dtype=np.float32)
        if affine_matrix.shape != (4, 4):
            raise ValueError(f"Affine matrix must be 4x4. Found shape: {affine_matrix.shape}")
        print("Successfully loaded 4x4 affine matrix from config.")
        return affine_matrix
    except Exception as e:
        raise ValueError(f"Error processing affine matrix in config: {e}")


def save_nifti(data, affine, output_path, prefix, description="Data"):
    """Creates a Nifti1Image and saves it as a compressed NIfTI file."""
    output_filename = os.path.join(output_path, f"{prefix}_{description}.nii.gz")

    nifti_img = nib.Nifti1Image(data.astype(np.float32), affine)

    nib.save(nifti_img, output_filename)
    print(f"-> Saved {description} to: {os.path.basename(output_filename)}")


def main():
    parser = argparse.ArgumentParser(
        description='Save strain/displacement NumPy arrays as NIfTI files using an affine matrix from a JSON config.')

    parser.add_argument('--config', required=True,
                        help='Path to the JSON configuration file containing the affine matrix.')
    parser.add_argument('--prefix', required=True,
                        help='The unique prefix/identifier used in the input file names (e.g., DATA_0029).')

    parser.add_argument('--input-dir', default='.',
                        help='Directory containing the input .npy files (defaults to current dir).')
    parser.add_argument('--output-dir', default=None,
                        help='Directory where output NIfTI files will be saved (defaults to current dir).')

    parser.add_argument('--save-mask', action='store_true', help='Save the 4D mask.')
    parser.add_argument('--save-vel-components', action='store_true',
                        help='Save the separate 4D velocity components (Vx, Vy, Vz).')
    parser.add_argument('--save-disp', action='store_true',
                        help='Save the 4D displacement components (disp_x, disp_y, disp_z).')

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir

    # --- Setup and Affine Load ---
    try:
        affine_matrix = load_json_config(args.config)
    except Exception as e:
        print(f"FATAL ERROR: Failed to load affine matrix. {e}")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")

    # --- Load Mandatory Files ---

    # 1. Load Processed Data (Velocities and Mask)
    processed_data_pattern = f"*{args.prefix}*_processed_data.npy"
    processed_file, err = find_unique_file(args.input_dir, processed_data_pattern)
    if err:
        print(f"FATAL ERROR: {err}")
        return

    print(f"Loading processed data from: {os.path.basename(processed_file)}")
    processed_data = np.load(processed_file, allow_pickle=True).item()
    velocities_5D = processed_data['velocities']
    mask_4d = processed_data['mask']

    # 2. Load Eigenvalues (Strains) - Using Specific Naming Patterns
    base_filename = os.path.basename(processed_file)
    numerical_id = None
    eig_v_file = None

    # Extract the numerical ID (e.g., '28' from 'MID00028')
    mid_match = re.search(r'MID000(\d+)', base_filename) or re.search(r'MID00(\d+)', base_filename)

    if mid_match:
        numerical_id = mid_match.group(1).lstrip('0')

        # Try pattern 1: Eig_v_outputX.npy
        pattern_1 = f"Eig_v_output{numerical_id}.npy"
        eig_v_file, _ = find_unique_file(args.input_dir, pattern_1)

        if not eig_v_file:
            # Try pattern 2: EIGX.npy
            pattern_2 = f"Eig{numerical_id}.npy"
            eig_v_file, _ = find_unique_file(args.input_dir, pattern_2)

        if eig_v_file:
            print(f"Successfully located EIG file using extracted numerical ID '{numerical_id}'.")

    # Fallback: If ID extraction failed or all simple searches failed, try the generic complex search
    if not eig_v_file:
        print(f"Attempting generic Eig search using complex prefix: '{args.prefix}'")
        eig_v_pattern = f"Eig_*{args.prefix}*.npy"
        eig_v_file, err = find_unique_file(args.input_dir, eig_v_pattern)

    if err and not eig_v_file:
        print(
            f"FATAL ERROR: Failed to find mandatory Eigenvalue file. Searched patterns: Eig_v_outputX.npy, EIGX.npy, and Eig_*{args.prefix}*.npy. Details: {err}")
        return

    print(f"Loading eigenvalues (strain) from: {os.path.basename(eig_v_file)}")
    eig_v_5D = np.load(eig_v_file)

    # --- Mandatory Saves ---

    # 1. Combined Velocities (5D: X, Y, Z, Phases, 3 Components)
    print("\n--- Mandatory Saving: Combined Velocities ---")
    save_nifti(velocities_5D, affine_matrix, args.output_dir, args.prefix, "velocity_combined_5D")

    # 2. Combined Eigenstrains (5D: X, Y, Z, Phases, 3 Components)
    print("\n--- Mandatory Saving: Combined Principal Strains ---")
    # Eig_v shape: (X, Y, Z, 3 Eigenvalues, Phases). Reorder axes to (X, Y, Z, Phases, 3 Eigenvalues).
    eig_v_reordered = np.moveaxis(eig_v_5D, 3, 4)

    save_nifti(eig_v_reordered, affine_matrix, args.output_dir, args.prefix, "strain_combined_5D")

    # --- Optional Saves ---

    # 3. Mask (4D: X, Y, Z, Phases)
    if args.save_mask:
        print("\n--- Optional Saving: Mask ---")
        save_nifti(mask_4d, affine_matrix, args.output_dir, args.prefix, "mask_4d")

    # 4. Separate Velocity Components (4D: X, Y, Z, Phases)
    if args.save_vel_components:
        print("\n--- Optional Saving: Velocity Components ---")

        Vx = velocities_5D[..., 0]
        Vy = velocities_5D[..., 1]
        Vz = velocities_5D[..., 2]

        save_nifti(Vx, affine_matrix, args.output_dir, args.prefix, "velocity_Vx")
        save_nifti(Vy, affine_matrix, args.output_dir, args.prefix, "velocity_Vy")
        save_nifti(Vz, affine_matrix, args.output_dir, args.prefix, "velocity_Vz")

    # 5. Displacements (4D: X, Y, Z, Phases)
    if args.save_disp:
        print("\n--- Optional Saving: Displacements ---")

        disp_files = {
            'disp_x': f"*{args.prefix}*disp_x.npy",
            'disp_y': f"*{args.prefix}*disp_y.npy",
            'disp_z': f"*{args.prefix}*disp_z.npy",
        }

        success = True
        for key, pattern in disp_files.items():
            disp_file, err = find_unique_file(args.input_dir, pattern)
            if err:
                print(f"Skipping {key}: {err}")
                success = False
                continue

            disp_data = np.load(disp_file)
            save_nifti(disp_data, affine_matrix, args.output_dir, args.prefix, key)

        if not success:
            print("Note: One or more displacement files were not found and skipped.")

    print("\nSaving process completed.")


if __name__ == "__main__":
    main()
