import os
import numpy as np
import sys

# Add bart path

import bart
from bart import cfl

def reconstruct_and_process_all(data_dir, venc_value=20):
    """
    Reconstructs images from k-space data, processes them to compute velocity components and mask,
    and saves all results as .npy files in the data_dir.
    """
    # Find all .cfl/.hdr pairs
    file_bases = [
        os.path.splitext(f)[0]
        for f in os.listdir(data_dir)
        if f.endswith('.cfl') and os.path.exists(os.path.join(data_dir, os.path.splitext(f)[0] + '.hdr'))
    ]
    if not file_bases:
        print("No .cfl/.hdr data files found in the directory. Exiting.")
        return

    for file_base in file_bases:
        cfl_path = os.path.join(data_dir, file_base)
        print(f"Reading: {cfl_path}")
        ksp = cfl.readcfl(cfl_path)

        # Infer dimensions
        num_x, num_y, num_slices = ksp.shape[0:3]
        num_cardiac_phases = ksp.shape[6]
        num_vencs = ksp.shape[11]
        print(f"Shape: {ksp.shape}, num_vencs: {num_vencs}, num_cardiac_phases: {num_cardiac_phases}")

        # Reconstruct images for all vencs and cardiac phases
        recon = np.zeros((num_x, num_y, num_slices, num_cardiac_phases, num_vencs), dtype=np.complex64)
        for venc in range(num_vencs):
            for phase in range(num_cardiac_phases):
                all_ksp_venc_phase = ksp[:, :, :, :, 0, 0, phase, 0, 0, 0, 0, venc]
                all_ksp_esprit = ksp[:, :, :, :, 0, 0, phase, 0, 0, 0, 0, 0]
                sensitivities = bart.bart(1, f"ecalib -c0 -m1 -r12:9", all_ksp_esprit)
                l1_wav_reg = 0.005
                image_l1_tv_wav = bart.bart(
                    1,
                    f"pics -R W:3:0:{l1_wav_reg} -e -i 20 -S -d5",
                    all_ksp_venc_phase,
                    sensitivities
                )
                recon[:, :, :, phase, venc] = image_l1_tv_wav.squeeze()

        # Save reconstructed images
        out_file = os.path.join(data_dir, f"DATA_{os.path.basename(file_base)}.npy")
        np.save(out_file, recon)
        print(f"Saved reconstructed image: {out_file}")

        # ---- Process velocities ----
        arr = recon  # shape: (X, Y, Z, Time, VENC_encodings)
        print(f"Processing velocities, arr shape: {arr.shape}")

        if arr.shape[-1] < 4:
            print(f"ERROR: Not enough VENC encodings to compute x, y, z velocities for {file_base}!")
            continue

        # Calculate phase differences for each component
        phase_diff_x = np.angle(arr[..., 1]) - np.angle(arr[..., 0])  # x-component
        phase_diff_y = np.angle(arr[..., 2]) - np.angle(arr[..., 0])  # y-component
        phase_diff_z = np.angle(arr[..., 3]) - np.angle(arr[..., 0])  # z-component

        # Combine phase differences into single array
        phase_diffs = np.stack([phase_diff_x, phase_diff_y, phase_diff_z], axis=-1)

        # Convert phase differences to velocities using constant VENC
        velocities = (phase_diffs / np.pi) * venc_value  # Shape: (X, Y, Z, Time, 3)

        # Create mask based on magnitude data
        mag_data = np.abs(arr[..., 0])  # Use reference magnitude
        mask = mag_data > 0.1 * np.max(mag_data)  # Adjust threshold as needed

        # Apply mask to velocities
        velocities = velocities * mask[..., np.newaxis]

        # Save velocity components in cm/s
        name = os.path.splitext(os.path.basename(out_file))[0]
        np.save(os.path.join(data_dir, f'{name}_velocity_x.npy'), velocities[..., 0])
        np.save(os.path.join(data_dir, f'{name}_velocity_y.npy'), velocities[..., 1])
        np.save(os.path.join(data_dir, f'{name}_velocity_z.npy'), velocities[..., 2])
        np.save(os.path.join(data_dir, f'mask_{name}.npy'), mask)
        print(f"Processed and saved velocity components and mask for {name}")
        

# Example usage:
# reconstruct_and_process_all("/path/to/your/data_dir", venc_value=20)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reconstruct and process 4D flow MRI data.")
    parser.add_argument("data_dir", type=str, help="Path to the data directory containing .cfl/.hdr files.")
    parser.add_argument("--venc", type=float, default=20, help="VENC value in cm/s.")
    args = parser.parse_args()
    reconstruct_and_process_all(args.data_dir, venc_value=args.venc)
