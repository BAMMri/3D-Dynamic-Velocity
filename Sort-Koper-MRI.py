# import os
# import shutil
# import re
# import pydicom
# from pydicom.errors import InvalidDicomError
#
# # === CONFIG ===
# input_folder = "/Volumes/ExtremeSSD/AA_PhD_Projects/MRI_Koper/DICOM/25081920/ALL"
# output_folder = "/Volumes/ExtremeSSD/AA_PhD_Projects/MRI_Koper/DICOM/25081920/all_sorted"
# move_files = False   # True -> move, False -> copy
# dry_run = False       # True -> don't actually copy/move; just show what would happen
# verbose = True
# # ==============
#
# os.makedirs(output_folder, exist_ok=True)
#
# def make_safe(name, maxlen=150):
#     if not name:
#         return "Unknown"
#     # replace weird chars with underscore, collapse whitespace into single underscore
#     s = re.sub(r'[^A-Za-z0-9 _\.-]', '_', str(name))
#     s = re.sub(r'\s+', '_', s).strip('_')
#     return s[:maxlen]
#
# processed = 0
# skipped = 0
# mapping = {}   # folder_name -> count
# warnings = []
#
# for root, _, files in os.walk(input_folder):
#     for fname in files:
#         fpath = os.path.join(root, fname)
#
#         # skip non-files / hidden files quickly
#         if not os.path.isfile(fpath) or fname.startswith('.'):
#             continue
#
#         ds = None
#         try:
#             # try reading normally (fast)
#             ds = pydicom.dcmread(fpath, stop_before_pixels=True)
#         except InvalidDicomError:
#             # try force-read (some anonymized / non-standard dicoms)
#             try:
#                 ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
#                 warnings.append(f"Force-read (non-standard) DICOM: {fpath}")
#             except Exception as e:
#                 if verbose:
#                     print(f"Skipping non-DICOM: {fpath} -> {e}")
#                 skipped += 1
#                 continue
#         except Exception as e:
#             if verbose:
#                 print(f"Skipping (read error): {fpath} -> {e}")
#             skipped += 1
#             continue
#
#         # Extract attributes in fallback order
#         seq = getattr(ds, "SequenceName", None)
#         prot = getattr(ds, "ProtocolName", None)
#         desc = getattr(ds, "SeriesDescription", None)
#         series_num = getattr(ds, "SeriesNumber", None)
#         series_uid = getattr(ds, "SeriesInstanceUID", None)
#
#         # build a human-readable label with fallbacks and small disambiguator
#         label_base = seq or prot or desc or ""
#         if not label_base:
#             label_base = "Unknown"
#         # append series number to disambiguate
#         if series_num is not None:
#             label_base = f"{label_base}_SN{series_num}"
#         elif series_uid:
#             label_base = f"{label_base}_UID{(series_uid[:8])}"
#
#         safe_name = make_safe(label_base)
#         target_folder = os.path.join(output_folder, safe_name)
#
#         if dry_run:
#             action = "MOVE" if move_files else "COPY"
#             print(f"[DRY] {action} {fpath} -> {target_folder}")
#         else:
#             os.makedirs(target_folder, exist_ok=True)
#             try:
#                 if move_files:
#                     shutil.move(fpath, os.path.join(target_folder, fname))
#                 else:
#                     shutil.copy2(fpath, target_folder)
#             except Exception as e:
#                 print(f"Failed to copy/move {fpath} -> {target_folder}: {e}")
#                 skipped += 1
#                 continue
#
#         processed += 1
#         mapping.setdefault(safe_name, 0)
#         mapping[safe_name] += 1
#
# # Summary
# print("\n=== SUMMARY ===")
# print(f"Processed (would process if dry_run): {processed}")
# print(f"Skipped: {skipped}")
# print(f"Distinct target folders: {len(mapping)}")
# for k, v in sorted(mapping.items(), key=lambda kv: -kv[1]):
#     print(f"  {k}: {v}")
# if warnings:
#     print("\nWarnings (force-reads or non-standard DICOMs):")
#     for w in warnings[:20]:
#         print("  -", w)
#
# # ============== # ============== # ============== # ==============
# ## Sort Subjects Apaprt
## ==============# ==============# ==============# ==============# ==============
import os
import shutil
import pydicom
from datetime import datetime

# === CONFIG ===
sorted_folder = "/Volumes/ExtremeSSD/AA_PhD_Projects/MRI_Koper/DICOM/25081920/all_sorted"
output_folder = "/Volumes/ExtremeSSD/AA_PhD_Projects/MRI_Koper/DICOM/25081920/BEAT_FQ-split"
move_folders = True   # True -> move, False -> copy
dry_run = False         # True -> only print what would happen
# ==============

os.makedirs(output_folder, exist_ok=True)

def get_datetime_from_folder(folder):
    """Read acquisition datetime from the first DICOM in a folder."""
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
        except:
            continue

        if hasattr(ds, "AcquisitionDateTime"):
            dt = ds.AcquisitionDateTime
            try:
                return datetime.strptime(dt, "%Y%m%d%H%M%S")
            except:
                pass
        else:
            date = getattr(ds, "AcquisitionDate", "")
            time = getattr(ds, "AcquisitionTime", "")
            if date and time:
                try:
                    return datetime.strptime(date + time.split('.')[0], "%Y%m%d%H%M%S")
                except:
                    pass
    return None

# collect all BEAT_FQ folders
beat_fq_folders = []
for sub in os.listdir(sorted_folder):
    if sub.startswith("BEAT_FQ"):
        folder_path = os.path.join(sorted_folder, sub)
        if os.path.isdir(folder_path):
            dt = get_datetime_from_folder(folder_path)
            beat_fq_folders.append((sub, folder_path, dt))

# sort by acquisition datetime
beat_fq_folders.sort(key=lambda x: x[2] or datetime.min)

# split evenly between Subject1 and Subject2
half = len(beat_fq_folders) // 2
for idx, (name, path, dt) in enumerate(beat_fq_folders):
    subject = "Subject1" if idx < half else "Subject2"
    target = os.path.join(output_folder, subject, name)
    if dry_run:
        print(f"[DRY] {path} -> {target} ({dt})")
    else:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if move_folders:
            shutil.move(path, target)
        else:
            shutil.copytree(path, target)

print(f"\nTotal BEAT_FQ folders: {len(beat_fq_folders)}")
print(f"Assigned {half} to Subject1 and {len(beat_fq_folders) - half} to Subject2")
