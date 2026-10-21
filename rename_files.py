from pathlib import Path
import re
from natsort import natsorted

'''
If the error generation output is named patientXXX ... , the DVH calculation will not work, because the patients have to be
paired with the original RTDOSE file. To solve this, we rename the perturbed segmentations.
(if just MSI is calculated, this issue will not cause problems)
'''

# Paths (adjust these to your actual directories)
ref_dir = Path("/home/zsvizi/PycharmProjects/bld/data/masks_ref")
error_dir = Path("/home/zsvizi/PycharmProjects/bld/data/segmentation_error_masks")

# 1. Get all actual reference IDs sorted naturally
ref_files = natsorted(ref_dir.glob("*.nii.gz"))
ref_ids = [f.name.split(".")[0] for f in ref_files]
print(f"Loaded {len(ref_ids)} reference IDs.")

# 2. Find all perturbed files
error_files = list(error_dir.glob("patient*_*.nii.gz"))
print(f"Found {len(error_files)} perturbed files to rename.")

# 3. Process and rename each file based on its embedded patient number
for error_file in error_files:
  old_name = error_file.name

  # Extract the number from 'patientX' using regex (e.g., 'patient45' -> 45)
  match = re.search(r"patient(\d+)", old_name)
  if match:
    patient_num = int(match.group(1))

    # Convert to 0-based index for python list (patient 1 -> index 0)
    ref_index = patient_num - 1

    if 0 <= ref_index < len(ref_ids):
      true_patient_id = ref_ids[ref_index]  # e.g., 'VS-SEG-045' (or with gap shifted)

      # Extract the rest of the filename after 'patientX_'
      # e.g., 'segmentation_error_erosion_1mm.nii.gz'
      suffix = old_name.split("_", 1)[1]

      new_name = f"{true_patient_id}_{suffix}"
      new_path = error_dir / new_name

      error_file.rename(new_path)
      print(f"Renamed: {old_name}  --->  {new_name}")
    else:
      print(
          f"Warning: Patient number {patient_num} is out of range for reference"
          f" list."
      )
  else:
    print(f"Skipping (could not find patient number): {old_name}")

print("Renaming complete!")