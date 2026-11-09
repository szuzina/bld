import os
import glob
import SimpleITK as sitk

'''
The segmentations are saved differently in the nifti files than in the DICOM files.
The nnUNet predictions use the grid of the T1 images, whereas the reference segmentations are nifti files.
This code modifies the test segmentations (or the perturbed segmentations) to align with the reference segmentations.
'''

# --- BEÁLLÍTÁSOK ---
# Ahová az nnU-Net lementette a predikciókat (512x512)
TEST_SEG_DIR = "/home/zsvizi/PycharmProjects/bld/data/segmentation_error_masks"

# Ahol az eredeti, érintetlen referenciák vannak (448x448)
REF_SEG_DIR = "/home/zsvizi/PycharmProjects/bld/data/masks_ref"

# Ide mentjük a referenciához igazított (448x448) teszt predikciókat
OUTPUT_DIR = "/home/zsvizi/PycharmProjects/bld/data/adjusted_masks_test"

def resample_predictions_to_reference():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    test_files = glob.glob(os.path.join(TEST_SEG_DIR, "*.nii.gz"))
    print(f"Összesen {len(test_files)} predikció illesztése indul a referenciákhoz...\n")

    for test_path in test_files:
        basename = os.path.basename(test_path)
        
        # Kinyerjük a tiszta 3 jegyű ID-t a fájlnévből a pontos párosításhoz
        # (Kezeli a VS-230, VS-SEG-230, stb. formátumokat is)
        #patient_num = "".join(filter(str.isdigit, basename))
        # 'VS-SEG-200' (everything before the first underscore), then removes 'VS-SEG-'
        # (for the segmentation_error_masks)
        patient_num = basename.split("_")[0].replace("VS-SEG-", "")
        
        # A referencia fájl útvonala
        ref_path = os.path.join(REF_SEG_DIR, f"VS-SEG-{patient_num}.nii.gz")
        out_path = os.path.join(OUTPUT_DIR, basename)
        
        if os.path.exists(ref_path):
            try:
                # 1. Beolvasás
                test_seg = sitk.ReadImage(test_path)
                ref_seg = sitk.ReadImage(ref_path)
                
                # 2. Resampling: A teszt predikciót a referencia maszk terére húzzuk
                resampled_test = sitk.Resample(
                    test_seg,                   # A transzformálandó kép (predikció)
                    ref_seg,                    # A cél tér (eredeti referencia maszk)
                    sitk.Transform(),           # Fizikai térbeli illesztés
                    sitk.sitkNearestNeighbor,   # Maszkhoz kötelező interpoláció
                    0.0,
                    test_seg.GetPixelID()
                )
                
                # 3. Mentés
                sitk.WriteImage(resampled_test, out_path)
                print(f"[VS-{patient_num}] KÉSZ: Teszt predikció a referenciához igazítva.")
                
            except Exception as e:
                print(f"[VS-{patient_num}] HIBA történt: {e}")
        else:
            print(f"[VS-{patient_num}] HIÁNYZIK a referencia maszk: {ref_path}")

if __name__ == "__main__":
    resample_predictions_to_reference()