import glob

from natsort import natsorted
import numpy as np
from scipy.spatial.distance import cdist
import SimpleITK as sitk

from bld.data.dataloader import DataLoader
from bld.metrics.msi_calculator import MSICalculator


class MetricsEvaluator:
    def __init__(self, patient, data_folder='data', root_folder='./', il=1, ol=1):
        self.patient = patient
        self.data_folder = data_folder
        self.root_folder = root_folder
        self.il = il
        self.ol = ol

        self.dl = DataLoader(data_folder=data_folder, number=patient, root_folder=root_folder)
        labels_test = natsorted(glob.glob(self.root_folder + self.data_folder + "/masks_test/*"))
        labels_ref = natsorted(glob.glob(self.root_folder + self.data_folder + "/masks_ref/*"))

        self.mask_t = sitk.ReadImage(labels_test[patient - 1])
        self.mask_r = sitk.ReadImage(labels_ref[patient - 1])
        self.mask_t_np = sitk.GetArrayViewFromImage(self.mask_t)
        self.mask_r_np = sitk.GetArrayViewFromImage(self.mask_r)

        # Get number of slices available in c_ref
        num_slices_ref = len([key for key in self.dl.c_ref if key.startswith('slice')])
        self.num_slices = min(self.mask_t.GetSize()[0],
                              num_slices_ref)  # Use minimum to avoid exceeding available slices

        self.msindex = []
        self.haus = []
        self.dice = []
        self.jacc = []
        self.idx = []

    @staticmethod
    def check_contours_on_slice(test_points, ref_points):
        if len(test_points) != len(ref_points) or len(test_points) == 0 or len(ref_points) == 0:
            # print("The number of test and reference contours are not equal. The slice should be evaluated manually.")
            error = True
        else:
            # Check if each array within test_points and ref_points is 2D
            for test_contour, ref_contour in zip(test_points, ref_points):
                if test_contour.ndim != 2 or ref_contour.ndim != 2:
                    # print("At least one contour is not 2D. The slice should be evaluated manually.")
                    error = True
                    return error  # Return immediately if an error is found

            # print("The number of test and reference contours are equal. The automatic evaluation can be continued.")
            error = False
        return error

    def find_msi_for_one_slice(self, slice_index):
        points_ref = self.dl.c_ref['slice' + str(slice_index)]
        points_test = self.dl.c_test['slice' + str(slice_index)]

        msi_calc = MSICalculator(
            il=self.il, ol=self.ol,
            ref_points=points_ref,
            test_points=points_test)
        msi_calc.run()

        return msi_calc.msi

    @staticmethod
    def find_traditional_metrics(mask_t_slice_np, mask_r_slice_np):
        hausdorff_distance = find_hausdorff(mask_t_slice_np, mask_r_slice_np)
        jaccard_index = find_jaccard(mask_t_slice_np, mask_r_slice_np)
        dice_coefficient = find_dice(mask_t_slice_np, mask_r_slice_np)

        return hausdorff_distance, dice_coefficient, jaccard_index

    def evaluate(self):
        for i in range(self.num_slices):
            points_ref = self.dl.c_ref['slice' + str(i)]
            points_test = self.dl.c_test['slice' + str(i)]
            is_run_correctly = self.check_contours_on_slice(
                test_points=points_test,
                ref_points=points_ref)

            if not is_run_correctly:  # there is no error while checking the contours

                mask_t_slice_np = sitk.GetArrayViewFromImage(self.mask_t[i, :, :])
                mask_r_slice_np = sitk.GetArrayViewFromImage(self.mask_r[i, :, :])

                m = self.find_msi_for_one_slice(slice_index=i)
                self.msindex.append(m)
                self.idx.append(i)

                hd, ds, ji = self.find_traditional_metrics(
                    mask_t_slice_np=mask_t_slice_np,
                    mask_r_slice_np=mask_r_slice_np)
                self.haus.append(hd)
                self.dice.append(ds)
                self.jacc.append(ji)

def find_jaccard(array1, array2):
    binary_array1 = np.where(array1 != 0, 1, 0)
    binary_array2 = np.where(array2 != 0, 1, 0)
    intersection = np.logical_and(binary_array1, binary_array2)
    union = np.logical_or(binary_array1, binary_array2)
    jaccard_index = np.sum(intersection) / np.sum(union)
    return jaccard_index

def find_dice(array1, array2):
    binary_array1 = np.where(array1 != 0, 1, 0)
    binary_array2 = np.where(array2 != 0, 1, 0)
    intersection = np.logical_and(binary_array1, binary_array2)
    dice_index = 2 * np.sum(intersection) / (np.sum(binary_array1) + np.sum(binary_array2))
    return dice_index

def find_hausdorff(array1, array2):
    binary_array1 = np.where(array1 != 0, 1, 0)
    binary_array2 = np.where(array2 != 0, 1, 0)
    coords1 = np.argwhere(binary_array1 == 1)
    coords2 = np.argwhere(binary_array2 == 1)
    distances = cdist(coords1, coords2)
    hausdorff_AB = np.max(np.min(distances, axis=1))
    hausdorff_BA = np.max(np.min(distances, axis=0))
    hausdorff_distance = max(hausdorff_AB, hausdorff_BA)
    return hausdorff_distance
