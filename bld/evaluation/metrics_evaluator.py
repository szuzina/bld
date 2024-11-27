import glob

from natsort import natsorted
from scipy.spatial.distance import directed_hausdorff
import SimpleITK as sitk
from sklearn.metrics import jaccard_score, f1_score

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

        self.msindex = []
        self.haus = []
        self.dice = []
        self.jacc = []
        self.idx = []

    @staticmethod
    def check_contours_on_slice(test_points, ref_points):
        if len(test_points) != len(ref_points) or len(test_points) == 0 or len(ref_points) == 0:
            print("The number of test and reference contours are not equal. The slice should be evaluated manually.")
            error = True
        else:
            # Check if each array within test_points and ref_points is 2D
            for test_contour, ref_contour in zip(test_points, ref_points):
                if test_contour.ndim != 2 or ref_contour.ndim != 2:
                    print("At least one contour is not 2D. The slice should be evaluated manually.")
                    error = True
                    return error  # Return immediately if an error is found

            print("The number of test and reference contours are equal. The automatic evaluation can be continued.")
            error = False
        return error

    def find_msi_for_one_slice(self, slice_index):
        im_slice = 'slice' + str(slice_index)

        il_const = self.il  # inside level
        ol_const = self.ol  # outside level

        points_ref = self.dl.c_ref[im_slice]
        points_test = self.dl.c_test[im_slice]

        msi_calc = MSICalculator(
            il=il_const, ol=ol_const,
            ref_points=points_ref,
            test_points=points_test)
        msi_calc.run()

        print("MSI:", msi_calc.msi)

        return msi_calc.msi

    @staticmethod
    def find_traditional_metrics(mask_t_slice_np, mask_r_slice_np):
        # Hausdorff
        d1 = directed_hausdorff(mask_t_slice_np, mask_r_slice_np)[0]
        d2 = directed_hausdorff(mask_t_slice_np, mask_r_slice_np)[0]
        hausdorff_distance = max(d1, d2)
        print("Hausdorff Distance:", hausdorff_distance)

        # Dice, Jaccard
        jaccard_index = jaccard_score(mask_t_slice_np.flatten(),
                                      mask_r_slice_np.flatten(),
                                      average='micro')
        dice_coefficient = f1_score(mask_t_slice_np.flatten(),
                                    mask_r_slice_np.flatten(),
                                    average='micro')
        print("Dice Coefficient:", dice_coefficient)
        print("Jaccard Index:", jaccard_index)

        return hausdorff_distance, dice_coefficient, jaccard_index

    def evaluate(self):

        num_slices_ref = len([key for key in self.dl.c_ref if key.startswith('slice')]) # Get number of slices available in c_ref
        num_slices = min(self.mask_t.GetSize()[0], num_slices_ref) #Use minimum to avoid exceeding available slices
        
        for i in range(num_slices):
            print(i, ":")
            points_ref = self.dl.c_ref['slice' + str(i)]
            points_test = self.dl.c_test['slice' + str(i)]
            e = self.check_contours_on_slice(test_points=points_test,
                                             ref_points=points_ref)

            if not e:  # there is no error while checking the contours

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

            else:
                print("Reference or test slice is empty.")
