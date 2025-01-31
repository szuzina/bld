from bld.data.dataloader import DataLoader
from bld.evaluation.traditional_metrics import TraditionalMetricsCalculator
from bld.metrics.msi_calculator import MSICalculator


class MetricsEvaluator:
    """
    Calculates the different metrics for all the image slices of one patient.

    Args:
        patient: patient number
        data_folder: will be modified
        root_folder: will be modified
        il: inside penalty level value
        ol: outside penalty level value

    Returns:
        dl: the DataLoader class for the selected patient which contains the patient data
        num_slices: the number of slices
        msindex: MSI values
        idx: the slice indices for which MSI was calculated
        dice: Dice index values
        jacc: Jaccard index values
        haus: Hausdorff distance values
    """

    def __init__(self, patient, data_folder='data', root_folder='./', il=1, ol=1):
        self.patient = patient
        self.data_folder = data_folder
        self.root_folder = root_folder
        self.il = il
        self.ol = ol

        self.dl = DataLoader(data_folder=data_folder, patient=patient, root_folder=root_folder)

        # Get number of slices available
        num_slices_test = len([key for key in self.dl.mask_test if key.startswith('slice')])
        num_slices_ref = len([key for key in self.dl.c_ref if key.startswith('slice')])
        self.num_slices = min(num_slices_test,
                              num_slices_ref)  # Use minimum to avoid exceeding available slices

        self.msindex = []
        self.idx = []
        self.dice = []
        self.jacc = []
        self.haus = []

    @staticmethod
    def check_contours_on_slice(test_points, ref_points):
        """
        Check if the reference and test contours are compatible and have at least one element.
        """
        if len(test_points) != len(ref_points) or len(test_points) == 0 or len(ref_points) == 0:
            error = True
        else:
            # Check if each array within test_points and ref_points is 2D
            for test_contour, ref_contour in zip(test_points, ref_points):
                if test_contour.ndim != 2 or ref_contour.ndim != 2:
                    error = True
                    return error  # Return immediately if an error is found
            error = False

        return error

    def find_metrics_for_one_slice(self, slice_index):
        """
        Calculate MSI and traditional metrics for one image slice.
        """
        slice_name = 'slice' + str(slice_index)
        # Finding the MSI
        points_ref = self.dl.c_ref[slice_name]
        points_test = self.dl.c_test[slice_name]
        msi_calc = MSICalculator(
            il=self.il, ol=self.ol,
            ref_points=points_ref,
            test_points=points_test)
        msi_calc.run()

        # Finding the traditional metrics
        trad_metrics_calc = TraditionalMetricsCalculator(
                                msi_calc=msi_calc,
                                slice_mask_ref=self.dl.mask_ref[slice_name],
                                slice_mask_test=self.dl.mask_test[slice_name]
        )

        return msi_calc.msi, trad_metrics_calc.dice, trad_metrics_calc.jaccard, trad_metrics_calc.hausdorff

    def evaluate(self):
        """
        Calculate the metrics for all image slices.
        """
        for i in range(self.num_slices):
            points_ref = self.dl.c_ref['slice' + str(i)]
            points_test = self.dl.c_test['slice' + str(i)]
            is_run_correctly = self.check_contours_on_slice(
                test_points=points_test,
                ref_points=points_ref)

            if not is_run_correctly:  # there is no error while checking the contours
                m, d, j, h = self.find_metrics_for_one_slice(slice_index=i)
                self.msindex.append(m)
                self.idx.append(i)
                self.haus.append(h)
                self.dice.append(d)
                self.jacc.append(j)
