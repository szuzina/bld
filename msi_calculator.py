import numpy as np
import pandas as pd

from bld_calculator import BLDCalculator
from distance_calculator import DistanceCalculator


class MSICalculator:
    def __init__(self, il, ol, test_points, ref_points):
        self.test_points = test_points
        self.ref_points = ref_points
        self.il = il
        self.ol = ol

        self.test_points_in_order = self.pair_contours()
        self.msi = []

    def pair_contours(self):
        # pair the test and reference contours on a slice
        # we consider the closest center of mass

        def func_find_com(array):
            com = array.mean(axis=1)
            return com

        test_coms = list(map(func_find_com, self.test_points))
        ref_coms = list(map(func_find_com, self.ref_points))

        test_points_in_order = []
        indices = []
        for c in ref_coms:
            res = [np.linalg.norm(x - c) for x in test_coms]
            p = min(res)
            test_points_in_order.append(self.test_points[res.index(p)])
            indices.append(res.index(p))

        # check for duplicates
        if check_duplicate(indices):
            print("The ordering is not correct, there is duplicate in the list.")
            print("This means that the COM of more than one reference contour has the same closest test COM.")

        return test_points_in_order

    def run(self):
        for t in self.test_points_in_order:
            self.msi.append(self.run_for_single_contour(t=t))

    def run_for_single_contour(self, t):
        test_contour = t
        reference_contour = self.ref_points[self.test_points_in_order.index(t)]

        points_test_corrected = move_coms(c_ref=reference_contour,
                                          c_test=test_contour)

        dist_calc = DistanceCalculator(reference_contour=reference_contour,
                                       test_contour=points_test_corrected)
        dist_calc.run()

        bld_calc = BLDCalculator(dist_calc=dist_calc, test_points=test_contour)
        bld_calc.run()

        msi = self.calculate_msi(final_bld=bld_calc.final_bld)

        return msi

    def calculate_msi(self, final_bld):
        """
        Calculates the value of MSI for the current slice.
        Parameters:
        final_bld: numpy array with the final values of BLD
        il: the user-defined inside level value
        ol: the user-defined outside level value
        Returns:
        msi: the value of MSI of the current slice
        """
        mcf_inside = pd.DataFrame(final_bld, columns=['corr.BLD'])
        mcf_outside = pd.DataFrame(final_bld, columns=['corr.BLD'])
        mcf_inside['WF value'] = MSICalculator.weight_function(
            d=mcf_inside.loc[mcf_inside['corr.BLD'] < 0],
            l=self.il)
        mcf_outside['WF value'] = MSICalculator.weight_function(
            d=mcf_outside.loc[mcf_outside['corr.BLD'] > 0],
            l=self.ol)
        msi = 1 / len(final_bld) * (
                mcf_inside['WF value'].sum() + mcf_outside['WF value'].sum()
        )
        return msi

    @staticmethod
    def weight_function(d: float, l: float):
        """
        Defines the weight function.
        Parameters:
        d: the distance for which we calculate the weight function
        l: the level parameter of the weight function
        Returns:
        wf: the value of the weight function
        """
        wf = np.exp(-d ** 2 / (2 * (10 / l) ** 2))
        return wf


def check_duplicate(items):
    hash_bucket = set()
    for item in items:
        if item in hash_bucket:
            return True
        hash_bucket.add(item)
    return False


def move_coms(c_ref, c_test):
    """
    Moves the test contour to align the center of mass with the COM of the reference contour.
    Returns: the moved test contour.
    """

    com_ref = c_ref.mean(axis=1)
    com_test = c_test.mean(axis=1)
    move_vector = (com_ref - com_test).reshape((2, 1))
    c_test_corrected = np.add(c_test, move_vector)

    return c_test_corrected
