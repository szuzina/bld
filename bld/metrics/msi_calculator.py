from typing import Union, List

import numpy as np
import pandas as pd

import bld.metrics as bldm


class MSICalculator:
    """
    Calculates the MSI values for all slices corresponding to a patient.

    Args:
        il: inside penalty level
        ol: outside penalty level
        test_points: the test points array (coordinates)
        ref_points: the reference points array (coordinates)

    Returns:
        msi: the calculated MSI values

    """
    def __init__(self, il: float, ol: float, test_points: np.ndarray, ref_points: np.ndarray):
        self.test_points = test_points
        self.ref_points = ref_points
        self.il = il
        self.ol = ol

        self.test_points_in_order = self.pair_contours()
        self.msi: list = []

    def pair_contours(self) -> List:
        """
        Pair the test and reference contours on a slice based on the closest center of mass
        """

        def func_find_com(array):
            if array.ndim == 1:  # Check if the array is 1D
                return array
            else:
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

        return test_points_in_order

    def run(self):
        for r, t in zip(self.ref_points, self.test_points_in_order):
            self.msi.append(self.run_for_single_contour(r=r, t=t))

    def run_for_single_contour(self, r: np.ndarray, t: np.ndarray) -> pd.Series:
        """
        Calculate MSI for a single contour.
        """
        test_contour = t
        reference_contour = r

        points_test_corrected = move_coms(c_ref=reference_contour,
                                          c_test=test_contour)

        dist_calc = bldm.DistanceCalculator(
            reference_contour=reference_contour,
            test_contour=points_test_corrected)
        dist_calc.run()

        bld_calc = bldm.BLDCalculator(dist_calc=dist_calc, test_points=test_contour)
        bld_calc.run()

        msi = self.calculate_msi(final_bld=bld_calc.final_bld)

        return msi

    def calculate_msi(self, final_bld: list) -> pd.Series:
        """
        Calculates the value of MSI for the current slice.
        """
        mcf_inside = pd.DataFrame(data=final_bld, columns=['corr.BLD'])
        mcf_outside = pd.DataFrame(data=final_bld, columns=['corr.BLD'])
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
    def weight_function(d: Union[float, pd.DataFrame],
                        l: Union[float, pd.DataFrame]) -> Union[float, pd.DataFrame]:
        """
        Defines the weight function.

        Args:
            d: the distance for which we calculate the weight function
            l: the level parameter of the weight function
        Returns:
            wf: the value of the weight function
        """

        wf = np.exp(-d ** 2 / (2 * (10 / l) ** 2))

        return wf


def check_duplicate(items):
    """
    Check if the pairing is not correct i.e. there is duplicate in the list.
    """
    hash_bucket = set()
    for item in items:
        if item in hash_bucket:
            return True
        hash_bucket.add(item)

    return False


def move_coms(c_ref: np.ndarray, c_test: np.ndarray) -> np.ndarray:
    """
    Moves the test contour to align the center of mass with the COM of the reference contour.

    Args:
        c_ref: the reference contour points (coordinates)
        c_test: the test contour points (coordinates)

    Returns:
        c_test_corrected: the moved test contour.
    """

    com_ref = c_ref.mean(axis=1)
    com_test = c_test.mean(axis=1)
    move_vector = (com_ref - com_test).reshape((2, 1))
    c_test_corrected = np.add(c_test, move_vector)

    return c_test_corrected
