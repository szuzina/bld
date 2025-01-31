import numpy as np
from scipy.spatial.distance import cdist

from bld.metrics import MSICalculator


class TraditionalMetricsCalculator:
    """
    Calculate the traditional metrics for a selected slice.

    Args:
        msi_calc: the MSICalculator class used for computing MSI
        slice_mask_ref: the reference mask of the current slice (all slice, not just one contour)
        slice_mask_test: the test mask of the current slice (all slice, not just one contour)

    Returns:
        dice: Dice index value
        jaccard: Jaccard index value
        Hausdorff: Hausdorff distance value
    """

    def __init__(self, msi_calc: MSICalculator, slice_mask_ref: np.ndarray[int], slice_mask_test: np.ndarray[int]):
        self.msicalc = msi_calc
        self.slice_mask_r = slice_mask_ref
        self.slice_mask_t = slice_mask_test

        self.dice = self.find_dice()
        self.jaccard = self.find_jaccard()
        self.hausdorff = self.find_max_hausdorff()

    def find_jaccard(self):
        """
        Calculates Jaccard index on a mask of one slice.
        """
        binary_array1 = np.where(self.slice_mask_r != 0, 1, 0)
        binary_array2 = np.where(self.slice_mask_t != 0, 1, 0)
        intersection = np.logical_and(binary_array1, binary_array2)
        union = np.logical_or(binary_array1, binary_array2)
        jaccard_index = np.sum(intersection) / np.sum(union)

        return jaccard_index

    def find_dice(self):
        """
        Calculates Dice index on a mask of one slice.
        """
        binary_array1 = np.where(self.slice_mask_r != 0, 1, 0)
        binary_array2 = np.where(self.slice_mask_t != 0, 1, 0)
        intersection = np.logical_and(binary_array1, binary_array2)
        dice_index = 2 * np.sum(intersection) / (np.sum(binary_array1) + np.sum(binary_array2))

        return dice_index

    @staticmethod
    def find_hausdorff(coords1: np.ndarray[int], coords2: np.ndarray[int]):
        """
        Calculates Hausdorff distance between two contours.
        """
        distances = cdist(coords1, coords2)
        hausdorff_ab = np.max(np.min(distances, axis=1))
        hausdorff_ba = np.max(np.min(distances, axis=0))
        hausdorff_distance = max(hausdorff_ab, hausdorff_ba)

        return hausdorff_distance

    def find_max_hausdorff(self):
        """
        Calculates Hausdorff distance on a mask of one slice as the maximum of Hausdorff distances
        of individual contours.
        """
        distances = []
        for r, t in zip(self.msicalc.ref_points, self.msicalc.test_points_in_order):
            distances.append(self.find_hausdorff(coords1=r.T, coords2=t.T))
        max_hausdorff = max(distances)

        return max_hausdorff
