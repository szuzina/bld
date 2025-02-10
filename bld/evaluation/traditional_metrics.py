import numpy as np
from scipy.spatial.distance import cdist

from bld.data import DataLoader
from bld.metrics import MSICalculator


class TraditionalMetricsCalculator:
    """
    Calculate the traditional metrics for a selected slice.

    Args:
        slice_mask_ref: the reference mask of the current slice (all slice, not just one contour)
        slice_mask_test: the test mask of the current slice (all slice, not just one contour)

    Returns:
        dice: Dice index value
        jaccard: Jaccard index value
        Hausdorff: Hausdorff distance value
    """

    def __init__(self,
                 points_test: np.ndarray[int],
                 points_ref: np.ndarray[int],
                 slice_mask_ref: np.ndarray[int],
                 slice_mask_test: np.ndarray[int]):
        self.slice_mask_r = slice_mask_ref
        self.slice_mask_t = slice_mask_test

        self.points_ref = points_ref
        self.points_test = points_test

        self.dice = self.find_dice()
        self.jaccard = self.find_jaccard()
        self.hausdorff = self.find_max_hausdorff()

    def find_jaccard(self):
        """
        Calculates Jaccard index on a mask of one slice.
        """
        if self.slice_mask_r.any() and self.slice_mask_t.any():
            binary_array1 = np.where(self.slice_mask_r != 0, 1, 0)
            binary_array2 = np.where(self.slice_mask_t != 0, 1, 0)
            intersection = np.logical_and(binary_array1, binary_array2)
            union = np.logical_or(binary_array1, binary_array2)
            jaccard_index = np.sum(intersection) / np.sum(union)
        else:
            jaccard_index = 0

        return jaccard_index

    def find_dice(self):
        """
        Calculates Dice index on a mask of one slice.
        """
        if self.slice_mask_r.any() and self.slice_mask_t.any():
            binary_array1 = np.where(self.slice_mask_r != 0, 1, 0)
            binary_array2 = np.where(self.slice_mask_t != 0, 1, 0)
            intersection = np.logical_and(binary_array1, binary_array2)
            dice_index = 2 * np.sum(intersection) / (np.sum(binary_array1) + np.sum(binary_array2))
        else:
            dice_index = 0

        return dice_index

    @staticmethod
    def find_hausdorff(coords1: np.ndarray[int], coords2: np.ndarray[int]):
        """
        Calculates Hausdorff distance between two contours.
        """
        if np.any(coords1) and np.any(coords2):
            distances = cdist(coords1, coords2)
            hausdorff_ab = np.max(np.min(distances, axis=1))
            hausdorff_ba = np.max(np.min(distances, axis=0))
            hausdorff_distance = max(hausdorff_ab, hausdorff_ba)
        else:
            hausdorff_distance = np.inf

        return hausdorff_distance

    def find_max_hausdorff(self):
        """
        Calculates Hausdorff distance on a mask of one slice as the maximum of Hausdorff distances
        of individual contours.
        """
        distances = []
        for r, t in zip(self.points_ref, self.points_test):  # itt igazából az msicalc.test_points_in_order kellene
            if len(self.points_ref) > 0 and len(self.points_test) > 0:
                distances.append(self.find_hausdorff(coords1=r.T.reshape(-1, 2), coords2=t.T.reshape(-1, 2)))
                # reshape (2,) to 2D array for the cdist
        if len(distances) > 0:
            max_hausdorff = max(distances)
        else:
            max_hausdorff = np.inf

        return max_hausdorff
