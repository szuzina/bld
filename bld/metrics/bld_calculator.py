import cv2 as cv
import numpy as np
import pandas as pd

from bld.metrics import DistanceCalculator


class BLDCalculator:
    """
    Calculates the BLD and corresponding calculations.

    Args:
        dist_calc: DistanceCalculator class with the pairwise distances
        test_points: the test point's numpy array

    Returns:
        visualization_data: contains bmaxd_indices, fmind and bmaxd, which are necessary for visualization
        dist_bld: BLD values calculated after aligning the reference and test COMs
        dist_bld_signed: signed BLD values (inside or outside location)
        final_bld: numpy array of the BLD values calculated after moving back the test contour
        location: 1 if the test point is inside, 0 if on the reference contour, -1 if outside
        paired_test_points_moved_back: numpy array containing the test points
            which are pairs of reference contour points based on BLD
    """

    def __init__(self, dist_calc: DistanceCalculator, test_points: np.ndarray[int]):
        self.distance_df = dist_calc.distance_table
        self.reference_points = dist_calc.reference_contour
        self.test_corrected_points = dist_calc.test_contour
        self.test_points = test_points

        self.visualization_data: dict = dict()
        self.dist_bld: list = []
        self.dist_bld_signed: list = []
        self.final_bld: list = []
        self.location: list = []
        self.paired_test_points_moved_back: np.ndarray = np.array([], dtype=np.int_)

    def run(self):
        self.calculate_bld()
        self.calculate_signed_distances()
        self.calculate_corrected_bld()

    def calculate_bld(self):
        """
        Calculates the BLD to each reference point.
        """

        # the columns correspond to the test points
        # the column minimum is the minimum distance from the reference points
        column_min_indices = np.argmin(self.distance_df.values, axis=0)

        table = column_min_indices.reshape((-1, 1)) == \
            np.arange(0, self.distance_df.shape[0], 1).reshape((1, -1))
        # filtering the reference points (rows) to which there exist column minimum
        filter_row_index_where_exists_column_min = table.sum(axis=0) > 0
        # the BMaxD is the maximum of the column minimums
        # we find the indices of BMaxD distances
        bmaxd_indices = np.arange(
            0, self.distance_df.shape[0], 1
        )[filter_row_index_where_exists_column_min]
        # if there is no BMaxD, then FMinD is used
        # we find the indices of FMinD distances
        fmind_indices = np.arange(
            0, self.distance_df.shape[0], 1
        )[~filter_row_index_where_exists_column_min]

        bld = np.zeros((self.distance_df.shape[0],))

        # we find the values of FMinD distances
        bld[fmind_indices] = self.distance_df.min(axis=1).iloc[fmind_indices]

        table_2 = (column_min_indices.reshape((1, -1)) ==
                   np.arange(0, self.distance_df.shape[0], 1).reshape((-1, 1)))

        # we find the values of BMaxD distances
        bmaxd = np.max(table_2 * self.distance_df.min(axis=0).values.reshape((-1,)),
                       axis=1
                       )[bmaxd_indices]

        fmind = self.distance_df.min(axis=1).iloc[bmaxd_indices].values

        # BLD is the maximum of BMaxD and FMinD
        bld_bmaxd_exists = np.maximum(bmaxd, fmind)
        # if BMaxD does not exist, then BLD = FMinD
        bld[bmaxd_indices] = bld_bmaxd_exists
        self.visualization_data = {
            "bmaxd_indices": bmaxd_indices,
            "fmind": fmind,
            "bmaxd": bmaxd
        }
        self.dist_bld = bld

    def calculate_signed_distances(self):
        """
        Finds if a test point is inside or outside the reference contour and gives signed BLD.
        """

        # polygon is a list of tuples representing the vertices of the polygon
        polygon = []
        for i in range(self.test_corrected_points.shape[1]):
            px = self.test_corrected_points.T[i][0]
            py = self.test_corrected_points.T[i][1]
            polygon.append((px, py))
        poly = np.array(polygon, dtype=np.float32)

        location = []
        for i in range(self.reference_points.shape[1]):
            # 1: inside, 0: on the contour, -1: outside
            pt = (np.float32(self.reference_points.T[i][0]),
                  np.float32(self.reference_points.T[i][1]))
            location.append(
                cv.pointPolygonTest(poly,
                                    pt,
                                    False))
        loc = np.array(location)
        bld_signed = np.multiply(loc, self.dist_bld)
        self.dist_bld_signed = bld_signed
        self.location = loc

    def calculate_corrected_bld(self):
        """
        Calculates the BLD distances after moving back the test contour to the original location.
        """

        row_bld_indices = np.zeros(len(self.distance_df))
        for i in range(len(self.distance_df)):
            # we assign the pairs to the reference contour points
            # Select the first index if np.argwhere returns a 2D array
            idx = np.argwhere(self.distance_df.iloc[i].values == self.dist_bld[i])
            if idx.ndim > 1:
                idx = idx[0]
            row_bld_indices[i] = idx

        test_df = pd.DataFrame(self.test_corrected_points.T, columns=['x', 'y'])
        list_row_bld_indices = row_bld_indices.tolist()
        test_paired = test_df.iloc[list_row_bld_indices]

        # we move back the test points to the original location
        # we only use the test points, which are paired with reference points
        # these test points are in the same order as the reference points
        # (the pair of the ith reference point is the ith test point)

        test_points_paired = test_paired.to_numpy()

        com_ref = self.reference_points.T.mean(axis=0)
        com_test = self.test_points.T.mean(axis=0)
        move_vector = com_ref - com_test

        paired_test_points_moved_back = test_points_paired - move_vector

        final_bld = np.multiply(
            self.location,
            np.sqrt(((self.reference_points.T - paired_test_points_moved_back) ** 2).sum(axis=1))
        )  # calculate the distance of the points in each pair
        self.final_bld = final_bld
        self.paired_test_points_moved_back = paired_test_points_moved_back
