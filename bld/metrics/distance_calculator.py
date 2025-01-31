import numpy as np
import pandas as pd


class DistanceCalculator:
    """
    Calculates the pairwise distances between reference and test contour points.

    Args:
        reference_contour: the np array of the reference contour points (coordinates)
        test_contour: the np array of the test contour points (coordinates)

    Returns:
        pairwise_distances: the Euclidean distance between the reference and test contour points
        distance_table: table of the pairwise distances
    """

    def __init__(self, reference_contour: np.ndarray[int], test_contour: np.ndarray[int]):
        self.reference_contour = reference_contour
        self.test_contour = test_contour

        self.pairwise_distance: np.ndarray = np.array([], dtype=np.float64)
        self.distance_table: pd.DataFrame = pd.DataFrame()

    def run(self):
        self.pairwise_distance = self.find_pairwise_dist()
        self.distance_table = self.create_table()

    def find_pairwise_dist(self):
        """
        Finds the pairwise distances between two contours (i.e. a reference and a test contour).
        """

        c_ref = self.reference_contour.T
        c_test = self.test_contour.T

        n_1 = c_ref.shape[0]
        v1v2 = c_ref @ c_test.T
        v12 = np.sum(c_ref ** 2, axis=1).reshape((n_1, 1))
        v22 = np.sum(c_test ** 2, axis=1)
        pairwise_dist = np.sqrt(v12 - 2 * v1v2 + v22)        # n_2 = c_test.shape[0]

        return pairwise_dist

    def create_table(self):
        """
        Creates the table with the distances. The rows correspond to the reference points,
        the columns correspond to the test points.
        Each cell contains the Euclidean distance between the two points of the row index and column index.
        """

        cols = []
        for i in range(len(self.pairwise_distance[0, :])):
            cols.append('Ptest' + str(i))

        ind = []
        for i in range(len(self.pairwise_distance[:, 0])):
            ind.append('Pref' + str(i))

        df = pd.DataFrame(data=self.pairwise_distance, columns=cols, index=ind)

        return df
