import cv2 as cv
import numpy as np
import pandas as pd


class BLDCalculator:
    def __init__(self, dist_calc, test_points):
        self.distance_df = dist_calc.distance_table
        self.reference_points = dist_calc.reference_contour
        self.test_corrected_points = dist_calc.test_contour
        self.test_points = test_points

        self.visualization_data = None
        self.dist_bld = None
        self.dist_bld_signed = None
        self.final_bld = None
        self.location = None
        self.paired_test_points_moved_back = None

    def run(self):
        self.calculate_bld()
        self.calculate_signed_distances()
        self.calculate_corrected_bld()

    def calculate_bld(self):
        """
        Calculates the BLD to each reference point.
        Parameters:
        df: Pandas Dataframe with the pairwise distances
        Returns:
        bmaxd_indexek: the index of the point to with the BLD is calculated
        fmind: the FminD values for each reference point
        bmaxd: the BMaxD values (where it exists)
        bld: the BLD values for each reference point
        """

        # the columns correspond to the test points
        # the column minimum is the minimum distance from the reference points
        oszlop_min_indexek = np.argmin(self.distance_df.values, axis=0)

        tabla = oszlop_min_indexek.reshape((-1, 1)) == \
            np.arange(0, self.distance_df.shape[0], 1).reshape((1, -1))
        # filtering the reference points (rows) to which there exist column minimum
        szures_sor_index_van_oszlop_minimum = tabla.sum(axis=0) > 0
        # the BMaxD is the maximum of the column minimums
        # we find the indices of BMaxD distances
        bmaxd_indexek = np.arange(
            0, self.distance_df.shape[0], 1
        )[szures_sor_index_van_oszlop_minimum]
        # if there is no BMaxD, then FMinD is used
        # we find the indices of FMinD distances
        fmind_indexek = np.arange(
            0, self.distance_df.shape[0], 1
        )[~szures_sor_index_van_oszlop_minimum]

        bld = np.zeros((self.distance_df.shape[0],))

        # we find the values of FMinD distances
        bld[fmind_indexek] = self.distance_df.min(axis=1).iloc[fmind_indexek]

        tabla_2 = (oszlop_min_indexek.reshape((1, -1)) ==
                   np.arange(0, self.distance_df.shape[0], 1).reshape((-1, 1)))

        # we find the values of BMaxD distances
        bmaxd = np.max(tabla_2 * self.distance_df.min(axis=0).values.reshape((-1,)),
                       axis=1
                       )[bmaxd_indexek]

        fmind = self.distance_df.min(axis=1).iloc[bmaxd_indexek].values

        # BLD is the maximum of BMaxD and FMinD
        bld_bmaxd_letezik = np.maximum(bmaxd, fmind)
        # if BMaxD does not exist, then BLD = FMinD
        bld[bmaxd_indexek] = bld_bmaxd_letezik
        self.visualization_data = {
            "bmaxd_indexek": bmaxd_indexek,
            "fmind": fmind,
            "bmaxd": bmaxd
        }
        self.dist_bld = bld

    def calculate_signed_distances(self):
        """
        Finds if a test point is inside or outside the reference contour and gives signed BLD.
        Parameters:
        test_corrected: the moved test contour (the COM of the test contour is aligned with
            the COM of the reference contour)
        ref: the reference point numpy array
        bld: list of the BLD values (in order of the reference point indices)
        Returns:
        loc: 1 if the test point is inside, 0 if on the reference contour, -1 if outside
        bld_signed: the list of signed BLD distances
        """

        # integers are needed for the PolygonTest function
        # test_cor = self.test_corrected_points.astype(int)

        # polygon is a list of tuples representing the vertices of the polygon
        polygon = []
        for i in range(self.test_corrected_points.shape[1]):
            px = self.test_corrected_points.T[i][0]
            py = self.test_corrected_points.T[i][1]
            polygon.append((px, py))
        # print('polygon', polygon)
        poly = np.array(polygon, dtype=np.float32)
        # print('poly', poly)

        location = []
        # print("Point:", pt)
        # print("Type of pt:", type(pt))
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
        # print(loc)

    def calculate_corrected_bld(self):
        """
        Calculates the BLD distances after moving back the test contour to the original location.
        Parameters:
        df: Pandas Dataframe with the pairwise distances
        bld: list of the BLD values (in order of the reference point indices)
        loc: 1 if the test point is inside, 0 if on the reference contour, -1 if outside
        ref: the reference point numpy array
        test: the test point numpy array
        test_corrected: the moved test contour (the COM of the test contour is aligned with
            the COM of the reference contour)
        Returns:
        final_bld: numpy array of the BLD values calculated after moving back the test contour
        paired_test_points_moved_back: numpy array containing the test points
            which are paires of reference contour points based on BLD
        """
        sor_bld_indexek = np.zeros(len(self.distance_df))
        for i in range(len(self.distance_df)):
            # we assign the pairs to the reference contour points
            sor_bld_indexek[i] = np.argwhere(
                self.distance_df.iloc[i].values == self.dist_bld[i])

        test_df = pd.DataFrame(self.test_corrected_points.T, columns=['x', 'y'])
        list_sor_bld_indexek = sor_bld_indexek.tolist()
        test_paired = test_df.iloc[list_sor_bld_indexek]

        # we move back the test points to the original location
        # we only use the test points, which are paired with reference points
        # these test points are in the same order as the reference points
        # (the pair of the ith reference point is the ith test point

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
