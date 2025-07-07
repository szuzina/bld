import numpy as np
import pandas as pd
from Typing import List


def calculate_bld_distribution(bmaxd: np.ndarray, fmind: np.ndarray,
                               bmaxd_indices: list, dt: pd.DataFrame) -> int, np.ndarray:
    """
    Calculates descriptive parameters of the BLD.
      Parameters:
          bmaxd: backward maximal distance
          fmind: forward mininum distance
          bmaxd_indices: the list of the indices of the point to which the BLD is calculated
          dt: Pandas Dataframe containing the pairwise distances between the test and reference points
      Returns:
          bld_average: single value with the average of BLD
          number_of_points_BMaxD_is_bigger_than_FMinD: the number of points where BMaxD was used to get BLD
                (instead of FMinD)
          bmd_length: the number of different BMaxD values from which the maximal is chosen
  """
    number_of_points_bmaxd_is_bigger_than_fmind = \
        bmaxd_indices[fmind - bmaxd < 0].shape[0]

    # how many different point has the same BMaxD value, corresponding to one reference point?
    bmd = []
    bmd_length = np.zeros(len(bmaxd_indices))
    for i in range(len(bmaxd_indices)):
        r = np.argwhere(dt.iloc[bmaxd_indices[i]].values == bmaxd[i])
        bmd.append(r)
        bmd_length[i] = len(r)

    return number_of_points_bmaxd_is_bigger_than_fmind, bmd_length


def calculate_ldp(dt: pd.DataFrame, loc: list, bld: list) -> np.ndarray, List:
    """
    Calculates local distance profile.

    Args:
        dt: Pandas Dataframe containing the pairwise distances between the test and reference points
        loc: the location of the test points compared to the reference contour (inside or outside:
                1 if the test point is inside, 0 if on the reference contour, -1 if outside)
        bld: list of the BLD values (in order of the reference point indices)

    Returns:
        fmins_signed: forward minimal distances, negative if outside, positive if inside the reference contour
        diff: the difference between bld and signed forward minimal distances
    """
    fmins = dt.min(axis=1)
    fmins_signed = np.multiply(loc, fmins)
    diff = bld - fmins

    return fmins_signed, diff
