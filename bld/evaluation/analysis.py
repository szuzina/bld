import numpy as np


def calculate_bld_distribution(bmaxd, fmind, bmaxd_indexek, dt):
    """
    Calculates descriptive parameters of the BLD.
      Parameters:
          bmaxd: backward maximal distance
          fmind: forward mininum distance
          bmaxd_indexek: the list of the indices of the point to which the BLD is calculated
          dt: Pandas Dataframe containing the pairwise distances between the test and reference points
      Returns:
          bld_average: single value with the average of BLD
          number_of_points_BMaxD_is_bigger_than_FMinD: the number of points where BMaxD was used to get BLD
                (instead of FMinD)
          bmd_length: the number of different BMaxD values from which the maximal is chosen
  """
    number_of_points_bmaxd_is_bigger_than_fmind = \
        bmaxd_indexek[fmind - bmaxd < 0].shape[0]

    # how many different point has the same BMaxD value, corresponding to one reference point?
    bmd = []
    bmd_length = np.zeros(len(bmaxd_indexek))
    for i in range(len(bmaxd_indexek)):
        r = np.argwhere(dt.iloc[bmaxd_indexek[i]].values == bmaxd[i])
        bmd.append(r)
        bmd_length[i] = len(r)

    return number_of_points_bmaxd_is_bigger_than_fmind, bmd_length


def calculate_ldp(dt, loc, bld):
    fmins = dt.min(axis=1)
    fmins_signed = np.multiply(loc, fmins)
    diff = bld - fmins

    return fmins_signed, diff
