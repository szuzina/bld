import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


def color_red_font_minimum_in_a_column(column):
    highlight = 'color: red;'
    default = ''
    minimum_in_column = column.min()
    return [highlight if e == minimum_in_column else default for e in column]

def color_green_minimum_value_in_row(row):
    highlight = 'background-color: green;'
    default = ''
    minimum_in_row = row.min()
    # must return one string per cell in this column
    return [highlight if v == minimum_in_row else default for v in row]


def rearrange_table(df):
  # the closest point to Pref0 will be in the first column
  # the closest point to Pref0 will be in the first column
  # based on FminD
  
  sor_min_indexek = np.argmin(df.values, axis=1)
  oszlop_min_indexek = np.argmin(df.values, axis=0)

  df_atrendezett = df[[df.columns[i] for i in sor_min_indexek]]

  diff_sor = np.diff(sor_min_indexek)
  diff_oszlop = np.diff(oszlop_min_indexek)

  return df_atrendezett, diff_sor, diff_oszlop


def calculate_bld_distribution(bmaxd, fmind, bmaxd_indexek, dt):
  """
  Calculates descriptive parameters of the BLD.
    Parameters:
      bld: list of BLD corresponding to the reference points
      bmaxd_indexek: the list of the indices of the point to which the BLD is calculated
      dt: Pandas Dataframe containing the pairwise distances between the test and reference points
    Returns:
      bld_average: single value with the average of BLD
      number_of_points_BMaxD_is_bigger_than_FMinD: the number of points where BMaxD was used to get BLD
            (instead of FMinD)
      bmd_length: the number of different BMaxD values from which the maximal is chosen
  """
  number_of_points_BMaxD_is_bigger_than_FMinD = \
    bmaxd_indexek[fmind-bmaxd < 0].shape[0]

  # how many different point has the same BMaxD value, corresponding to one reference point?
  bmd = []
  bmd_length = np.zeros(len(bmaxd_indexek))
  for i in range(len(bmaxd_indexek)):
    r = np.argwhere(dt.iloc[bmaxd_indexek[i]].values == bmaxd[i])
    bmd.append(r)
    bmd_length[i] = len(r)

  return number_of_points_BMaxD_is_bigger_than_FMinD, bmd_length


