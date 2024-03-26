import glob
import io
import os
import zipfile

import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from natsort import natsorted
import numpy as np
import pandas as pd
import requests
import SimpleITK as sitk


def upload_files(ref_url, test_url):
    """
    Upload all the currently available reference and test segmentations.
    Create a folder for the reference files (masks_ref) and test files (masks_test) in the data folder.
    """

    os.makedirs("data", exist_ok=True)
    r = requests.get(ref_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("/content/data/masks_ref")
    r = requests.get(test_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("/content/data/masks_test")
    return 0


def get_the_labels():
    """
    Finds the labels of the uploaded files (folder name + file name).
    Returns:
    labels_ref: the labels of the reference files
    labels_test: the labels of the test files
    """
    folder = '/content/data/'
    labels_test = natsorted(glob.glob(folder + "masks_test/*"))
    labels_ref = natsorted(glob.glob(folder + "masks_ref/*"))
    return labels_ref, labels_test


def get_contour_from_image(file_path):
    """
    Converts a nii.gz image to a list of contours.

    Parameters:
        file_path (str): the path of the selected nii.gz image
    Returns:
        list: the list of the contours, the length of the list is the number of slices in the original image
              each contour is one 2D numpy array with the coordinates of the contour points
    """

    im = sitk.ReadImage(file_path)
    img = sitk.GetArrayFromImage(im)

    directory = r'/content/data'
    os.chdir(directory)

    cont = []

    for i in range(img.shape[0]):
        cv.imwrite('image.png', img[i] * 255)
        image = cv.imread('/content/data/image.png')
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        edged = cv.Canny(gray, 30, 200)
        contours, hierarchy = cv.findContours(
            edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        c = np.array(contours).squeeze().T
        cont.append(c)

    return cont


def move_coms(c_ref, c_test):
    """
    Moves the test contour to align the center of mass with the COM of the reference contour.
    Returns: the moved test contour.
    """
    com_ref = c_ref.mean(axis=0)
    com_test = c_test.mean(axis=0)
    move_vector = com_ref - com_test
    c_test_corrected = c_test + move_vector
    return c_test_corrected


def find_pairwise_dist(c_ref, c_test):
    """
    Finds the pairwise distances between two contours (i.e. a reference and a test contour).
    """
    n_1 = c_ref.shape[0]
    n_2 = c_test.shape[0]
    v1v2 = c_ref @ c_test.T
    v12 = np.sum(c_ref ** 2, axis=1).reshape((n_1, 1))
    v22 = np.sum(c_test ** 2, axis=1)
    pairwise_dist = np.sqrt(v12 - 2 * v1v2 + v22)
    return pairwise_dist


def create_table(pairwise_dist):
    """
    Creates the table with the distances. The rows correspond to the reference points,
    the columns correspond to the test points.
    Each cell contains the Euclidean distance between the two points of the row index and column index.
    """
    cols = []
    for i in range(len(pairwise_dist[0, :])):
        cols.append('Ptest' + str(i))

    ind = []
    for i in range(len(pairwise_dist[:, 0])):
        ind.append('Pref' + str(i))

    df = pd.DataFrame(pairwise_dist, columns=cols, index=ind)

    return df


def calculate_bld(df):
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
    oszlop_min_indexek = np.argmin(df.values, axis=0)

    tabla = oszlop_min_indexek.reshape((-1, 1)) == \
        np.arange(0, df.shape[0], 1).reshape((1, -1))
    # filtering the reference points (rows) to which there exist column minimum
    szures_sor_index_van_oszlop_minimum = tabla.sum(axis=0) > 0
    # the BMaxD is the maximum of the column minimums
    # we find the indices of BMaxD distances
    bmaxd_indexek = np.arange(
      0, df.shape[0], 1
    )[szures_sor_index_van_oszlop_minimum]
    # if there is no BMaxD, then FMinD is used
    # we find the indices of FMinD distances
    fmind_indexek = np.arange(
      0, df.shape[0], 1
    )[~szures_sor_index_van_oszlop_minimum]

    bld = np.zeros((df.shape[0], ))

    # we find the values of FMinD distances
    bld[fmind_indexek] = df.min(axis=1).iloc[fmind_indexek]

    tabla_2 = (oszlop_min_indexek.reshape((1, -1)) ==
               np.arange(0, df.shape[0], 1).reshape((-1, 1)))
    
    # we find the values of BMaxD distances
    bmaxd = np.max(tabla_2 * df.min(axis=0).values.reshape((-1, )),
                   axis=1
                   )[bmaxd_indexek]

    fmind = df.min(axis=1).iloc[bmaxd_indexek].values

    # BLD is the maximum of BMaxD and FMinD
    bld_bmaxd_letezik = np.maximum(bmaxd, fmind)
    # if BMaxD does not exist, then BLD = FMinD
    bld[bmaxd_indexek] = bld_bmaxd_letezik

    return bmaxd_indexek, fmind, bmaxd, bld


def calculate_signed_distances(test_corrected, ref, bld):
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
    test_cor = test_corrected.astype(int)
    location = []
    # 1: inside, 0: on the contour, -1: outside
    for i in range(len(ref)):
        location.append(
            cv.pointPolygonTest(contour=test_cor,
                                pt=ref[i].astype(float),
                                measureDist=False))
    loc = np.array(location)
    bld_signed = np.multiply(loc, bld)
    return loc, bld_signed


def calculate_corrected_bld(df, bld, loc,
                            ref, test,
                            test_corrected):
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
    sor_bld_indexek = np.zeros(len(df))
    
    for i in range(len(df)):
        # we assign the pairs to the reference contour points
        sor_bld_indexek[i] = np.argwhere(df.iloc[i].values == bld[i])

    test_df = pd.DataFrame(test_corrected, columns=['x', 'y'])
    list_sor_bld_indexek = sor_bld_indexek.tolist()
    test_paired = test_df.iloc[list_sor_bld_indexek]

    # we move back the test points to the original location
    # we only use the test points, which are paired with reference points
    # these test points are in the same order as the reference points
    # (the pair of the ith reference point is the ith test point

    test_points_paired = test_paired.to_numpy()

    com_ref = ref.mean(axis=0)
    com_test = test.mean(axis=0)
    move_vector = com_ref - com_test

    paired_test_points_moved_back = test_points_paired - move_vector

    final_bld = np.multiply(
      loc,
      np.sqrt(((ref - paired_test_points_moved_back) ** 2).sum(axis=1))
    )  # calculate the distance of the points in each pair

    return final_bld, paired_test_points_moved_back


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


def calculate_msi(final_bld, il, ol):
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
    mcf_inside['WF value'] = weight_function(
      d=mcf_inside.loc[mcf_inside['corr.BLD'] < 0],
      l=il)
    mcf_outside['WF value'] = weight_function(
      d=mcf_outside.loc[mcf_outside['corr.BLD'] > 0],
      l=ol)
    msi = 1 / len(final_bld) * (
      mcf_inside['WF value'].sum() + mcf_outside['WF value'].sum()
    )
    return msi



# functions after the "main" part

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


