import glob

import SimpleITK as sitk
import cv2 as cv
from natsort import natsorted
import numpy as np

class DataLoader:
    def __init__(self, number, data_folder="data", root_folder="./"):
        self.labels_test = None
        self.labels_ref = None
        self.data_folder = data_folder
        self.root_folder = root_folder
        self.get_the_labels()

        self.c_ref = self.get_contour_from_image(file_path=self.labels_ref[number - 1])
        self.c_test = self.get_contour_from_image(file_path=self.labels_test[number - 1])

    def get_the_labels(self):
        """
        Finds the labels of the uploaded files (folder name + file name).
        Returns:
        labels_ref: the labels of the reference files
        labels_test: the labels of the test files
        """
        folder = self.root_folder + self.data_folder + "/"
        self.labels_test = natsorted(glob.glob(folder + "masks_test/*"))
        self.labels_ref = natsorted(glob.glob(folder + "masks_ref/*"))

    def get_contour_from_image(self, file_path):
        """
        Converts a nii.gz image to a list of contours.

        Parameters:
            file_path (str): the path of the selected nii.gz image
        Returns:
          dictionary:
            keys - slice number
            values - contours of the corresponding slice, each contour is one 2D numpy array with the coordinates of the contour points
        """

        im = sitk.ReadImage(file_path)
        img = sitk.GetArrayFromImage(im)

        directory = self.root_folder + self.data_folder
        # os.chdir(directory)

        # initialize dictionary
        dictionary_contours = dict()
        # get the contours
        for i in range(img.shape[0]):
            f_path = self.root_folder + self.data_folder + "/" + 'image.png'
            cv.imwrite(f_path, img[i] * 255)
            image = cv.imread(f_path)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            edged = cv.Canny(gray, 30, 200)
            contours, hierarchy = cv.findContours(
                edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

            c = []
            for contour in contours:
                c.append(contour.T.squeeze())
            dictionary_contours['slice' + str(i)] = c
        return dictionary_contours


def check_contours_on_slice(test_points, ref_points):
    if len(test_points) != len(ref_points):
        print("The number of test and reference contours are not equal. The slice should be evaluated manually.")
    else:
        print("The number of test and reference contours are equal. The automatic evaluation can be continued.")
