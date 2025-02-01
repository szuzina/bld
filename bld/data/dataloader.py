import glob
import os
from typing import Optional

import cv2 as cv
from natsort import natsorted
import SimpleITK as SITK

from bld.data import DataDownloader


class DataLoader:
    """
    Load the data of a patient in the necessary format for further analysis.

    Args:
        patient: patient number
        data_folder: will be removed
        root_folder: will be removed

    Returns:
        labels_test: the labels (paths) of all the patient to the test contours
        labels_ref: the labels (paths) of all the patient to the reference contours
        c_ref: reference contours with coordinates
        c_test: test contours with coordinates
        mask_test: test masks in np arrays
        mask_ref: reference masks in np arrays

    """
    def __init__(self, patient: int, datadownloader: DataDownloader):
        self.folder = datadownloader.root_folder + datadownloader.data_folder
        self.patient = patient

        self.labels_test: list = []
        self.labels_ref: list = []
        self.c_ref: dict = dict()
        self.c_test: dict = dict()

        self.get_the_labels()
        self.get_contours(number=patient)

        self.mask_test: dict = dict()
        self.mask_ref: dict = dict()
        self.get_masks()

    def get_contours(self, number: int):
        """
        Finds the contours from one image slice.

        Args:
            number: patient number

        Returns:
            c_ref: the reference contour(s)
            c_test: the test contour(s)
        """
        self.c_ref = self.get_contour_from_image(file_path=self.labels_ref[number - 1])
        self.c_test = self.get_contour_from_image(file_path=self.labels_test[number - 1])

    def get_the_labels(self):
        """
        Finds the labels of the uploaded files (folder name + file name).

        Returns:
        labels_ref: the labels of the reference files
        labels_test: the labels of the test files
        """
        self.labels_test = natsorted(glob.glob(os.path.join(self.folder, "masks_test", "*")))
        self.labels_ref = natsorted(glob.glob(os.path.join(self.folder, "masks_ref", "*")))

    def get_contour_from_image(self, file_path: str):
        """
        Converts a nii.gz image to a list of contours.

        Args:
            file_path: the path of the selected nii.gz image

        Returns:
          dictionary:
            keys - slice number
            values - contours of the corresponding slice, each contour is one 2D numpy array
            with the coordinates of the contour points
        """

        im = SITK.ReadImage(fileName=file_path)
        img = SITK.GetArrayFromImage(image=im)

        # initialize dictionary
        dictionary_contours = dict()
        # get the contours
        for i in range(img.shape[0]):
            f_path = os.path.join(self.folder, 'image.png')
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

    def get_masks(self):
        """
        Creates a dictionary for a patient, contains the slice masks in np array.
        """
        labels_test = natsorted(glob.glob(os.path.join(self.folder, "masks_test/*")))
        labels_ref = natsorted(glob.glob(os.path.join(self.folder, "masks_ref/*")))
        mask_t = SITK.ReadImage(fileName=labels_test[self.patient - 1])
        mask_r = SITK.ReadImage(fileName=labels_ref[self.patient - 1])
        test = SITK.GetArrayFromImage(image=mask_t)
        ref = SITK.GetArrayFromImage(image=mask_r)
        number_of_slices = min(test.shape[0], ref.shape[0])
        for i in range(number_of_slices):
            self.mask_test['slice' + str(i)] = test[i, :, :]
            self.mask_ref['slice' + str(i)] = ref[i, :, :]
