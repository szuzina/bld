import glob
import os
import re

import cv2 as cv
from natsort import natsorted
import SimpleITK as SITK
import nibabel as nib

from bld.data import DataDownloader


class DataLoaderErrors:
    """
    Load the data of a patient in the necessary format for further analysis.
    Needs the test file path as input, finds the corresponding reference segmentation and loads both.

    Args:
        data_downloader: data downloader object
        test_file_path: path to the test file

    Returns:
        labels_ref: the labels (paths) of all the patient to the reference contours
        c_ref: reference contours with coordinates
        c_test: test contours with coordinates
        mask_test: test masks in np arrays
        mask_ref: reference masks in np arrays

    """
    def __init__(self, data_downloader: DataDownloader, test_file_path: str):
        self.folder = os.path.join(data_downloader.root_folder, data_downloader.data_folder)
        self.test_file_path = test_file_path
        self.number = self.find_patient_number()

        self.labels_ref: list = []
        self.get_the_labels()
        self.ref_patient = self.labels_ref[self.number - 1] # store the reference patient label for later sanity check

        self.c_ref: dict = dict()
        self.c_test: dict = dict()
        self.get_contours(number=self.number)

        self.mask_test: dict = dict()
        self.mask_ref: dict = dict()
        self.get_masks()

        self.spacing = ()
        self.get_spacing()

    def find_patient_number(self):
        match = re.search(r"patient(\d+)", str(self.test_file_path))
        if not match:
            raise ValueError(f"Patient number not found in: {self.test_file_path}")
        return int(match.group(1))

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
        self.c_test = self.get_contour_from_image(file_path=self.test_file_path)

    def get_the_labels(self):
        """
        Finds the labels of the uploaded files (folder name + file name).

        Returns:
        labels_ref: the labels of the reference files
        """

        self.labels_ref = natsorted(glob.glob(os.path.join(self.folder, "masks_ref", "*")))

    def get_contour_from_image(self, file_path: str) -> dict:
        """
        Converts a nii.gz image to a list of contours.

        Args:
            file_path: the path of the selected nii.gz image

        Returns:
          dictionary:
            keys - slice number (starting from 0)
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

        mask_t = SITK.ReadImage(fileName=self.test_file_path)
        mask_r = SITK.ReadImage(fileName=self.labels_ref[self.number - 1])
        test = SITK.GetArrayFromImage(image=mask_t)
        ref = SITK.GetArrayFromImage(image=mask_r)
        number_of_slices = min(test.shape[0], ref.shape[0])
        for i in range(number_of_slices):
            self.mask_test['slice' + str(i)] = test[i, :, :]
            self.mask_ref['slice' + str(i)] = ref[i, :, :]

    def get_spacing(self):
        img = nib.load(self.labels_ref[self.number - 1])
        self.spacing = img.header.get_zooms()[:3] # (x,y,z)

