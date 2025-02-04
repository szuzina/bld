import os
from typing import Optional
import zipfile

import gdown


class DataDownloader:
    """Downloads all the necessary data from the cloud, including masks and manual scores.

    Args:
        ref_url: the link of the directory containing the reference masks.
        test_url: the link of the directory containing the test masks.
        csv_data_id: the link of the zip file containing the manual score csv files
        data_folder: the folder which will be used for data storage
        root_folder: the root folder

    Returns:
        download the files to the specified directory
    """

    def __init__(self, ref_url: str, test_url: str, csv_data_id: str, data_folder: Optional[str] = "data",
                 root_folder: Optional[str] = "./"):
        self.root_folder = root_folder
        self.data_folder = data_folder
        self.ref_url = ref_url
        self.test_url = test_url
        self.csv_data_id = csv_data_id

        self.download_files()
        self.download_csv_dir()

    def download_files(self):
        """
        Download all the currently available reference and test segmentations.
        Create a folder for the reference files (masks_ref) and test files (masks_test) in the data folder.
        """
        if not os.path.isdir(os.path.join(self.root_folder, self.data_folder)):
            os.makedirs(self.data_folder, exist_ok=True)
            
            gdown.download(
                url=self.ref_url,
                output=os.path.join(self.root_folder, self.data_folder, "masks_ref.zip"),
                quiet=False
            )
            gdown.download(
                url=self.test_url,
                output=os.path.join(self.root_folder, self.data_folder, "masks_test.zip"),
                quiet=False
            )
        
            with zipfile.ZipFile(
                    file=os.path.join(self.root_folder, self.data_folder, "masks_ref.zip"),
                    mode='r') as zip_ref:
                zip_ref.extractall(
                    path=os.path.join(self.root_folder, self.data_folder, "masks_ref"))
            with (zipfile.ZipFile(
                    file=os.path.join(self.root_folder, self.data_folder, "masks_test.zip"),
                    mode='r') as zip_test):
                zip_test.extractall(
                    path=os.path.join(self.root_folder, self.data_folder, "masks_test"))
            
            return 0

    def download_csv_dir(self):
        """
        Download the CSV directory containing the manual scores of the patients.

        One csv file corresponds to one patient.
        The CSV file contain the slice index in the first column, the manual score in the second column.
        """

        drive_url = 'https://drive.google.com/uc?export=download&id='
        csv_directory_url = drive_url + self.csv_data_id

        # download the csv directory
        gdown.download(url=csv_directory_url, output=os.path.join(self.root_folder, 'data/csv_zip'), quiet=False)
        with zipfile.ZipFile(file=os.path.join(self.root_folder, 'data/csv_zip'), mode='r') as zip_ref:
            zip_ref.extractall(path=os.path.join(self.root_folder, 'data/csv_dir'))
        return 0
