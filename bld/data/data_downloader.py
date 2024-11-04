import io
import os
import zipfile

import requests
import gdown


class DataDownloader:
    def __init__(self, ref_url, test_url, data_folder="data",
                 root_folder="./"):
        self.root_folder = root_folder
        self.data_folder = data_folder
        self.ref_url = ref_url
        self.test_url = test_url

        self.upload_files()

    def upload_files(self):
        """
        Upload all the currently available reference and test segmentations.
        Create a folder for the reference files (masks_ref) and test files (masks_test) in the data folder.
        """
        if not os.path.isdir(self.root_folder + self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)
            
            gdown.download(self.ref_url, output=self.root_folder + self.data_folder + "/masks_ref.zip", quiet=False)
            gdown.download(self.test_url, output=self.root_folder + self.data_folder + "/masks_test.zip", quiet=False)
        
            with zipfile.ZipFile(self.root_folder + self.data_folder + "/masks_ref.zip", 'r') as zip_ref:
                zip_ref.extractall(self.root_folder + self.data_folder + "/masks_ref")
            with zipfile.ZipFile(self.root_folder + self.data_folder + "/masks_test.zip", 'r') as zip_test:
                zip_test.extractall(self.root_folder + self.data_folder + "/masks_test")
            
            return 0
