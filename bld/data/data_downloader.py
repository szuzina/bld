import os
import zipfile

import gdown


class DataDownloader:
    def __init__(self, ref_url, test_url, csv_data_id, data_folder="data",
                 root_folder="./"):
        self.root_folder = root_folder
        self.data_folder = data_folder
        self.ref_url = ref_url
        self.test_url = test_url
        self.csv_data_id = csv_data_id

        self.download_files()
        self.download_csv_dir()

    def download_files(self):
        """
        Upload all the currently available reference and test segmentations.
        Create a folder for the reference files (masks_ref) and test files (masks_test) in the data folder.
        """
        if not os.path.isdir(os.path.join(self.root_folder, self.data_folder)):
            os.makedirs(self.data_folder, exist_ok=True)
            
            gdown.download(self.ref_url, output=os.path.join(self.root_folder, self.data_folder, "/masks_ref.zip"), quiet=False)
            gdown.download(self.test_url, output=os.path.join(self.root_folder, self.data_folder, "/masks_test.zip"), quiet=False)
        
            with zipfile.ZipFile(os.path.join(self.root_folder, self.data_folder, "/masks_ref.zip"), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.root_folder, self.data_folder, "/masks_ref"))
            with zipfile.ZipFile(os.path.join(self.root_folder, self.data_folder, "/masks_test.zip"), 'r') as zip_test:
                zip_test.extractall(os.path.join(self.root_folder, self.data_folder, "/masks_test"))
            
            return 0

    def download_csv_dir(self):

        drive_url = 'https://drive.google.com/uc?export=download&id='
        csv_directory_url = os.path.join(drive_url, self.csv_data_id)

        # download the csv directory
        gdown.download(csv_directory_url, output=os.path.join(self.root_folder, '/data/csv_zip'), quiet=False)
        with zipfile.ZipFile(os.path.join(self.root_folder, '/data/csv_zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(self.root_folder, '/data/csv_dir'))
        return 0
