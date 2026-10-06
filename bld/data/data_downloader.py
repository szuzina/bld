import os
import zipfile
from typing import Optional

import gdown


class DataDownloader:
    """Download and extract the data required for segmentation evaluation.

    The downloader independently checks whether reference masks, test masks and manual-score CSV files
    are already available. Only missing data are downloaded.

    Args:
        ref_url: URL of the ZIP archive containing the reference masks.
        test_url: URL of the ZIP archive containing the test masks.
        csv_data_id: Google Drive ID of the ZIP archive containing the CSV files.
        data_folder: Folder used for data storage.
        root_folder: Root folder of the project.
    """

    def __init__(self, ref_url: str, test_url: str, csv_data_id: str,
                 data_folder: Optional[str] = "data",
                 root_folder: Optional[str] = "./",
    ):
        self.root_folder = root_folder
        self.data_folder = data_folder
        self.ref_url = ref_url
        self.test_url = test_url
        self.csv_data_id = csv_data_id

        self.data_path = os.path.join(self.root_folder, self.data_folder)

        self.download_files()
        self.download_csv_dir()

    def download_files(self):
        """Download and extract reference and test segmentation masks.
        Each dataset is checked independently. Already available datasets are not downloaded again.
        """
        self._download_and_extract(url=self.ref_url, zip_name="masks_ref.zip", extract_dir="masks_ref")
        self._download_and_extract(url=self.test_url, zip_name="masks_test.zip", extract_dir="masks_test")

    def download_csv_dir(self):
        """Download and extract the CSV files containing manual scores.

        Each CSV file corresponds to one patient. The first column contains the slice index and the second column
        contains the manual score.
        The CSV archive is downloaded only if the extracted CSV directory does not already exist.
        """
        csv_dir = os.path.join(self.data_path, "csv_dir")
        csv_zip = os.path.join(self.data_path, "csv_zip")

        if os.path.isdir(csv_dir):
            print(f"CSV data already exists: {csv_dir}")
            return

        os.makedirs(self.data_path, exist_ok=True)

        drive_url = "https://drive.google.com/uc?export=download&id="
        csv_directory_url = drive_url + self.csv_data_id
        gdown.download(url=csv_directory_url, output=csv_zip, quiet=False)
        self._extract_zip(zip_path=csv_zip, extract_dir=csv_dir)

    def _download_and_extract(self, url: str, zip_name: str, extract_dir: str):
        """Download and extract a ZIP archive if its data are missing."""
        extract_path = os.path.join(self.data_path, extract_dir)
        zip_path = os.path.join(self.data_path, zip_name)

        if os.path.isdir(extract_path):
            print(f"Data already exists: {extract_path}")
            return

        os.makedirs(self.data_path, exist_ok=True)
        gdown.download(url=url, output=zip_path, quiet=False,)
        self._extract_zip(zip_path=zip_path, extract_dir=extract_path)

    @staticmethod
    def _extract_zip(zip_path: str, extract_dir: str):
        """Extract a ZIP archive to the specified directory."""
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, mode="r") as zip_ref:
            zip_ref.extractall(path=extract_dir)

