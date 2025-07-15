import os
from typing import Optional

import gdown


class DataDownloaderUNet:
    """Downloads the image data from the cloud.

    Args:
        url: the link of the directory containing the images.
        data_folder: the folder which will be used for data storage
        root_folder: the root folder

    Returns:
        download the files to the specified directory
    """

    def __init__(self, train_url: str, test_url: str, data_folder: Optional[str] = "nnUNet",
                 root_folder: Optional[str] = "./"):
        self.root_folder = root_folder
        self.data_folder = data_folder
        self.train_url = train_url
        self.test_url = test_url

        self.train_zip_name = 'train_images.zip'
        self.test_zip_name = 'test_images.zip'
        self.zip_path = os.path.join(self.root_folder, self.data_folder)

        self.download_files()

    def download_files(self):
        """
        Download all the currently available images.
        Create a folder for the images in the data folder.
        """
        if not os.path.isdir(os.path.join(self.root_folder, self.data_folder)):
            os.makedirs(self.data_folder, exist_ok=True)

            gdown.download(
                url=self.train_url,
                output=os.path.join(self.zip_path, self.train_zip_name),
                quiet=False
            )

            gdown.download(
                url=self.test_url,
                output=os.path.join(self.zip_path, self.test_zip_name),
                quiet=False
            )

            return 0
