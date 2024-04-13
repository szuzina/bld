class DataDownloaderColab:
    def __init__(self, ref_url, test_url, data_folder="data"):
        self.data_folder = data_folder
        self.ref_url = ref_url
        self.test_url = test_url

        self.upload_files()

    def upload_files(self):
        """
        Upload all the currently available reference and test segmentations.
        Create a folder for the reference files (masks_ref) and test files (masks_test) in the data folder.
        """
        if not os.path.isdir("/content/" + self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)
            r = requests.get(self.ref_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall("/content/" + self.data_folder + "/masks_ref")
            r = requests.get(self.test_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall("/content/" + self.data_folder + "/masks_test")
            return 0
