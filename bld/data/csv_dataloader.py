import os

import pandas as pd

from bld.data.data_downloader import DataDownloader


class CSVDataLoader:
    """
    Load the csv files containing the manual scores in the necessary format for futher analysis.

    Args:
        p_number: patient number
        idx: the indices for which MSI was calculated
        datadownloader: data downloader object

    Returns:
        patient_data: all the manual scores corresponding to the selected patient
        filtered_scores: the slice scores which are needed for correlation analysis

    """

    def __init__(self, p_number: int, idx: list,
                 datadownloader: DataDownloader):

        self.folder = os.path.join(datadownloader.root_folder,
                                   datadownloader.data_folder)
        self.p_number = p_number

        self.patient_data = self.find_patient_data()
        self.filtered_scores = self.find_filtered_scores(filtered_rows=idx)

    def find_patient_data(self):
        """
        Load the csv data for the selected patient.
        """
        csv_directory = os.path.join(self.folder, 'csv_dir')
        patient_path = os.path.join(csv_directory, f'patient{self.p_number}.csv')
        df = pd.read_csv(filepath_or_buffer=patient_path, header=None, sep=';')

        return df

    def find_filtered_scores(self, filtered_rows: list):
        """
        Filter the manual scores to have just the slices for which MSI was calculated.

        Args:
            filtered_rows: the row indices for which MSI was calculated

        Returns:
            filtered_scores: the scores for the filtered rows
        """
        filtered_scores = self.patient_data.loc[self.patient_data.iloc[:, 0].isin(filtered_rows)].iloc[:, 1].tolist()

        return filtered_scores
