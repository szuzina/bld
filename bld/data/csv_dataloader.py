import os
import glob

import pandas as pd
from natsort import natsorted
from typing import Optional

from bld.data.data_downloader import DataDownloader


class CSVDataLoader:
    """
    Load the csv files containing the manual scores in the necessary format for further analysis.

    Args:
        p_number: patient number
        idx: the indices for which MSI was calculated
        datadownloader: data downloader object
        aggregation: the slice aggregation method

    Returns:
        patient_data: all the manual scores corresponding to the selected patient
        filtered_scores: the slice scores which are needed for correlation analysis

    """

    def __init__(self, p_number: int, idx: list,
                 datadownloader: DataDownloader, aggregation: Optional[int] = 1):

        self.folder = os.path.join(datadownloader.root_folder,
                                   datadownloader.data_folder)
        self.p_number = p_number
        self.aggregation = aggregation  # 1: median, 2: min, 3: max

        self.patient_data = self.find_patient_data()
        self.filtered_scores = self.find_filtered_scores(filtered_rows=idx)

    def find_patient_data(self):
        """
        Load the csv data for the selected patient.
        """
        csv_directory = os.path.join(self.folder, 'csv_dir')

        labels_ref = natsorted(glob.glob(os.path.join(self.folder, "masks_ref", "*")))
        n = str(labels_ref[self.p_number-1][-10:-7])

        patient_path = os.path.join(csv_directory, f'p{n}.csv')
        df = pd.read_csv(filepath_or_buffer=patient_path, header=None, sep=';')

        return df

    def find_filtered_scores(self, filtered_rows: list):
        """
        Filter the manual scores to have just the slices which we want to include in the correlation analysis.

        Args:
            filtered_rows: the row indices for which the manual scores are needed
            (without zeros case: where MSI was calculated)
            (with zeros case: all slices)

        Returns:
            filtered_scores: the scores for the filtered rows
        """

        filtered_scores = self.patient_data.loc[
                              self.patient_data.iloc[:, 0].isin(filtered_rows)].iloc[:, self.aggregation].tolist()

        return filtered_scores
