import glob
import os
import zipfile

import gdown
import pandas as pd


class CSVDataLoader:

    def __init__(self, p_number, idx, csv_data_id, root_folder='/content'):

        self.csv_data_id = csv_data_id
        self.root_folder = root_folder
        
        self.data = self.upload_csv_dir()

        self.patient_data = self.find_patient_data()
        self.filtered_scores = self.find_filtered_scores(filtered_rows=idx)


    def upload_csv_dir(self):
        drive_url = 'https://drive.google.com/uc?export=download&id='
        csv_directory_url = drive_url + self.csv_data_id
        # download the csv directory
        gdown.download(csv_directory_url, output=self.root_folder+'/bld/data/csv_zip', quiet=False)
        with zipfile.ZipFile(self.root_folder + '/bld/data/csv_zip', 'r') as zip_ref:
            zip_ref.extractall(self.root_folder + '/bld/data/csv_dir')
        return 0

    def find_patient_data(self):

        csv_directory = self.root_folder + '/bld/data/csv_dir'
        patient_path = csv_directory + f'/patient{self.p_number}.csv'
        df = pd.read_csv(patient_path, header=None, sep=';')

        return df

    def find_filtered_scores(self, filtered_rows):
        filtered_scores = self.patient_data.loc[self.patient_data.iloc[:, 0].isin(filtered_rows)].iloc[:, 1].tolist()
        return filtered_scores
