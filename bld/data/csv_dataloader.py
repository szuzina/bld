import os

import pandas as pd


class CSVDataLoader:

    def __init__(self, p_number, idx, root_folder='./'):

        self.root_folder = root_folder
        self.p_number = p_number

        self.patient_data = self.find_patient_data()
        self.filtered_scores = self.find_filtered_scores(filtered_rows=idx)

    def find_patient_data(self):
        csv_directory = 'data/csv_dir'
        patient_path = os.path.join(csv_directory, f'patient{self.p_number}.csv')
        df = pd.read_csv(patient_path, header=None, sep=';')

        return df

    def find_filtered_scores(self, filtered_rows):
        filtered_scores = self.patient_data.loc[self.patient_data.iloc[:, 0].isin(filtered_rows)].iloc[:, 1].tolist()

        return filtered_scores
