import glob
import os
import zipfile

import gdown
import pandas as pd


class CSVDataLoader:

    def __init__(self, p_number, idx, csv_data_id, root_folder='/content'):

        self.csv_data_id = csv_data_id
        self.data = self.create_json()
        self.patient_data = self.data[self.data['source_index'] == p_number]
        self.filtered_scores = self.find_filtered_scores(filtered_rows=idx)
        self.root_folder = root_folder

    def create_json(self):
        drive_url = 'https://drive.google.com/uc?export=download&id='
        csv_directory_url = drive_url + self.csv_data_id
        # download the csv directory
        gdown.download(csv_directory_url, output=self.root_folder+'/bld/data/csv_zip', quiet=False)
        with zipfile.ZipFile(self.root_folder + '/bld/data/csv_zip', 'r') as zip_ref:
            zip_ref.extractall(self.root_folder + '/bld/data/csv_dir')

        # create json file
        csv_directory = self.root_folder + '/bld/data/csv_dir'
        json_file_path = self.root_folder + '/bld/data/combined_data.json'

        dataframes = []

        for idx, csv_file in enumerate(glob.glob(os.path.join(csv_directory, '*.csv')), start=1):
            df = pd.read_csv(csv_file, header=None, sep=';')
            df['source_index'] = idx  # Add source index column
            dataframes.append(df)

        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.to_json(json_file_path, orient='records', lines=True)

        print(f"Combined JSON file saved as {json_file_path}")

        return combined_df

    def find_filtered_scores(self, filtered_rows):
        filtered_scores = self.patient_data.loc[self.patient_data.iloc[:, 0].isin(filtered_rows)].iloc[:, 2].tolist()
        return filtered_scores
