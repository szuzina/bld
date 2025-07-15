import os
import shutil
import zipfile

import json
from collections import OrderedDict

from bld.utils import DataDownloaderUNet

class DataLoaderUNet:
    def __init__(self, task_name: str, datadownloader: DataDownloaderUNet):
        self.root_folder = datadownloader.root_folder
        self.task_name = task_name
        self.ddl = datadownloader

        self.nnunet_dir = os.path.join(
            self.root_folder,
            'nnUNet/nnunet/nnUNet_raw')

        self.task_folder_name = os.path.join(self.nnunet_dir, task_name)
        self.train_image_dir = os.path.join(self.task_folder_name, 'imagesTr')
        self.train_label_dir = os.path.join(self.task_folder_name, 'labelsTr')
        self.test_dir = os.path.join(self.task_folder_name, 'imagesTs')

        self.train_zip_filename = 'train_prostate_images.zip'

        self.main_dir = os.path.join(self.root_folder, 'nnUNet/nnunet')

        self.create_folders()
        self.move_images_to_folder()
        self.verify_dataset()
        self.rename_for_single_modality(self.train_image_dir, modality_code='0000')
        self.rename_for_single_modality(self.test_dir, modality_code='0000')
        self.create_json()

    @staticmethod
    def make_if_dont_exist(folder_path, overwrite=False):
        if os.path.exists(folder_path):
            if not overwrite:
                print(f"{folder_path} exists.")
            else:
                print(f"{folder_path} overwritten")
                shutil.rmtree(folder_path)
                os.makedirs(folder_path)
        else:
            os.makedirs(folder_path)
            print(f"{folder_path} created!")

    @staticmethod
    def copy_and_rename(old_location, old_file_name, new_location,
                        new_filename, delete_original=False):
        shutil.copy(os.path.join(old_location, old_file_name), new_location)
        os.rename(os.path.join(new_location, old_file_name),
                  os.path.join(new_location, new_filename))
        if delete_original:
            os.remove(os.path.join(old_location, old_file_name))

    def create_folders(self):
        self.make_if_dont_exist(folder_path=self.task_folder_name)
        self.make_if_dont_exist(folder_path=self.train_image_dir)
        self.make_if_dont_exist(folder_path=self.train_label_dir)
        self.make_if_dont_exist(folder_path=self.test_dir)
        self.make_if_dont_exist(folder_path=os.path.join(self.main_dir, 'nnUNet_results'))
        self.make_if_dont_exist(folder_path=os.path.join(self.main_dir, 'nnUNet_preprocessed'))

        os.environ['nnUNet_raw'] = os.path.join(
                                        self.main_dir, 'nnUNet_raw')
        os.environ['nnUNet_preprocessed'] = os.path.join(
                                        self.main_dir, 'nnUNet_preprocessed')
        os.environ['nnUNet_results'] = os.path.join(
                                        self.main_dir, 'nnUNet_results')

    def move_images_to_folder(self):
        self.copy_and_rename(old_location=self.ddl.zip_path,
                             new_location=self.task_folder_name,
                             old_file_name=self.ddl.train_zip_name,
                             new_filename=self.train_zip_filename,
                             delete_original=False)
        # majd a végén kitörölhetjük, de amíg dolgozom, jobb ha nem

        # unzip train files
        with zipfile.ZipFile(os.path.join(self.ddl.zip_path, self.ddl.train_zip_name), 'r') as zip_ref:
            zip_ref.extractall(self.task_folder_name)

        # move train images to imagesTr, masks to labelsTr
        for file in os.listdir(self.task_folder_name):
            if file.endswith('.nii.gz'):
                if 'segmentation' in file:
                    shutil.move(os.path.join(self.task_folder_name, file),
                                self.train_label_dir)
                    for file in os.listdir(self.train_label_dir):
                        new_name = file.replace("_segmentation", "")
                        os.rename(os.path.join(self.train_label_dir, file), os.path.join(self.train_label_dir, new_name))
                else:
                    shutil.move(os.path.join(self.task_folder_name, file),
                                self.train_image_dir)
        # unzip test files
        with zipfile.ZipFile(os.path.join(self.ddl.zip_path, self.ddl.test_zip_name), 'r') as zip_ref:
            zip_ref.extractall(self.test_dir)

    def verify_dataset(self):
        train_files = os.listdir(self.train_image_dir)
        label_files = os.listdir(self.train_label_dir)
        test_files = os.listdir(self.test_dir)
        print("train image files:", len(train_files))
        print("train label files:", len(label_files))
        print("test image files:", len(test_files))

    @staticmethod
    def check_modality(filename):
        end = filename.find('.nii.gz')
        modality = filename[end-4:end]
        for mod in modality:
            if not mod.isdigit():
                return False
        return True

    def rename_for_single_modality(self, directory, modality_code):
        for file in os.listdir(directory):
            if not self.check_modality(file):
                new_name = file[:file.find('.nii.gz')] + '_' + modality_code + '.nii.gz'
                os.rename(os.path.join(directory, file), os.path.join(directory, new_name))

    def create_json(self):
        overwrite_json_file = True
        json_file_exist = False

        if os.path.exists(os.path.join(self.task_folder_name, 'dataset.json')):
            print('dataset.json already exists')
            json_file_exist = True

        if not json_file_exist or overwrite_json_file:
            json_dict = OrderedDict()
            json_dict['name'] = self.task_name

            json_dict['channel_names'] = {
                "0": "T2"
            }

            json_dict['labels'] = {
                "background": "0",
                "prostate": "1",
            }

            train_ids = os.listdir(self.train_label_dir)
            test_ids = os.listdir(self.test_dir)
            json_dict['numTraining'] = len(train_ids)
            json_dict['numTest'] = len(test_ids)
            json_dict['file_ending'] = ".nii.gz"

            # no modality in train image and labels in dataset.json
            json_dict['training'] = [{'image': "./imagesTr/%s" % i, "label": "./labelsTr/%s" % i} for i in train_ids]

            # removing the modality from test image name to be saved in dataset.json
            json_dict['test'] = ["./imagesTs/%s" % (i[:i.find("_0000")] + '.nii.gz') for i in test_ids]

            with open(os.path.join(self.task_folder_name, "dataset.json"), 'w') as f:
                json.dump(json_dict, f, indent=4, sort_keys=True)

            if os.path.exists(os.path.join(self.task_folder_name, 'dataset.json')):
                if not json_file_exist:
                    print('dataset.json created!')
                else:
                    print('dataset.json overwritten!')
