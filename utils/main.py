import nnunetv2
from utils import DataDownloaderUNet
from utils import DataLoaderUNet

import torch
import os

train_img_url = 'https://drive.google.com/uc?export=download&id=' + '1xpppHMhZCDBytnt-eYLtbPA26xAgrlUO'
test_img_url = 'https://drive.google.com/uc?export=download&id=' + '17nR0E6WTDpnyiV055xJwHEbzDPUFqimA'
ddl = DataDownloaderUNet(train_url=train_img_url, test_url=test_img_url, root_folder='./')

dl = DataLoaderUNet(task_name='Dataset000_prostate', datadownloader=ddl)

#print(torch.cuda.is_available())

# nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

# nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD
# caution: default 1000 epoch per fold

# nnUNetv2_predict -d DATASET_ID -i INPUT_FOLDER -o OUTPUT_FOLDER -c CONFIGURATION -f 0 1 2 3 4-
