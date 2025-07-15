import nnunetv2
from DataDownloader import DataDownloader
from DataLoader import DataLoader

import torch

train_img_url = 'https://drive.google.com/uc?export=download&id=' + '1xpppHMhZCDBytnt-eYLtbPA26xAgrlUO'
test_img_url = 'https://drive.google.com/uc?export=download&id=' + '17nR0E6WTDpnyiV055xJwHEbzDPUFqimA'
ddl = DataDownloader(train_url=train_img_url, test_url=test_img_url, root_folder='./')

dl = DataLoader(task_name='Dataset000_prostate', datadownloader=ddl)

#print(torch.cuda.is_available())

