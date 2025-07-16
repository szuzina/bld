from utils import DataDownloaderUNet
from utils import DataLoaderUNet

train_img_url = 'https://drive.google.com/uc?export=download&id=' + '1xpppHMhZCDBytnt-eYLtbPA26xAgrlUO'
test_img_url = 'https://drive.google.com/uc?export=download&id=' + '17nR0E6WTDpnyiV055xJwHEbzDPUFqimA'
ddl = DataDownloaderUNet(train_url=train_img_url, test_url=test_img_url, root_folder='./')

dl = DataLoaderUNet(task_name='Dataset000_prostate', datadownloader=ddl)

