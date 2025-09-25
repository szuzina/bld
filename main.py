from bld.data import DataDownloader, DataLoader
from bld.metrics import MSICalculator
from bld.evaluation import MetricsEvaluator

import pandas as pd
import statistics


def main():
    folder_url_ref = 'https://drive.google.com/uc?export=download&id=1u2CMExEtQSi1iMclEdlr84YkgY-fd2C-'
    folder_url_test = 'https://drive.google.com/uc?export=download&id=1U4o0AhgpF9RsS6nlGeJk8kvz2nDnVwmt'

    ddl = DataDownloader(ref_url=folder_url_ref, test_url=folder_url_test,
                         data_folder="data", root_folder='./')

    # select the number of the patient (first patient: 1)
    number = 1
    # select the current slice (first slice: slice0)
    im_slice = 'slice100'
    # define the penalty values for MSI
    il_const = 1  # inside level
    ol_const = 1  # outside level

    # load the data corresponding the selected patient
    dl = DataLoader(patient=number, data_downloader=ddl)

    # get the contours from the images
    points_ref = dl.c_ref[im_slice]
    points_test = dl.c_test[im_slice]

    # calculate the corresponding MSI
    msi_calc = MSICalculator(
        il=il_const, ol=ol_const,
        ref_points=points_ref, test_points=points_test)
    msi_calc.run()

    print("The value of the MSI corresponding the selected slice is ", msi_calc.msi)

    # evaluate all the slices for one patient
    evaluator = MetricsEvaluator(patient=number, data_downloader=ddl, il=il_const, ol=ol_const)
    evaluator.evaluate()

    m = []
    for i in range(len(evaluator.msindex)):
        msi_median = statistics.median(evaluator.msindex[i])
        m.append(float(msi_median))

    d = []
    for i in range(len(evaluator.dice)):
        d.append(float(evaluator.dice[i]))

    j = []
    for i in range(len(evaluator.jacc)):
        j.append(float(evaluator.jacc[i]))

    h = []
    for i in range(len(evaluator.haus)):
        h.append(float(evaluator.haus[i]))

    idx = []
    for i in range(len(evaluator.idx)):
        idx.append(int(evaluator.idx[i]))

    data = pd.DataFrame({'MSI': m, 'Dice': d, 'Jaccard': j, 'Hausdorff': h, 'index': idx})


if __name__ == '__main__':
    main()
