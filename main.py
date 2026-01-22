from bld.data import DataDownloader, DataLoader
from bld.metrics import MSICalculator
from bld.evaluation import MetricsEvaluator

import pandas as pd
import statistics


def main():
    folder_url_ref = 'https://drive.google.com/uc?export=download&id=11oI9T_Rc0kReHvPqZlxDVwYjQ7we7CaC'
    # myoma reference masks -> 6 cases (5 fold nnUNet) JAV!
    folder_url_test = 'https://drive.google.com/uc?export=download&id=114ZIpgQ50gDrom0Sl_S9OdsL-Fau_5DB'
    # myoma test masks

    ddl = DataDownloader(ref_url=folder_url_ref, test_url=folder_url_test,
                         data_folder="data", root_folder='./')

    # select the number of the patient (first patient: 1)
    number = 1
    # select the current slice (first slice: slice0)
    im_slice = 'slice173'
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

    data = pd.DataFrame({'MSI': m, 'Dice': evaluator.dice, 'Jaccard': evaluator.jacc, 'Hausdorff': evaluator.haus,
                         'SDCD': evaluator.sdice, 'APL': evaluator.apl, 'HD95': evaluator.hd95, 'index': evaluator.idx})
    print(data.to_string())

    # the number of slices with MSI
    print('number of slices with calculated MSI score:', len(evaluator.msindex))
    # the number of nonempty slices (including the incorrect pairing resulted to MSI=0
    print('number of nonempty slices:', len(evaluator.msi_with_zeros))

    # patient-level MSI score (average)
    print('patient-level average MSI:', statistics.mean(m))
    print('patient-level average Dice:', statistics.mean(evaluator.dice))
    print('patient-level average Jaccard:', statistics.mean(evaluator.jacc))
    print('patient-level average Hausdorff:', statistics.mean(evaluator.haus))


if __name__ == '__main__':
    main()
