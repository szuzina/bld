from scipy.spatial.distance import jaccard

from bld.metrics import MSICalculator, EvaluationMetrics
from bld.data import DataDownloader, CSVDataLoader, DataLoader
from bld.evaluation import MetricsEvaluator, CorrelationAnalyzer

import pandas as pd
import statistics


def main():
    # myoma 40 test cases
    #folder_url_ref = 'https://drive.google.com/uc?export=download&id=1u2CMExEtQSi1iMclEdlr84YkgY-fd2C-'
    #folder_url_test = 'https://drive.google.com/uc?export=download&id=1U4o0AhgpF9RsS6nlGeJk8kvz2nDnVwmt'

    # myoma 6 test cases
    # folder_url_ref = 'https://drive.google.com/uc?export=download&id=1KaVRqftKKNZyoMACF6t_m4gaabSr0We8'
    # folder_url_test = 'https://drive.google.com/uc?export=download&id=114ZIpgQ50gDrom0Sl_S9OdsL-Fau_5DB'

    # prostate 6 test cases
    #folder_url_ref = 'https://drive.google.com/uc?export=download&id=1jc_2-7LKX1PkJC0jpvd8uMEDfyfL7R5n'
    #folder_url_test = 'https://drive.google.com/uc?export=download&id=1rqhUyEBWo-rCo8qv6E8j5BK01ZaCn1hK'

    # myoma 26 test cases (2025.09.)
    folder_url_ref = 'https://drive.google.com/uc?export=download&id=1-0-N2WoFuTY2VFRcgS7B3m8Keneol6lj'
    folder_url_test = 'https://drive.google.com/uc?export=download&id=1ypr1BGSc0Ivm2mw-ta6wX3vLprP6RWmn'
    csv_link = '16l1-D3uKkWS12gELR2XN5C-SRxH-BhPQ'

    # the number of patients
    patient_number = 26

    ddl = DataDownloader(ref_url=folder_url_ref, test_url=folder_url_test, csv_data_id=csv_link,
                         data_folder="data", root_folder='./')

    # select the number of the patient (first patient: 1)
    number = 2
    # select the current slice (first slice: slice0)
    im_slice = 'slice130'
    # define the penalty values for MSI
    il_const = 1  # inside level
    ol_const = 10  # outside level

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

    # DO THE CORRELATION CALCULATIONS
    d = {}
    d_with_zeros = {}

    for i in range(1, patient_number+1, 1):
        evaluator = MetricsEvaluator(patient=i, data_downloader=ddl, il=il_const, ol=ol_const)
        evaluator.evaluate()

        csvdl = CSVDataLoader(
            p_number=i,
            idx=evaluator.idx,
            datadownloader=ddl)

        evmet = EvaluationMetrics(
                msi=evaluator.msindex,
                hausdorff=evaluator.haus,
                dice=evaluator.dice,
                jaccard=evaluator.jacc
            )

        score = csvdl.filtered_scores
        if len(score) >= 2 and len(score) == len(evaluator.msindex):
            analyzer = CorrelationAnalyzer(
                evaluation_metrics=evmet,
                manual_score=score)
            analyzer.run()

            # save the data
            d_ix = 'p' + str(i)
            d[d_ix] = csvdl, evmet, analyzer, evaluator
            print(f"Ccorrelation analysis for patient {i} is done (without zero MSI).")
        else:
            print(f"Skipping correlation analysis for patient {i} due to insufficient data points (without zero MSI).")

        csvdl_with_zeros = CSVDataLoader(
            p_number=i,
            idx=evaluator.idx_all_slices,
            datadownloader=ddl
        )

        evmet_with_zeros = EvaluationMetrics(
                msi=evaluator.msi_with_zeros,
                hausdorff=evaluator.hausdorff_all_slices,
                dice=evaluator.dice_all_slices,
                jaccard=evaluator.jaccard_all_slices
            )

        # Check if score has enough elements for correlation analysis
        score2 = csvdl_with_zeros.filtered_scores
        if len(score2) >= 2 and len(score2) == len(evaluator.msi_with_zeros):
            analyzer2 = CorrelationAnalyzer(
                evaluation_metrics=evmet_with_zeros,
                manual_score=score2)
            analyzer2.run()

            # save the data
            d_ix = 'p' + str(i)
            d_with_zeros[d_ix] = csvdl_with_zeros, evmet_with_zeros, analyzer2, evaluator
            print(f"Correlation analysis for patient {i} is done (with zero MSI).")
        else:
            print(f"Skipping correlation analysis for patient {i} due to insufficient data points (with zero MSI).")


if __name__ == '__main__':
    main()
