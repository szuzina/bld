from bld.data.data_downloader import DataDownloader
from bld.data.dataloader import DataLoader
from bld.metrics.msi_calculator import MSICalculator

from bld.evaluation.metrics_evaluator import MetricsEvaluator
from bld.data.csv_dataloader import CSVDataLoader
from bld.metrics.evaluation_metrics import EvaluationMetrics
from bld.evaluation.correlation_analyzer import CorrelationAnalyzer

import pprint


def main():
    folder_url_ref = 'https://drive.google.com/uc?export=download&id=1u2CMExEtQSi1iMclEdlr84YkgY-fd2C-'
    folder_url_test = 'https://drive.google.com/uc?export=download&id=1U4o0AhgpF9RsS6nlGeJk8kvz2nDnVwmt'
    csv_link = '1QFFfHTOHFNj2HEjX0XKbRwbB4ylw70ni'

    DataDownloader(ref_url=folder_url_ref, test_url=folder_url_test, csv_data_id=csv_link,
                   data_folder="data", root_folder='./')

    # select the number of the patient (first patient: 1)
    number = 2
    # select the current slice (first slice: slice0)
    im_slice = 'slice100'
    # define the penalty values for MSI
    IL_CONST = 1 # inside level
    OL_CONST = 1 # outside level

    # load the data corresponding the selected patient
    dl = DataLoader(data_folder="data", patient=number, root_folder='./')

    # get the contours from the images
    points_ref = dl.c_ref[im_slice]
    points_test = dl.c_test[im_slice]

    # calculate the corresponding MSI
    msi_calc = MSICalculator(
        il=IL_CONST, ol=OL_CONST,
        ref_points=points_ref, test_points=points_test)
    msi_calc.run()

    print("The value of the MSI corresponding the selected slice is ", msi_calc.msi)


    CONST = 40

    d = {}

    for i in range(1, CONST + 1, 1):
        evaluator = MetricsEvaluator(patient=i, il=1, ol=1)
        print(f"patient {i} is done")
        evaluator.evaluate()
        csvdl = CSVDataLoader(
            p_number=i,
            idx=evaluator.idx,
            root_folder='/content')

        evmet = EvaluationMetrics(
            msi=evaluator.msindex,
            hausdorff=evaluator.haus,
            dice=evaluator.dice,
            jaccard=evaluator.jacc)

        # Check if score has enough elements for correlation analysis
        score = csvdl.filtered_scores
        if len(score) >= 2 and len(score) == len(evaluator.msindex):
            analyzer = CorrelationAnalyzer(
                evaluation_metrics=evmet,
                manual_score=score)
            analyzer.run()

            # save the data
            d_ix = 'p' + str(i)
            d[d_ix] = csvdl, evmet, analyzer, evaluator
        else:
            print(f"Skipping correlation analysis for patient {i} due to insufficient data points.")


if __name__ == '__main__':
    main()
