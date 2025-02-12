from bld.data import DataDownloader, CSVDataLoader, DataLoader
from bld.evaluation import MetricsEvaluator, CorrelationAnalyzer
from bld.metrics import MSICalculator, EvaluationMetrics


def main():
    folder_url_ref = 'https://drive.google.com/uc?export=download&id=1u2CMExEtQSi1iMclEdlr84YkgY-fd2C-'
    folder_url_test = 'https://drive.google.com/uc?export=download&id=1U4o0AhgpF9RsS6nlGeJk8kvz2nDnVwmt'
    csv_link = '1QFFfHTOHFNj2HEjX0XKbRwbB4ylw70ni'

    ddl = DataDownloader(ref_url=folder_url_ref, test_url=folder_url_test, csv_data_id=csv_link,
                         data_folder="data", root_folder='./')

    # select the number of the patient (first patient: 1)
    number = 2
    # select the current slice (first slice: slice0)
    im_slice = 'slice100'
    # define the penalty values for MSI
    il_const = 1  # inside level
    ol_const = 1  # outside level

    # load the data corresponding the selected patient
    dl = DataLoader(patient=number, datadownloader=ddl)

    # get the contours from the images
    points_ref = dl.c_ref[im_slice]
    points_test = dl.c_test[im_slice]

    # calculate the corresponding MSI
    msi_calc = MSICalculator(
        il=il_const, ol=ol_const,
        ref_points=points_ref, test_points=points_test)
    msi_calc.run()

    print("The value of the MSI corresponding the selected slice is ", msi_calc.msi)

    cons = 40

    d = {}
    dwithzeros = {}

    for i in range(1, cons + 1, 1):
        evaluator = MetricsEvaluator(patient=i, datadownloader=ddl, il=1, ol=1)
        print(f"patient {i} is done")
        evaluator.evaluate()
        csvdl = CSVDataLoader(
            p_number=i,
            idx=evaluator.idx,
            datadownloader=ddl)

        csvdl_with_zeros = CSVDataLoader(
            p_number=i,
            idx=evaluator.idxallslices,
            datadownloader=ddl
        )

        evmet = EvaluationMetrics(
            msi=evaluator.msindex,
            hausdorff=evaluator.haus,
            dice=evaluator.dice,
            jaccard=evaluator.jacc)

        evmet_with_zeros = EvaluationMetrics(
            msi=evaluator.msiwithzeros,
            hausdorff=evaluator.hausdorffallslices,
            jaccard=evaluator.jaccardallslices,
            dice=evaluator.diceallslices
        )

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
            print(f"Skipping correlation analysis for patient {i} due to insufficient data points (without zero MSI).")

        # Check if score has enough elements for correlation analysis
        score2 = csvdl_with_zeros.filtered_scores
        if len(score2) >= 2 and len(score2) == len(evaluator.msiwithzeros):
            analyzer2 = CorrelationAnalyzer(
                evaluation_metrics=evmet_with_zeros,
                manual_score=score2)
            analyzer2.run()

            # save the data
            d_ix = 'p' + str(i)
            dwithzeros[d_ix] = csvdl_with_zeros, evmet_with_zeros, analyzer2, evaluator
        else:
            print(f"Skipping correlation analysis for patient {i} due to insufficient data points (with zero MSI).")


if __name__ == '__main__':
    main()
