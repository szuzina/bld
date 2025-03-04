from bld.data import DataDownloader, CSVDataLoader, DataLoader
from bld.evaluation import MetricsEvaluator, CorrelationAnalyzer
from bld.metrics import MSICalculator, EvaluationMetrics


def main():
    folder_url_ref = 'https://drive.google.com/uc?export=download&id=17vcCxmTjumMgXPK7y_4uDe-pT0tnQOQc' # ez a prostata
    folder_url_test = 'https://drive.google.com/uc?export=download&id=17vcCxmTjumMgXPK7y_4uDe-pT0tnQOQc' # ez még nincs meg!!
    csv_link = '1QFFfHTOHFNj2HEjX0XKbRwbB4ylw70ni' # ez szintén még nem a prosztata!!!!

    ddl = DataDownloader(ref_url=folder_url_ref, test_url=folder_url_test, csv_data_id=csv_link,
                         data_folder="data", root_folder='./')

    # select the number of the patient (first patient: 1)
    number = 2
    # select the current slice (first slice: slice0)
    im_slice = 'slice10'
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

    cons = 85

    for i in range(1, cons + 1, 1):
        evaluator = MetricsEvaluator(patient=i, datadownloader=ddl, il=1, ol=1)
        print(f"patient {i} is done")
        evaluator.evaluate()


if __name__ == '__main__':
    main()
