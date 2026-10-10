from bld.data import DataDownloader, DataLoader
from bld.segmentation_errors import SegmentationErrorPatient
from bld.segmentation_errors import CreateVisualization
from bld.segmentation_errors import ErrorMetricsAnalyzer
from bld.segmentation_errors import MultipleErrorsEvaluator

ERROR_PATH = "/home/fazekas/PycharmProjects/MSI/data/segmentation_error_masks"
OUTPUT_PATH = "/home/fazekas/PycharmProjects/MSI/data/segmentation_error_masks/metrics.csv"
RESULT_FILE = "/home/fazekas/PycharmProjects/MSI/data/segmentation_error_masks/metrics.csv"


def main():
    # myoma 40 test cases
    # folder_url_ref = 'https://drive.google.com/uc?export=download&id=1u2CMExEtQSi1iMclEdlr84YkgY-fd2C-'
    # folder_url_test = 'https://drive.google.com/uc?export=download&id=1U4o0AhgpF9RsS6nlGeJk8kvz2nDnVwmt'

    # myoma 6 test cases
    # folder_url_ref = 'https://drive.google.com/uc?export=download&id=1KaVRqftKKNZyoMACF6t_m4gaabSr0We8'
    # folder_url_test = 'https://drive.google.com/uc?export=download&id=114ZIpgQ50gDrom0Sl_S9OdsL-Fau_5DB'

    # prostate 6 test cases
    # folder_url_ref = 'https://drive.google.com/uc?export=download&id=1jc_2-7LKX1PkJC0jpvd8uMEDfyfL7R5n'
    # folder_url_test = 'https://drive.google.com/uc?export=download&id=1rqhUyEBWo-rCo8qv6E8j5BK01ZaCn1hK'

    # myoma 26 test cases (2025.09.)
    folder_url_ref = 'https://drive.google.com/uc?export=download&id=1-0-N2WoFuTY2VFRcgS7B3m8Keneol6lj'
    folder_url_test = 'https://drive.google.com/uc?export=download&id=1ypr1BGSc0Ivm2mw-ta6wX3vLprP6RWmn'
    csv_link = '1wKgNBsnbCTSlyLNkP-8ElyGXAniujG4l'

    # pancreas cysts new scoring (4 patients)
    # folder_url_ref = 'https://drive.google.com/uc?export=download&id=1gMLWCnHnm8TFqVJfGJwdf2OHTstGdyMS'
    # folder_url_test = 'https://drive.google.com/uc?export=download&id=1dcDD-nZnFvPAxH4hRe6wgM4N2qqJDOXC'
    # csv_link = '1nQvEUEoAE8O73rNE4GLoESs9XUG-tD8u'

    ddl = DataDownloader(ref_url=folder_url_ref, test_url=folder_url_test, csv_data_id=csv_link,
                         data_folder="data", root_folder='./')

# ------------------------------------------------------------------------------------------------
#    ERROR GENERATION
# ------------------------------------------------------------------------------------------------

    # all patients
    patient_number = 26
    # select the number of the patient (first patient: 1)
    number = 1
    # load the data corresponding the selected patient
    dl = DataLoader(patient=number, data_downloader=ddl)

# ----------------------------------------
# CREATE A SEGMENTATION ERROR AND VISUALIZE

    create_one_error = False

    if create_one_error:

        segm_error_patient = SegmentationErrorPatient(dl=dl, error_type="expansion", magnitude_mm=1)
#        segm_error_patient.save_masks_as_nifti()
#        print(segm_error_patient.metadata)

        im_slice = 'slice115'

        viz = CreateVisualization(segmentations=segm_error_patient.results[im_slice],
                                  original_mask=dl.mask_ref[im_slice])
        viz.show_comparison()
        viz.show_contour_overlay()

# ---------------------------------------
# CREATE DIFFERENT ERRORS FOR ONE PATIENT

    create_errors = False

    if create_errors:

        for i in [1, 3, 5, 10]:
            error_type_list = ["expansion",
                               "erosion",
                               "directional_expansion",
                               "directional_erosion",
                               "translation",
                               "random_boundary"]
            for error_item in error_type_list:
                segm_e_p = SegmentationErrorPatient(dl=dl, error_type=error_item, magnitude_mm=i)
                segm_e_p.save_masks_as_nifti()

# ---------------------------------------
# CREATE DIFFERENT ERRORS FOR ALL PATIENTS

    create_errors_for_all = False

    if create_errors_for_all:
        for n in range(1, patient_number+1):  # first patient is n.o. 1.
            dl_for_all_errors = DataLoader(patient=n, data_downloader=ddl)
            # create the original test mask with expansion=0 mm to have all the data together
            segm_original_test = SegmentationErrorPatient(dl=dl_for_all_errors, error_type="expansion", magnitude_mm=0)
            segm_original_test.save_masks_as_nifti()
            # create the errors
            for i in [0, 1, 3, 5, 10]:
                error_type_list = ["expansion",
                                   "erosion",
                                   "directional_expansion",
                                   "directional_erosion",
                                   "translation",
                                   "random_boundary"]
                for error_item in error_type_list:
                    segm_e_p = SegmentationErrorPatient(dl=dl_for_all_errors, error_type=error_item, magnitude_mm=i)
                    segm_e_p.save_masks_as_nifti()


# ------------------------------------------------------------------------------------------------
#    EVALUATION
# ------------------------------------------------------------------------------------------------

# CALCULATE METRICS

    calculate_metrics = True

    il_const = 10  # inside level
    ol_const = 1  # outside level

    if calculate_metrics:

        evaluator = MultipleErrorsEvaluator(ddl=ddl,
                                            error_dir=ERROR_PATH,
                                            il=il_const, ol=ol_const)
        evaluator.save_results_as_csv(
            output_path=OUTPUT_PATH
        )

# -----------------------------------
# ANALYZE RESULTS

    analyze_result = False

    if analyze_result:
        analyzer = ErrorMetricsAnalyzer(
            results_file=RESULT_FILE
        )

        summary = analyzer.get_summary()
        print(summary)

        mean = analyzer.get_mean_by_error()
        print(mean)

        correlations = analyzer.get_correlation_by_error()
        print(correlations)


if __name__ == '__main__':
    main()
