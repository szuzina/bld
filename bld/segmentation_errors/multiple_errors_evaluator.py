from typing import Optional
import csv
from natsort import natsorted
from pathlib import Path

from bld.data import DataDownloader
from bld.metrics import EvaluationMetrics
from .dataloader_errors import DataLoaderErrors
from .metrics_evaluator_errors import MetricsEvaluatorErrors


class MultipleErrorsEvaluator:
    """
    Calculates the different metrics for all the perturbed contours corresponding to one patient.

    Args:
        error_dir: the directory path containing the perturbed contours
        il: inside penalty level value for MSI
        ol: outside penalty level value for MSI

    Returns:
        msindex: MSI values
        idx: the slice indices for which MSI was calculated
        dice: Dice index values
        jacc: Jaccard index values
        haus: Hausdorff distance values
    """

    def __init__(self,
                 ddl: DataDownloader,
                 error_dir: str,
                 il: Optional[float] = 1, ol: Optional[float] = 1):

        self.il = il
        self.ol = ol

        self.error_dir = error_dir
        self.ddl = ddl

        self.results = self.evaluate_all_tests(dir_path=error_dir)

    def evaluate_one_test(self, test_path):
        """
        Calculate the metrics for all image slices of one test segmentation.
        """

        dl_errors = DataLoaderErrors(data_downloader=self.ddl, test_file_path=test_path)
        print('ref:', dl_errors.ref_patient)
        metrics_ev = MetricsEvaluatorErrors(dataloader_errors=dl_errors)

        evaluation_metrics = EvaluationMetrics(msi=metrics_ev.msindex,
                                               dice=metrics_ev.dice,
                                               jaccard=metrics_ev.jacc,
                                               hausdorff=metrics_ev.haus,
                                               idx=metrics_ev.idx)
        evaluation_metrics_with_zeros = EvaluationMetrics(msi=metrics_ev.msi_with_zeros,
                                                          dice=metrics_ev.dice_all_slices,
                                                          jaccard=metrics_ev.jaccard_all_slices,
                                                          hausdorff=metrics_ev.hausdorff_all_slices,
                                                          idx=metrics_ev.idx_all_slices)

        return evaluation_metrics, evaluation_metrics_with_zeros

    def evaluate_all_tests(self, dir_path):
        error_results_metrics = {}
        path_to_dir = Path(dir_path)
        for item in natsorted(path_to_dir.glob("*.nii.gz")):
            print("start", item)
            error_results_metrics[item.name] = self.evaluate_one_test(item)

        return error_results_metrics

    def save_results_as_csv(self, output_path: str) -> None:
        """Save the evaluation results to a CSV file.

        One row is created for each evaluated slice of each perturbed
        segmentation. Both metric versions, with and without zero-filled
        slices, are included.

        Parameters
        ----------
        output_path:
            Path of the output CSV file.
        """

        fieldnames = [
            "test_segmentation",
            "slice_index",
            "include_zeros",
            "msi",
            "dice",
            "jaccard",
            "hausdorff",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for test_segmentation, results in self.results.items():
                evaluation_metrics, evaluation_metrics_with_zeros = results

                # Metrics without zeros
                for idx, msi, dice, jaccard, hausdorff in zip(
                    evaluation_metrics.idx,
                    evaluation_metrics.msi,
                    evaluation_metrics.dice,
                    evaluation_metrics.jaccard,
                    evaluation_metrics.hausdorff,
                ):
                    writer.writerow({
                        "test_segmentation": test_segmentation,
                        "slice_index": idx,
                        "include_zeros": False,
                        "msi": msi,
                        "dice": dice,
                        "jaccard": jaccard,
                        "hausdorff": hausdorff,
                    })

                # Metrics with zeros
                for idx, msi, dice, jaccard, hausdorff in zip(
                    evaluation_metrics_with_zeros.idx,
                    evaluation_metrics_with_zeros.msi,
                    evaluation_metrics_with_zeros.dice,
                    evaluation_metrics_with_zeros.jaccard,
                    evaluation_metrics_with_zeros.hausdorff,
                ):
                    writer.writerow({
                        "test_segmentation": test_segmentation,
                        "slice_index": idx,
                        "include_zeros": True,
                        "msi": msi,
                        "dice": dice,
                        "jaccard": jaccard,
                        "hausdorff": hausdorff,
                    })
