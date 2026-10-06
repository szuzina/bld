from pathlib import Path
import ast
import re

import numpy as np
import pandas as pd


class ErrorMetricsAnalyzer:
    """
    Analyze segmentation-error evaluation results stored in a CSV file.

    The expected CSV columns are:
        test_segmentation
        slice_index
        include_zeros
        msi
        dice
        jaccard
        hausdorff

    Parameters
    ----------
    results_file : str or Path
        Path to the CSV file containing the evaluation results.

    Attributes
    ----------
    df : pandas.DataFrame
        Complete results dataframe.
    df_without_zeros : pandas.DataFrame
        Results where MSI values from unchanged slices are excluded.
    df_with_zeros : pandas.DataFrame
        Results including zero MSI values.
    """

    METRIC_COLUMNS = [
        "msi",
        "dice",
        "jaccard",
        "hausdorff",
    ]

    REQUIRED_COLUMNS = [
        "test_segmentation",
        "slice_index",
        "include_zeros",
        "msi",
        "dice",
        "jaccard",
        "hausdorff",
    ]

    def __init__(self, results_file: str | Path):

        self.results_file = Path(results_file)
        if not self.results_file.is_file():
            raise FileNotFoundError(f"Results file does not exist: {self.results_file}")

        self.df = pd.read_csv(self.results_file)

        self._validate_columns()
        self._clean_data()
        self._extract_error_information()

        self.df_without_zeros = self.df[~self.df["include_zeros"]].copy()
        self.df_with_zeros = self.df[self.df["include_zeros"]].copy()

    def _validate_columns(self):
        """Check that all required columns are present."""
        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in self.df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing columns in CSV: {missing_columns}")

    def _clean_data(self):
        """Clean and convert metric columns to numeric values."""
        # MSI is currently saved as strings such as "[0.05268]"

        self.df["msi"] = self.df["msi"].apply(self._parse_msi)

        for column in ["slice_index"] + self.METRIC_COLUMNS:
            self.df[column] = pd.to_numeric(
                self.df[column],
                errors="coerce"
            )

        self.df["include_zeros"] = self.df["include_zeros"].astype(bool)

    @staticmethod
    def _parse_msi(value):
        """
        Convert MSI values such as '[0.05268]' to a float.
        Also accepts already numeric values.
        """

        if pd.isna(value):
            return np.nan

        if isinstance(value, (int, float, np.number)):
            return float(value)
        try:
            parsed = ast.literal_eval(str(value))
            if isinstance(parsed, (list, tuple)):
                if len(parsed) == 0:
                    return np.nan
                return float(parsed[0])
            return float(parsed)

        except (ValueError, SyntaxError, TypeError):
            return np.nan

    def _extract_error_information(self):
        """
        Extract error type and magnitude from the segmentation filename.

        Example:
            patient4_segmentation_error_directional_erosion_5mm.nii.gz

        becomes:
            error_type = directional_erosion
            magnitude_mm = 5
        """

        pattern = re.compile(r"segmentation_error_(.+)_(\d+(?:\.\d+)?)mm")
        extracted = self.df["test_segmentation"].str.extract(pattern)

        self.df["error_type"] = extracted[0]
        self.df["magnitude_mm"] = pd.to_numeric(extracted[1], errors="coerce")

    def get_summary(self, include_zeros=False):
        """
        Calculate descriptive statistics for all metrics.

        Parameters
        ----------
        include_zeros : bool
            If True, include slices with zero MSI.
            If False, exclude them.

        Returns
        -------
        pandas.DataFrame
            Descriptive statistics.
        """

        data = (
            self.df_with_zeros
            if include_zeros
            else self.df_without_zeros
        )

        return data[self.METRIC_COLUMNS].describe()

    def get_summary_by_segmentation(self, include_zeros=False):
        """
        Calculate mean, median and standard deviation for each perturbed segmentation.
        """

        data = (
            self.df_with_zeros
            if include_zeros
            else self.df_without_zeros
        )

        summary = (
            data
            .groupby("test_segmentation")[self.METRIC_COLUMNS]
            .agg(["mean", "median", "std", "min", "max"])
        )

        return summary

    def get_summary_by_error(self, include_zeros=False):
        """
        Calculate metric statistics grouped by error type and perturbation magnitude.
        """

        data = (
            self.df_with_zeros
            if include_zeros
            else self.df_without_zeros
        )

        summary = (
            data
            .groupby(["error_type", "magnitude_mm"])[self.METRIC_COLUMNS]
            .agg(["mean", "median", "std"])
        )

        return summary

    def get_mean_by_error(self, include_zeros=False):
        """
        Return mean metric values for every error type and magnitude.
        """

        data = (
            self.df_with_zeros
            if include_zeros
            else self.df_without_zeros
        )

        return (
            data
            .groupby(["error_type", "magnitude_mm"])[self.METRIC_COLUMNS]
            .mean()
            .reset_index()
        )

    def get_correlation(self, include_zeros=False, method="spearman"):
        """
        Calculate correlations between MSI and traditional metrics.

        Parameters
        ----------
        include_zeros : bool
            Whether zero-MSI slices should be included.

        method : str
            Correlation method: 'pearson', 'spearman', or 'kendall'.

        Returns
        -------
        pandas.DataFrame
            Correlation matrix.
        """

        data = (
            self.df_with_zeros
            if include_zeros
            else self.df_without_zeros
        )

        return data[self.METRIC_COLUMNS].corr(method=method)

    def get_msi_correlation(self, include_zeros=False):
        """
        Return correlation of MSI with Dice, Jaccard and Hausdorff.
        """

        correlation = self.get_correlation(include_zeros=include_zeros, method="spearman")

        return correlation.loc["msi", ["dice", "jaccard", "hausdorff"]]

    def get_correlation_by_error(self, include_zeros=False):
        """
        Calculate Spearman correlation between MSI and each traditional metric separately for every error type.
        """

        data = (
            self.df_with_zeros
            if include_zeros
            else self.df_without_zeros
        )

        results = []
        for error_type, group in data.groupby("error_type"):
            row = {"error_type": error_type, "n": len(group)}
            for metric in ["dice", "jaccard", "hausdorff"]:
                if len(group) >= 2:
                    row[f"msi_{metric}_spearman"] = (group["msi"].corr(group[metric], method="spearman"))
                else:
                    row[f"msi_{metric}_spearman"] = np.nan
            results.append(row)

        return pd.DataFrame(results)

    def get_slice_results(self, test_segmentation: str, include_zeros=False):
        """
        Return all slice-level results for one segmentation.
        """

        data = (
            self.df_with_zeros
            if include_zeros
            else self.df_without_zeros
        )

        return data[data["test_segmentation"] == test_segmentation].copy()

    def get_error_type_results(self, error_type: str, include_zeros=False):
        """
        Return all results belonging to one error type.
        """

        data = (
            self.df_with_zeros
            if include_zeros
            else self.df_without_zeros
        )

        return data[data["error_type"] == error_type].copy()

    def save_summary(self, output_file: str | Path, include_zeros=False):
        """
        Save the mean metric values grouped by error type and magnitude to CSV.
        """

        output_file = Path(output_file)
        summary = self.get_mean_by_error(include_zeros=include_zeros)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_file, index=False)

    def save_correlation(self, output_file: str | Path, include_zeros=False):
        """
        Save the correlation matrix to CSV.
        """

        output_file = Path(output_file)
        correlation = self.get_correlation(include_zeros=include_zeros)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        correlation.to_csv(output_file)
