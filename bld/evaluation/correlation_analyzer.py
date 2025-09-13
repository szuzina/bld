import numpy as np
from scipy.stats import spearmanr, pearsonr

from bld.metrics import EvaluationMetrics


class CorrelationAnalyzer:
    """
    Calculate the correlation between manual scores, MSI and traditional metrics.

    Args:
        evaluation_metrics: EvaluationMetrics class
        manual_score: manual scores loaded previously
        method: define the method of the aggregation of MSI (if there are more than one contour, we have to aggregate
            the contour MSI values into one slice MSI value

    Returns:
         results: dictionary with the correlation values
    """

    def __init__(self, evaluation_metrics: EvaluationMetrics, manual_score: list, method="median"):
        self.metrics = evaluation_metrics
        self.manual_score = manual_score
        self.aggregated: list = []
        self.method = method

        self.results: dict = dict()

    def calculate_aggregation(self):
        """Calculate the median for each element in the msi list."""
        aggr = []
        for i in range(len(self.metrics.msi)):
            if self.method == "median":
                aggr.append(np.median(self.metrics.msi[i]))
            if self.method == "min":
                aggr.append(np.min(self.metrics.msi[i]))
            if self.method == "max":
                aggr.append(np.max(self.metrics.msi[i]))
        self.aggregated = aggr
        return 0

    def find_spearman_correlation_with_metrics(self):
        """Find the Spearman correlation between the median and the other metrics."""
        self.results.update({
            "spearman_msi_hausdorff": spearmanr(self.aggregated, self.metrics.hausdorff),
            "spearman_msi_dice": spearmanr(self.aggregated, self.metrics.dice),
            "spearman_msi_jaccard": spearmanr(self.aggregated, self.metrics.jaccard),
            "spearman_msi_manual": spearmanr(self.aggregated, self.manual_score)
        })

    def find_pearson_correlation_with_metrics(self):
        """Find the Pearson correlation between the median and traditional metrics."""
        self.results.update({
            "pearson_msi_hausdorff": pearsonr(self.aggregated, self.metrics.hausdorff),
            "pearson_msi_dice": pearsonr(self.aggregated, self.metrics.dice),
            "pearson_msi_jaccard": pearsonr(self.aggregated, self.metrics.jaccard),
            "pearson_msi_manual": pearsonr(self.aggregated, self.manual_score)
        })

    def find_spearman_pearson_traditional_with_manual_score(self):
        """Find both Spearman and Pearson correlations between a traditional metric and manual score."""
        self.results.update({
            "spearman_hausdorff_manual": spearmanr(self.metrics.hausdorff, self.manual_score),
            "pearson_hausdorff_manual": pearsonr(self.metrics.hausdorff, self.manual_score),
            "spearman_dice_manual": spearmanr(self.metrics.dice, self.manual_score),
            "pearson_dice_manual": pearsonr(self.metrics.dice, self.manual_score),
            "spearman_jaccard_manual": spearmanr(self.metrics.jaccard, self.manual_score),
            "pearson_jaccard_manual": pearsonr(self.metrics.jaccard, self.manual_score)
        })

    def run(self):
        self.calculate_aggregation()
        self.find_spearman_correlation_with_metrics()
        self.find_pearson_correlation_with_metrics()
        self.find_spearman_pearson_traditional_with_manual_score()
